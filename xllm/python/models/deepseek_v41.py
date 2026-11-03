# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4.1 Flash Python model (CED + CSA2, TORCH backend).

Faithful port of the official reference implementation
(``/weights/DeepSeek/DeepSeek-V4.1-Flash/inference/model.py``; semantics
spec: ``docs/deepseek_v41_reference_semantics.md``). The forward is pure
torch and NPU-only (there is no CPU/CUDA attention backend); the reference
dequant numerics are preserved by the inlined helpers at the bottom of
this file (device-agnostic, unit-tested on CPU; ue8m0 decode, FP8 block
dequant, MXFP4 dequant):

  * ``CSA2LayerPlan`` -- static Full/Reindex/Reuse/SWA assignment. Every
    kv-source layer compresses its OWN attention input (there is no
    cross-layer CED hidden capture in the reference); the group's Reindex /
    Reuse layers read that source's caches (``main_kv_source_layer``).
  * ``DeepseekV41Attention`` -- latent Q/KV projections + interleaved-pair
    RoPE, the ratio-1/ratio-2 compressor weights, the CSA2 indexer weights
    (``wk`` acts on the 512-d latent, not the hidden), and the grouped
    ``wo_a``/``wo_b`` output projection with output de-rotation.
  * ``Csa2AttentionBackend`` (``xllm/python/attention/csa2_attention.py``)
    owns the cache orchestration: SWA ring, state parking/pooling, index
    writes, scoring, candidate pool, top-k and the sink softmax.
  * ``DeepseekV41HyperConnection`` -- single-pass Mega-mHC with the
    Sinkhorn split; attention collapses with the previous block's
    ``ffn_pre``, the FFN with this block's ``attn_pre``, and the final
    merge is the learned ``hc_pre(h, last_ffn_pre)`` weighted sum.
  * ``DeepseekV41MoE`` -- fp32 sqrtsoftplus routing with bias-steered
    selection (``gate.bias`` / per-token ``gate.bias_vl``), clamped SwiGLU
    experts and one shared expert.
  * ``DeepseekV41ForCausalLM`` -- checkpoint-mode-aware construction and
    loading: ``fp8`` (official FP8-block/MXFP4 release, dequantized to bf16
    at load), ``ascend`` (W8A8 export, existing kernel path) and ``none``.

Vision and the DSpark draft model are out of scope here; their checkpoint
tensors are simply not consumed. Engram IS implemented
(:mod:`deepseek_v41_engram`): the layers listed in ``engram_layer_ids``
carry the gated n-gram lookup injected into the mHC residual stream, with
the tokenizer-derived hash state loaded from the
``engram_hash_state.bin`` sidecar.
"""

from __future__ import annotations

import json
import os
import socket
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.distributed import (
    engram_all_gather_object,
    engram_barrier,
    engram_dp_shard_size,
    init_engram_dp_group,
    world_topology,
)
from xllm.python.layers.attention import Attention
from xllm.python.layers.embedding import HiddenParallelEmbedding
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear
from xllm.python.model_executor.forward_context import (
    capturing_acl_graph,
    get_execution_buffer,
    get_forward_context,
    get_forward_context_or_none,
    in_acl_graph,
    record_layer_event,
    wait_for_layer_load,
)
from xllm.python.model_executor.runners.base import GraphInputPreparation
from xllm.python.model_loader.module_loaders import load_w8a8_dynamic_projection
from xllm.python.models.aux_hidden_capture import AuxHiddenCapture
from xllm.python.models.base import PyModelBase
from xllm.python.models.deepseek_v4 import (
    DeepseekV4Config,
    DeepseekV4ForCausalLM,
    DeepseekV4MoE,
    DeepseekV4RotaryEmbedding,
    _find_checkpoint_prefix,
    _pick,
)
from xllm.python.models.deepseek_v32 import DeepseekV3MLP, W8A8DynamicLinear
from xllm.python.models.deepseek_v41_engram import (
    ENGRAM_STORAGE_FP8,
    ENGRAM_STORAGE_INT8,
    Engram,
    EngramEmbedding,
    EngramHashState,
    EngramLayout,
    engram_offload_warning_fields,
    image_sentinel_mask,
    load_quarot_gate_unrotate,
    stage_offload_rows,
)
from xllm.python.models.engram_config import EngramConfig
from xllm.python.models.engram_shared_memory import ensure_dev_shm_capacity, estimate_shared_bytes
from xllm.python.models.multimodal import (
    EncoderOutputs,
    EncoderTokenCounts,
    MultimodalData,
    MultimodalInputs,
    merge_multimodal_embeddings,
)
from xllm.python.models.swa_bounded_replay import (
    SWA_BOUNDED_FIRST_CONSUMER,
    run_swa_bounded_consumers,
    should_run_swa_bounded_replay,
    validate_swa_bounded_replay,
)
from xllm.python.models.weight_utils import W8A8WeightLoader

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict

    from xllm.python.attention.csa2_attention import Csa2LayerContext

# ---------------------------------------------------------------------------
# CSA2 layer plan (Full / Reindex / Reuse / SWA)
# ---------------------------------------------------------------------------

CSA2_MODE_SWA = "swa"
CSA2_MODE_FULL = "full"
CSA2_MODE_REINDEX = "reindex"
CSA2_MODE_REUSE = "reuse"


@dataclass(frozen=True)
class CSA2LayerMode:
    """Static CSA2 assignment for one layer."""

    layer_id: int
    mode: str
    # Main-KV compression ratio: 2 for encoder CSA2 layers, 1 for decoder
    # main-KV layers, 0 for pure SWA layers.
    compress_ratio: int
    # Layer whose main-KV / index caches this layer reads (-1 for SWA layers,
    # self for Full layers, the group's kv source for Reindex/Reuse layers).
    main_kv_source_layer: int

    @property
    def is_csa2(self) -> bool:
        return self.mode != CSA2_MODE_SWA

    @property
    def is_full(self) -> bool:
        return self.mode == CSA2_MODE_FULL

    @property
    def is_reindex(self) -> bool:
        return self.mode == CSA2_MODE_REINDEX

    @property
    def is_reuse(self) -> bool:
        return self.mode == CSA2_MODE_REUSE


class CSA2LayerPlan:
    """Static per-layer CSA2 assignment, built once at model init.

    Derivation (reference model.py:653-661): ``kv_source_layer_ids`` are the
    Full layers that own a Compressor and publish the group's compressed KV
    and index K; ``index_source_layer_ids`` additionally lists the Reindex
    layers that re-score the shared K with their own indexer Q; every other
    layer after the nearest preceding kv source reuses the shared main KV and
    top-k. Layers before the first kv source (or with compress ratio 0) are
    pure SWA. For the reference config (``kv=[2,8,14,20]``,
    ``index=[2,8,14,20,24,28,32,36]``) this reproduces the documented
    assignment: layers 0-1 SWA, 2-19 encoder groups (Full at 2/8/14, Reuse
    otherwise), 20-39 decoder group (Full at 20, Reindex at 24/28/32/36,
    Reuse otherwise).

    There is NO cross-layer hidden capture: every kv-source layer compresses
    its own attention input (semantics doc section 3).
    """

    def __init__(self, modes: list[CSA2LayerMode]) -> None:
        self._modes = modes

    def mode(self, layer_id: int) -> CSA2LayerMode:
        if layer_id < 0 or layer_id >= len(self._modes):
            raise IndexError(f"layer {layer_id} outside the CSA2 plan's {len(self._modes)} layers")
        return self._modes[layer_id]

    def __len__(self) -> int:
        return len(self._modes)

    @classmethod
    def build(
        cls,
        n_layers: int,
        compress_ratios: Sequence[int],
        kv_source_layer_ids: Sequence[int],
        index_source_layer_ids: Sequence[int],
    ) -> CSA2LayerPlan:
        """Validate the config-derived inputs and derive the per-layer modes."""
        kv_sources = sorted({int(source) for source in kv_source_layer_ids})
        index_sources = sorted({int(source) for source in index_source_layer_ids})
        kv_source_set = set(kv_sources)
        index_source_set = set(index_sources)
        for source in kv_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"kv source layer {source} is outside the model's {n_layers} layers")
        for source in index_sources:
            if not 0 <= source < n_layers:
                raise ValueError(f"index source layer {source} is outside the model's {n_layers} layers")
        non_indexed = [source for source in kv_sources if source not in index_source_set]
        if non_indexed:
            raise ValueError(f"kv source layers must also be index sources: {non_indexed}")
        for source in index_sources:
            if not any(kv_source <= source for kv_source in kv_sources):
                raise ValueError(f"index source layer {source} has no preceding kv source layer")
        ratios = [int(ratio) for ratio in compress_ratios]
        for source in kv_sources:
            if source >= len(ratios) or ratios[source] <= 0:
                raise ValueError(f"kv source layer {source} must have a positive compress ratio")

        modes: list[CSA2LayerMode] = []
        for layer_id in range(n_layers):
            ratio = ratios[layer_id] if layer_id < len(ratios) else 0
            kv_source = max((source for source in kv_sources if source <= layer_id), default=-1)
            if ratio <= 0 or kv_source < 0:
                mode = CSA2_MODE_SWA
                main_kv_source = -1
            elif layer_id in kv_source_set:
                mode = CSA2_MODE_FULL
                main_kv_source = layer_id
            elif layer_id in index_source_set:
                mode = CSA2_MODE_REINDEX
                main_kv_source = kv_source
            else:
                mode = CSA2_MODE_REUSE
                main_kv_source = kv_source
            modes.append(
                CSA2LayerMode(
                    layer_id=layer_id,
                    mode=mode,
                    compress_ratio=max(ratio, 0),
                    main_kv_source_layer=main_kv_source,
                )
            )
        return cls(modes)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _int_list(d: dict, key: str, default: list[int] | None = None) -> list[int]:
    value = d.get(key)
    if value is None:
        return list(default or [])
    return [int(item) for item in value]


def _parse_v41_compress_ratios(raw: Sequence[int], n_layers: int) -> list[int]:
    """Parse V4.1 ``compress_ratios`` preserving ratio 0 / 1 / 2.

    V4 normalized every ``ratio <= 1`` to 1, which erases V4.1's distinction
    between pure-SWA layers (ratio 0) and decoder main-KV layers (ratio 1).
    The checkpoint config carries ``n_layers + num_nextn_predict_layers``
    entries (trailing zeros for the DSpark draft blocks); the main model
    consumes the first ``n_layers``. Missing trailing entries pad with 0
    (pure SWA) so no cache group is fabricated.
    """
    ratios = [int(ratio) for ratio in raw]
    for ratio in ratios:
        if ratio not in (0, 1, 2, 4):
            raise ValueError(f"unsupported DeepSeek-V4.1 compression ratio: {ratio}")
    if len(ratios) < n_layers:
        ratios.extend([0] * (n_layers - len(ratios)))
    return ratios[:n_layers]


# Defaults for config fields whose V4 values differ; injected only when the
# config omits them so an explicit value always wins.
_V41_FIELD_DEFAULTS: dict[str, Any] = {
    "model_type": "deepseek_v41",
    "hidden_size": 5120,
    "q_lora_rank": 1280,
    "n_routed_experts": 384,
    "moe_intermediate_size": 2304,
    "index_n_heads": 32,
    "rms_norm_eps": 1e-20,
}

# Deadline for the node-local Engram shared-table load fence. The Gloo sharing
# group keeps the 5-minute timeout ``new_group`` gave it; this shorter deadline
# only bounds the monitor's (group rank 0) wait at the one post-load rendezvous,
# turning a crashed or hung peer into a 30s failure on rank 0 (see
# ``collectives.engram_barrier``). PyTorch's ``monitored_barrier`` enforces the
# per-call deadline on rank 0 only: a non-zero survivor rank whose peer never
# reaches the fence still blocks until the group's own 5-minute timeout. It is a
# candidate value: the real deadline behavior still has to be measured on the
# NPU host.
_ENGRAM_SHARED_LOAD_BARRIER_TIMEOUT = timedelta(seconds=30)


@dataclass
class DeepseekV41Config(DeepseekV4Config):
    """DeepSeek-V4.1 Flash model config (V4 fields + CSA2/CED/Engram/DSpark)."""

    model_type: str = "deepseek_v41"
    hidden_size: int = 5120
    n_layers: int = 40
    q_lora_rank: int = 1280
    rms_norm_eps: float = 1e-20
    n_routed_experts: int = 384
    moe_intermediate_size: int = 2304
    index_n_heads: int = 32
    # -- CSA2 / CED --
    # candidate_source_layer_id / image_token_id default to the C++ "disabled"
    # sentinels (LOAD_ARG_OR in xllm/models/llm/deepseek_v41.h) so a config that
    # omits them does not silently activate a candidate pool or fabricate image
    # token matches. The released config always carries the real values.
    kv_source_layer_ids: list[int] = field(default_factory=list)
    index_source_layer_ids: list[int] = field(default_factory=list)
    candidate_source_layer_id: int = -1
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    # -- Engram (gated n-gram lookup injected on engram_layer_ids) --
    engram_layer_ids: list[int] = field(default_factory=list)
    engram_num_embeddings: list[int] = field(default_factory=list)
    engram_max_ngram_size: int = 4
    engram_vocab_size: int = 16000000
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_compressed_vocab_size: int = 99092
    # -1 is the "unset" sentinel so DeepseekV41Model.__init__ can reject an
    # enabled engram with no explicit pad id (guard: `engram_pad_token_id < 0`).
    # C++ defaults to the same -1 (deepseek_v41.h / model_args.h), so the
    # sentinel is not overwritten on the service path.
    engram_pad_token_id: int = -1
    # -- DSpark draft geometry (consumed by the draft model, Phase 3) --
    num_nextn_predict_layers: int = 3
    dspark_target_layer_ids: list[int] = field(default_factory=lambda: [37, 38, 39])
    dspark_block_size: int = 5
    dspark_markov_rank: int = 256
    dspark_noise_token_id: int = 128799
    dspark_n_routed_experts: int = 128
    dspark_num_experts_per_tok: int = 3
    enable_dsa_cp: bool = False
    # Explicit runtime opt-in: approximate prefill, default full computation.
    enable_swa_bounded_replay: bool = False
    # ``--enable_fused_mc2`` flag (PyCausalLM::build_config_dict bridges the
    # C++ ``enable_fused_mc2`` gflag - kernel_config.cpp: -1 auto / 0 off /
    # 1 DFC / 2 MegaMoe - as >0). Decode-only slice: the dispatch/combine
    # collective pair only pays off at small token counts, so prefill and
    # chunked prefill stay on the portable path.
    enable_fused_mc2: bool = False
    # -- Aux-hidden capture (DSpark target side) --
    # 0-based layer ids whose residual stream the target buffers for the draft's
    # ``main_proj``; the C++ target reads the draft config's
    # ``dspark_target_layer_ids`` and reflects it here. Empty for plain serving,
    # so capture is a no-op unless speculative decoding is on.
    layers_to_capture: list[int] = field(default_factory=list)
    # Graph-mode aux-capture opt-in, bridged from the C++
    # ``--enable_graph_aux_hidden_states`` flag (PyCausalLM::build_config_dict).
    # When set, the capture buffer is persistent so the decode graph runner can
    # capture and replay it; the eager paths publish the capture out of band
    # (executor.execute appends ``aux_hidden_capture.buffer`` to the output).
    # Only meaningful together with a non-empty ``layers_to_capture``.
    enable_graph_aux_hidden_states: bool = False
    # -- Vision (Phase 4); image_token_id also drives gate.bias_vl --
    # -1 = no image tokens (text-only), matching the C++ loader default.
    image_token_id: int = -1
    # -- Engram host-offload switches (from --engram-config) --
    # Parsed strictly from the raw ``engram_config`` JSON string by
    # EngramConfig.from_value; {} (the C++ default) disables both switches.
    engram_config: EngramConfig = field(default_factory=EngramConfig)
    # -- Parallel / weight-format contract --
    # Passed in by the C++ executor (py_causal_lm.cpp). Engram currently
    # requires PP=1 and DCP=1 and is only validated against safetensors/dummy
    # weights; the values are surfaced here so config validation and the
    # Python executor read them from one place.
    pp_size: int = 1
    dcp_size: int = 1
    load_format: str = "safetensors"

    @property
    def uses_mx_formats(self) -> bool:
        """V4.1 never takes the V4 MXFP8/MXFP4 execution path.

        V4.1 reuses the V4 machinery (``DeepseekV4MoE`` and the shared
        loaders) but serves official FP8 checkpoints through its own
        load-time dequantization (``V41_QUANT_FP8``), and its dense layers
        are fixed-format. A checkpoint with ``weight_quant_method == "fp8"``
        must therefore NOT enable the V4 MXFP modules here -- that would
        build a half-format model (MXFP4 experts beside non-MXFP dense
        projections).
        """
        return False

    @classmethod
    def from_dict(cls, d: dict) -> DeepseekV41Config:
        merged = dict(d)
        for key, value in _V41_FIELD_DEFAULTS.items():
            if merged.get(key) is None:
                merged[key] = value
        # The layer count has two aliases; only inject when both are absent
        # so an explicit value of either alias always wins.
        if merged.get("num_hidden_layers") is None and merged.get("n_layers") is None:
            merged["num_hidden_layers"] = 40
        cfg = super().from_dict(merged)
        cfg = replace(
            cfg,
            # Re-parse: V4's from_dict normalized "ratio <= 1 -> 1".
            compress_ratios=_parse_v41_compress_ratios(list(merged.get("compress_ratios") or []), cfg.n_layers),
            # V4.1 removed hash routing (gate.tid2eid); the field stays for
            # the inherited V4 machinery.
            n_hash_layers=0,
            kv_source_layer_ids=_int_list(merged, "kv_source_layer_ids"),
            index_source_layer_ids=_int_list(merged, "index_source_layer_ids"),
            candidate_source_layer_id=int(_pick(merged, "candidate_source_layer_id", default=-1)),
            candidate_topk_blocks=int(_pick(merged, "candidate_topk_blocks", default=2048)),
            candidate_block_size=int(_pick(merged, "candidate_block_size", default=8)),
            engram_layer_ids=_int_list(merged, "engram_layer_ids"),
            engram_num_embeddings=_int_list(merged, "engram_num_embeddings"),
            engram_max_ngram_size=int(_pick(merged, "engram_max_ngram_size", default=4)),
            engram_vocab_size=int(_pick(merged, "engram_vocab_size", default=16000000)),
            engram_n_heads=int(_pick(merged, "engram_n_heads", default=8)),
            engram_head_dim=int(_pick(merged, "engram_head_dim", default=256)),
            engram_compressed_vocab_size=int(_pick(merged, "engram_compressed_vocab_size", default=99092)),
            engram_pad_token_id=int(_pick(merged, "engram_pad_token_id", default=-1)),
            num_nextn_predict_layers=int(_pick(merged, "num_nextn_predict_layers", default=3)),
            dspark_target_layer_ids=_int_list(merged, "dspark_target_layer_ids", default=[37, 38, 39]),
            dspark_block_size=int(_pick(merged, "dspark_block_size", default=5)),
            dspark_markov_rank=int(_pick(merged, "dspark_markov_rank", default=256)),
            dspark_noise_token_id=int(_pick(merged, "dspark_noise_token_id", default=128799)),
            dspark_n_routed_experts=int(_pick(merged, "dspark_n_routed_experts", default=128)),
            dspark_num_experts_per_tok=int(_pick(merged, "dspark_num_experts_per_tok", default=3)),
            enable_dsa_cp=bool(_pick(merged, "enable_dsa_cp", default=False)),
            enable_swa_bounded_replay=bool(_pick(merged, "enable_swa_bounded_replay", default=False)),
            enable_fused_mc2=bool(_pick(merged, "enable_fused_mc2", default=False)),
            layers_to_capture=_int_list(merged, "layers_to_capture"),
            enable_graph_aux_hidden_states=bool(_pick(merged, "enable_graph_aux_hidden_states", default=False)),
            image_token_id=int(_pick(merged, "image_token_id", default=-1)),
            engram_config=EngramConfig.from_value(merged.get("engram_config", {})),
            pp_size=int(_pick(merged, "pp_size", default=1)),
            dcp_size=int(_pick(merged, "dcp_size", default=1)),
            load_format=str(_pick(merged, "load_format", default="safetensors")),
        )
        _validate_engram_config(cfg)
        return cfg


def _validate_engram_config(cfg: DeepseekV41Config) -> None:
    """Reject engram configs the TP-sharding / weight-load contract cannot serve.

    Only checked when the checkpoint actually carries engram layers; a config
    with an empty ``engram_layer_ids`` ignores the switches entirely, so the
    validation never blocks non-engram deployments.
    """
    if not cfg.engram_layer_ids:
        return
    # ``{1, 2, 4, 8}`` is the TP set this model's current configuration and
    # Engram deployment path support, deliberately stricter than the
    # uniform-head-shard divisibility rule owned by EngramEmbedding
    # (``n_hash_cols % tp_size == 0``). Do not relax this to that divisibility
    # rule alone: an evenly shardable Engram embedding does not mean the whole
    # model deploys at that TP (this model's head-group layout rules out
    # TP=3/6/12), and EngramEmbedding repeats the size check for the shard
    # contract itself.
    if cfg.tp_size not in (1, 2, 4, 8) or cfg.pp_size != 1 or cfg.dcp_size != 1:
        raise ValueError(
            "Engram requires TP in {1, 2, 4, 8} with PP=1 and DCP=1; got "
            f"TP={cfg.tp_size}, PP={cfg.pp_size}, DCP={cfg.dcp_size}. "
            "dp_shared_memory does not change TP sharding."
        )
    if cfg.load_format not in ("auto", "safetensors", "dummy"):
        raise ValueError(
            "Engram requires safetensors weights (load_format=auto/safetensors), "
            f"or dummy weights; got load_format={cfg.load_format}."
        )
    # Shared-without-offload is already rejected by EngramConfig.from_value;
    # re-checked here so no future construction path can bypass that invariant.
    if cfg.engram_config.dp_shared_memory and not cfg.engram_config.cpu_offload:
        raise ValueError("engram_config.dp_shared_memory=true requires engram_config.cpu_offload=true")
    # TODO: once V4.1 Engram is wired into a P/D-disaggregated topology it will
    # require dp_shared_memory=true there; not enforced at runtime yet.


# ---------------------------------------------------------------------------
# Checkpoint quantization modes (adaptation contract section 4)
# ---------------------------------------------------------------------------

V41_QUANT_FP8 = "fp8"
V41_QUANT_ASCEND = "ascend"
V41_QUANT_NONE = "none"


def _detect_v41_quant_mode(config: dict) -> str:
    """Read ``{model_path}/config.json`` and detect the checkpoint mode.

    ``quantization_config.quant_method == "fp8"`` selects the official
    FP8-block/MXFP4 release (dequantized to bf16 at load); a
    ``quant_model_description.json`` or ``quant_method == "ascend"`` selects
    the W8A8 Eco-Tech export; anything else is bf16.
    """
    model_path = config.get("model_path")
    if not model_path:
        logger.warning("DeepSeek-V4.1 quant mode detection: model_path missing; defaulting to bf16 (V41_QUANT_NONE)")
        return V41_QUANT_NONE
    if os.path.isfile(os.path.join(model_path, "quant_model_description.json")):
        return V41_QUANT_ASCEND
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return V41_QUANT_NONE
    with open(config_path, encoding="utf-8") as handle:
        checkpoint_config = json.load(handle)
    quant_config = checkpoint_config.get("quantization_config") or {}
    method = quant_config.get("quant_method")
    if method == "fp8":
        return V41_QUANT_FP8
    if method == "ascend":
        return V41_QUANT_ASCEND
    return V41_QUANT_NONE


def _make_v41_linear(
    in_features: int,
    out_features: int,
    quant_mode: str,
    dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """Construct one attention/dense linear per checkpoint mode."""
    if quant_mode == V41_QUANT_ASCEND:
        return W8A8DynamicLinear(
            in_features,
            out_features,
            device,
            transpose_weight_after_loading=False,
        )
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype, device=device)


# ---------------------------------------------------------------------------
# Pure-torch RMSNorm / RoPE tables / mHC
# ---------------------------------------------------------------------------


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize in FP32 with the existing native row reduction, then cast once."""
    return kernels.rms_norm(x.float(), weight.float(), eps).to(x.dtype)


# --- Fused RMSNorm+cast path (always on) -------------------------------------
#
# ``rms_norm_cast_va(x, gamma, eps) -> (y, y_fp32)`` (ACLNN ``aclnnRmsNormCastVa``)
# fuses the MoE FFN's ``post_attention_layernorm`` with the gate's
# ``Tensor.float()`` widening: ``y_fp32`` is the widened ROUNDED result
# (``y.float()``), NOT a separate fp32 normalisation. The norm sits in the MoE
# (``post_norm=``) because the forward all-gathers ``hidden`` across DP BEFORE
# the gate/expert input; RMSNorm is row-wise, so norm-then-gather holds.

_rms_norm_cast_op: Any = None


def _get_rms_norm_cast_op() -> Any:
    """Resolve the fused RMSNorm+cast op (``xllm_ops.rms_norm_cast_va``)."""
    global _rms_norm_cast_op
    if _rms_norm_cast_op is None:
        try:
            from xllm.python.kernels_npu.normalization import rms_norm_cast_va as _kernel
        except Exception:  # noqa: BLE001 - torch_npu absent on CPU-only hosts.
            return None
        if _kernel is None:
            return None
        _rms_norm_cast_op = _kernel
    return _rms_norm_cast_op


def _rms_norm_cast_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    """Guards for the fused RMSNorm+cast op.

    Every clause is a structural availability guard: when the op is
    registered and the shapes/dtypes match, the fused path engages; hosts
    whose CANN toolkit lacks ``aclnnRmsNormCastVa`` keep the bit-identical
    eager semantics.

    Evaluated by ``DeepseekV41W8A8MoE._apply_post_norm`` on the MoE input, which
    is already the pad+all-gathered DP stream (``DeepseekV4MoE.forward`` gathers
    BEFORE the gate/expert input). Normalising there is what makes the fusion
    valid under ``dp_size > 1``: RMSNorm is row-wise, so its fp32 routing input
    describes exactly the gathered rows the gate routes (and padded rows are
    zero, so they stay zero).

    All clauses are static (dtype, device type, registered op), so the
    decision is graph-capture safe. The NPU-platform check sits last so the
    dtype conditions stay observable on CPU-only hosts; every clause is
    pure, so the order does not change the result.
    """
    return (
        x.dim() in (2, 3)
        and x.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and weight.dim() == 1
        and weight.numel() == x.size(-1)
        and _get_rms_norm_cast_op() is not None
        and x.device.type in ("npu", "privateuseone")
    )


_rms_norm_cast_active_logged = False
_rms_norm_cast_fallback_logged = False


def _log_rms_norm_cast_active() -> None:
    """One-shot info when the fused RMSNorm+cast path first engages."""
    global _rms_norm_cast_active_logged
    if not _rms_norm_cast_active_logged:
        _rms_norm_cast_active_logged = True
        logger.info("rms_norm_cast fused path active (op=xllm_ops.rms_norm_cast_va)")


def _log_rms_norm_cast_fallback(x: torch.Tensor, weight: torch.Tensor) -> None:
    """One-shot warning when the fused path is requested but guards fail."""
    global _rms_norm_cast_fallback_logged
    if _rms_norm_cast_fallback_logged:
        return
    _rms_norm_cast_fallback_logged = True
    logger.warning(
        "the fused RMSNorm+cast op is unavailable "
        f"(device={x.device.type} dim={x.dim()} dtype={x.dtype} "
        f"weight_dtype={weight.dtype} op_registered={_get_rms_norm_cast_op() is not None}); "
        "staying on the eager path"
    )


_fused_shared_act_active_logged = False
_fused_shared_act_fallback_logged = False


def _fused_shared_act_device_supported(x: torch.Tensor) -> bool:
    """Whether ``x`` lives on an NPU (the fused W8A8 tail's only device contract).

    Split out so pure-Python tests can stub the platform predicate without an
    NPU, exactly like ``csa2_attention._rope_fused_device_supported``.
    """
    return x.device.type in ("npu", "privateuseone")


def _fused_shared_act_grouped_supported(shared: nn.Module) -> bool:
    """Whether the grouped single-group activation call is constructible.

    The clamp only survives on the operator's GROUPED branch (the dense branch
    ignores ``clamp_limit`` on device, see
    ``DeepseekV41SharedMLP.forward_dequant_swiglu_quant``), so the fused
    shared expert must build the same shape the golden clamped-activation case
    uses (``tests/core/kernels/npu/glm5_next_moe_cases.py:52-105``): a 1-D int64
    ``group_index`` holding per-group row COUNTS, and the gate-up scale carrying a
    leading group dimension, i.e. ``[1, 2H]`` for the single group this path
    uses. The scale is flattened into that layout, so its length has to equal the
    gate-up output width -- the same 2H the down projection consumes, as the
    alongside-``out_features`` scale of a loaded W8A8 projection does. Anything
    else -- an empty scale, or a length that is even but not 2H -- cannot express
    the pair layout and must fall back to eager rather than issue a call the
    operator's shape checks reject on device.
    """
    gate_up = getattr(shared, "gate_up_proj", None)
    down = getattr(shared, "down_proj", None)
    weight_scale = getattr(gate_up, "weight_scale", None)
    gate_up_width = getattr(gate_up, "out_features", None)
    down_in = getattr(down, "in_features", None)
    if weight_scale is None or gate_up_width is None or down_in is None:
        return False
    if weight_scale.dim() > 2:
        return False
    # The flattened scale must BE the gate-up output width -- the same 2H the down
    # projection consumes -- because ``[1, 2H]`` means one scale row per gate|up
    # pair. A merely even (or non-empty) length would pass a parity check and only
    # be rejected by the operator's shape check on device.
    return weight_scale.numel() == gate_up_width == 2 * down_in


def _fused_shared_act_supported(x: torch.Tensor, shared: nn.Module, limit: float) -> bool:
    """Guards for the fused shared-expert activation (always on).

    Replaces the eager clamped-SwiGLU chain with the W8A8 fused tail, keeping
    the SAME clamp contract as eager (gate upper-clamped only, up clamped on
    both sides). Three traps, all closed by this guard:

    * ``clamp_limit`` defaults to 0.0 (no clamp) on the operator, so the
      limit MUST be passed explicitly or the clamp is silently dropped.
    * SWIGLU_MODE MUST STAY 1: only ``aclnnDequantSwigluQuantV2`` honours the
      clamp; ``swiglu_mode=1`` makes a V1-only device ABORT loudly instead of
      silently running unclamped.
    * THE CALL MUST BE GROUPED, NOT DENSE: the dense branch (no ``group_index``)
      ignores ``clamp_limit`` on device, so ``_run_shared_experts`` passes
      ``grouped_activation=True`` and this guard requires it.

    Every clause is static, so the decision is graph-capture safe.
    """
    return (
        isinstance(shared, DeepseekV3MLP)
        and hasattr(shared, "forward_dequant_swiglu_quant")
        and 0.0 < limit < 1_000_000.0
        and x.dim() == 2
        and x.numel() > 0
        and x.dtype in (torch.bfloat16, torch.float16)
        and getattr(shared.gate_up_proj, "weight_scale", None) is not None
        and getattr(shared.down_proj, "weight_scale", None) is not None
        and _fused_shared_act_grouped_supported(shared)
        and _fused_shared_act_device_supported(x)
    )


def _log_fused_shared_act_active() -> None:
    """One-shot info when the fused shared-expert activation first engages."""
    global _fused_shared_act_active_logged
    if not _fused_shared_act_active_logged:
        _fused_shared_act_active_logged = True
        logger.info("shared expert fused clamped-SwiGLU active (op=xllm_ops.dequant_swiglu_quant)")


def _log_fused_shared_act_fallback(x: torch.Tensor, shared: nn.Module, limit: float) -> None:
    """One-shot warning when the fused path is asked for but the guards fail."""
    global _fused_shared_act_fallback_logged
    if _fused_shared_act_fallback_logged:
        return
    _fused_shared_act_fallback_logged = True
    logger.warning(
        "the fused shared-expert activation is unavailable "
        f"(device={x.device.type} dim={x.dim()} dtype={x.dtype} "
        f"max={x.numel()} limit={limit} module={type(shared).__name__} "
        f"grouped_form_ok={_fused_shared_act_grouped_supported(shared)}); "
        "staying on the eager clamped SwiGLU"
    )


#: Escape hatch that waives the DSpark-draft MC2 veto. DEFAULT OFF: the draft's
#: per-rank row counts are structurally uneven, so ``global_bs=0`` (which implies
#: equal rows per rank) mis-sizes the collective windows; an operator may opt in
#: explicitly for experiments. Like the enable decision it is process-wide and
#: therefore identical on every rank (collective-symmetric).

# A2 ``npu_moe_distribute_dispatch_v2`` tiling requires the per-rank batch in
# ``[1, 256]`` and its infershape rejects ``bs <= 0`` outright (see the guard).
_MOE_MC2_MAX_LOCAL_TOKENS = 256

_moe_mc2_active_logged = False
_moe_mc2_fallback_logged = False
#: Independent one-shot latch for the DSpark-draft exit (Fix-1). The target's
#: warmup reason would otherwise consume ``_moe_mc2_fallback_logged`` and keep
#: the draft exit out of the logs.
_moe_mc2_draft_fallback_logged = False
_moe_mc2_collective_warned = False


def _moe_mc2_allow_draft() -> bool:
    """Whether the draft MC2 veto is waived (it never is in production).

    Deliberate invariant, not a disabled feature: the DSpark draft's per-rank
    row counts are structurally uneven (a busy rank sees one sequence's whole
    block; an empty DP shard sees a single dummy row), while ``global_bs=0``
    means "every rank has the same row count". The two are incompatible, so the
    guard always declines the draft; the static form exists as the test escape
    hatch only.
    """
    return False


def _moe_mc2_ops_available() -> bool:
    """Whether this torch_npu registers both MC2 distribute kernels."""
    try:
        import torch_npu
    except Exception:  # noqa: BLE001 - torch_npu absent on CPU-only hosts.
        return False
    return hasattr(torch_npu, "npu_moe_distribute_dispatch_v2") and hasattr(torch_npu, "npu_moe_distribute_combine_v2")


def _moe_mc2_device_supported(x: torch.Tensor) -> bool:
    """Split out so pure-Python tests can stub the platform predicate without an NPU."""
    return x.device.type in ("npu", "privateuseone")


def _moe_mc2_decode_step() -> bool:
    """Whether the current forward is a non-dummy decode step.

    ``dp == 1`` predicate only. The ``is_dummy`` / ``is_prefill`` reads are
    rank-local, so under ``dp > 1`` the guard must use the host-planned
    ``_moe_mc2_all_dp_ranks_decode`` instead: an empty DP rank is dummy locally
    yet C++ forces it to decode so it stays in lockstep with its peers.
    """
    ctx = get_forward_context_or_none()
    if ctx is None:
        return False
    metadata = ctx.metadata
    if getattr(metadata, "is_dummy", False):
        return False
    return not (metadata.is_prefill or metadata.is_chunked_prefill)


def _moe_mc2_all_dp_ranks_decode(metadata: Any) -> bool:
    """Whether every DP rank is on a decode step (C++ ``all_dp_ranks_are_decode``).

    Mirrors ``dp_utils.cpp:all_dp_ranks_are_decode``: the per-DP decode flags are
    host-planned and identical on every rank, so this predicate is
    collective-symmetric -- unlike a rank-local ``is_prefill`` or
    ``hidden.numel()`` read. Falls back to the single-replica test when the
    scheduler did not publish the flags (``dp_global_token_nums.size() <= 1``).
    """
    dp_is_decode = getattr(metadata, "dp_is_decode", None)
    if dp_is_decode:
        return all(int(flag) == 1 for flag in dp_is_decode)
    token_counts = getattr(metadata, "dp_execution_token_counts", None)
    return not token_counts or len(token_counts) <= 1


def _moe_mc2_dispatch(**kwargs: Any) -> tuple:
    """The dispatch kernel, behind a seam so tests can substitute a fake."""
    import torch_npu

    return torch_npu.npu_moe_distribute_dispatch_v2(**kwargs)


def _moe_mc2_combine(**kwargs: Any) -> torch.Tensor:
    """The combine kernel, behind a seam so tests can substitute a fake."""
    import torch_npu

    return torch_npu.npu_moe_distribute_combine_v2(**kwargs)


def _moe_mc2_grouped_matmul(**kwargs: Any) -> torch.Tensor:
    """The grouped GEMM, behind a seam so tests can substitute a fake."""
    return torch.ops.npu.npu_grouped_matmul(**kwargs)[0]


def _log_moe_mc2_active(ep_rank: int, ep_size: int) -> None:
    """One-shot info when the MC2 decode expert path first engages."""
    global _moe_mc2_active_logged
    if not _moe_mc2_active_logged:
        _moe_mc2_active_logged = True
        logger.info(
            f"MoE MC2 decode path active (ep_rank={ep_rank} ep_size={ep_size}, "
            "chain=dispatch_v2->dynamic_quant->grouped_matmul->dequant_swiglu_quant->grouped_matmul->combine_v2)"
        )


def _log_moe_mc2_fallback(reason: str, enabled: bool) -> None:
    """One-shot warning when MC2 is requested but a guard declines."""
    global _moe_mc2_fallback_logged
    if _moe_mc2_fallback_logged:
        return
    if not enabled:
        return
    _moe_mc2_fallback_logged = True
    logger.warning(
        f"the MC2 decode path is unavailable for the target ({reason}); using the portable grouped-expert path"
    )


def _moe_mc2_draft_veto_applies(moe: Any) -> bool:
    """Whether the DSpark-draft MC2 veto (Fix-1) is what declines ``moe``.

    Single source of truth shared by ``_mc2_guard_reason`` and the fallback-log
    routing so the decision and its log can never drift: when this is ``True``
    the guard returns the draft clause ahead of every other probe, so ``reason``
    is always the draft one. Static module state, identical on every rank.
    """
    return not _moe_mc2_allow_draft() and bool(getattr(moe, "_moe_is_dspark_draft", False))


def _log_moe_mc2_draft_fallback(reason: str, enabled: bool) -> None:
    """One-shot warning that the DSpark draft leaves the MC2 decode path (Fix-1).

    Uses its OWN latch, deliberately separate from ``_moe_mc2_fallback_logged``:
    target warmup (``not all DP ranks are decode``) fires first and would consume
    the shared process latch, hiding the draft's exit. A dedicated latch makes
    the draft's portable fallback observable on machine without changing the
    target's one-shot semantics or any control flow.
    """
    global _moe_mc2_draft_fallback_logged
    if _moe_mc2_draft_fallback_logged:
        return
    if not enabled:
        return
    _moe_mc2_draft_fallback_logged = True
    logger.warning(
        "the MC2 decode path is unavailable for the DSpark "
        f"draft (Fix-1, target keeps MC2): {reason}; using the portable "
        "grouped-expert path"
    )


def _log_moe_mc2_collective(ep_rank: int, ep_size: int) -> None:
    """One-shot warning that the MC2 path is a collective that must stay symmetric."""
    global _moe_mc2_collective_warned
    if not _moe_mc2_collective_warned:
        _moe_mc2_collective_warned = True
        logger.warning(
            "the MC2 decode path engages an all-to-all *collective* expert path: every rank of the "
            f"moe_ep group must run it unconditionally and in lockstep (ep_rank={ep_rank} "
            f"ep_size={ep_size}); a rank that skips a layer will hang the dispatch/combine kernels"
        )


class DeepseekV41RMSNorm(nn.Module):
    """Pure-torch RMSNorm (the shared layer RMSNorm calls NPU kernels)."""

    def __init__(
        self,
        dim: int,
        eps: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None) -> torch.Tensor:
        if residual is not None:
            x = x + residual
        return _rms_norm(x, self.weight, self.eps)


def _build_plain_cos_sin_cache(
    rotary_dim: int,
    max_positions: int,
    theta: float,
    device: torch.device,
) -> torch.Tensor:
    """Plain RoPE table (theta 10000, YaRN off) for the ratio-0 layers.

    Mirrors ``precompute_freqs_cis`` with ``original_seq_len == 0``
    (model.py:368-389): one cos/sin pair per frequency per position, stored
    as ``[max_positions, rotary_dim]`` (cos | sin halves).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim))
    positions = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return cache.to(device).contiguous()


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split the hc mixes into (pre, post, comb) coefficients.

    Faithful port of the reference ``hc_split_sinkhorn`` (kernel.py:406-474):
    ``pre = sigmoid(mix * scale[0] + base) + eps``,
    ``post = 2 * sigmoid(mix * scale[1] + base)``, and ``comb`` from a row
    softmax (+eps) followed by one column normalization and
    ``sinkhorn_iters - 1`` row+column normalization rounds.
    """
    pre = torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult])
    comb = mixes[:, 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = comb * hc_scale[2] + hc_base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=1, keepdim=True) + eps)
    return pre, post, comb


# --- Fused hc_post path (always on) -----------------------------------------
#
# ``xllm_ops.hc_post_va`` (ACLNN ``aclnnHcPostVa``) replaces the eager hc_post
# recombination when registered; a build without the VA registration keeps the
# eager path. Semantics:
# ``y[t, o, :] = sum_i comb[t, i, o] * residual[t, i, :] + post[t, o] * x[t, :]``
# (``out = x * post + comb^T @ residual``; the vllm/tilelang reference).
# The kernel consumes the natural [out, in] comb as comb^T, so no transpose is
# needed before the call.


_hc_post_op: Any = None


def _get_hc_post_op() -> Any:
    """Resolve the DSV4.1-specific VA hc_post op (``xllm_ops.hc_post_va``).

    DSV4.1 is bound to the VA HyperConnection kernel (``aclnnHcPostVa``),
    registered under its own ``hc_post_va`` op name so the shared ``hc_post``
    op keeps its original (ops-transformer) semantics for the other models.
    Fail-closed: a build without the VA registration resolves to None and the
    forward uses the eager recombination.
    """
    global _hc_post_op
    if _hc_post_op is None:
        try:
            va_op = getattr(torch.ops.xllm_ops, "hc_post_va", None)
        except (AttributeError, RuntimeError):
            va_op = None
        if va_op is None:
            return None
        _hc_post_op = va_op
    return _hc_post_op


def _hc_post_fused_supported(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> bool:
    """Guards for the fused hc_post op."""
    hidden = x.size(-1)
    return (
        x.device.type in ("npu", "privateuseone")
        and x.dtype == torch.bfloat16
        and x.dim() == 2
        and residual.dim() == 3
        and post.dim() == 2
        and comb.dim() == 3
        and hc_mult == 4
        and residual.size(1) == hc_mult
        and residual.size(2) == hidden
        and post.size(1) == hc_mult
        and comb.size(1) == hc_mult
        and comb.size(2) == hc_mult
        and post.dtype == torch.float32
        and comb.dtype == torch.float32
        and residual.dtype == x.dtype
        and x.is_contiguous()
        and residual.is_contiguous()
        and post.is_contiguous()
        and hidden % 64 == 0
        and _get_hc_post_op() is not None
    )


_fused_hc_post_active_logged = False
_fused_hc_post_fallback_logged = False


def _log_fused_hc_post_active() -> None:
    """One-shot info when the fused hc_post path first engages."""
    global _fused_hc_post_active_logged
    if not _fused_hc_post_active_logged:
        _fused_hc_post_active_logged = True
        logger.info("hc_post fused path active")


def _log_fused_hc_post_fallback(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
) -> None:
    """One-shot warning when the fused path is requested but guards fail."""
    global _fused_hc_post_fallback_logged
    if _fused_hc_post_fallback_logged:
        return
    _fused_hc_post_fallback_logged = True
    logger.warning(
        "the fused hc_post op is unavailable "
        f"(device={x.device.type} dtype={x.dtype} x_dim={x.dim()} residual_dim={residual.dim()} "
        f"post_dtype={post.dtype} comb_dtype={comb.dtype} hc_mult={hc_mult} "
        f"op_registered={_get_hc_post_op() is not None}); staying on the eager path"
    )


# --- Fused hc_pre path (always on) ------------------------------------------
#
# ``xllm_ops.hc_pre_va`` (aclnnHcPreVa, sourced from the vllm-ascend open
# AscendC tree) replaces the eager ``_hc_split_sinkhorn`` + ``hc_pre`` collapse
# pair: one call collapses with the EXTERNAL ``pre_mix`` and returns this
# block's own ``pre`` plus the recombination coefficients. The two-state
# decision is taken once in ``DeepseekV41HyperConnection.__init__`` (see
# ``_hc_pre_mode_decision``); the forward only compares the cached mode string.

_HC_PRE_MODE_EAGER = "eager"
_HC_PRE_MODE_FUSED = "fused"

_hc_pre_fused_op: Any = None


def _get_hc_pre_fused_op() -> Any:
    """Resolve the ``xllm_ops::hc_pre_va`` wrapper (external ``pre_mix`` collapse)."""
    global _hc_pre_fused_op
    if _hc_pre_fused_op is None:
        try:
            from xllm.python.kernels_npu.hyper_connection import hc_pre_va as _kernel
        except Exception:  # noqa: BLE001 - torch_npu absent on CPU-only hosts.
            return None
        if _kernel is None:
            return None
        _hc_pre_fused_op = _kernel
    return _hc_pre_fused_op


def _hc_pre_fused_guards(dtype: torch.dtype, device: torch.device, hc_mult: int) -> bool:
    """Static preconditions for the fused op, excluding its registration.

    The kernel only supports the 4-stream mHC geometry (like hc_post) and
    bf16 activations on an NPU device.
    """
    device_type = torch.device(device).type
    return hc_mult == 4 and dtype == torch.bfloat16 and device_type in ("npu", "privateuseone")


def _hc_pre_mode_decision(dtype: torch.dtype, device: torch.device, hc_mult: int) -> str:
    """Two-state static hc_pre decision, taken once when the module is built.

    fused wins when its guards hold and the op is registered; eager is the
    fallback. Every input (hc_mult, dtype/device, op registration) is fixed
    before the first forward, so the decision needs no per-step tensor branch
    and stays ACL-graph safe.
    """
    if _hc_pre_fused_guards(dtype, device, hc_mult) and _get_hc_pre_fused_op() is not None:
        return _HC_PRE_MODE_FUSED
    return _HC_PRE_MODE_EAGER


_fused_hc_pre_active_logged = False
_fused_hc_pre_fallback_logged = False


def _log_fused_hc_pre_active() -> None:
    """One-shot info when the fused hc_pre path first engages."""
    global _fused_hc_pre_active_logged
    if not _fused_hc_pre_active_logged:
        _fused_hc_pre_active_logged = True
        logger.info("hc_pre fused path active (native pre vector and external pre_mix)")


def _log_fused_hc_pre_fallback(dtype: torch.dtype, device: torch.device, hc_mult: int) -> None:
    """One-shot warning when fused is requested on a supported geometry but missing."""
    global _fused_hc_pre_fallback_logged
    if _fused_hc_pre_fallback_logged:
        return
    if not _hc_pre_fused_guards(dtype, device, hc_mult):
        # Intentional fallback (unsupported geometry or a non-NPU host): the
        # eager path is the documented behaviour, so do not warn.
        return
    _fused_hc_pre_fallback_logged = True
    logger.warning(
        "the fused hc_pre op is unavailable "
        f"(device={torch.device(device).type} dtype={dtype} hc_mult={hc_mult} "
        f"op_registered={_get_hc_pre_fused_op() is not None}); staying on the eager sinkhorn path"
    )


def _log_hc_pre_mode(dtype: torch.dtype, device: torch.device, hc_mult: int, mode: str) -> None:
    """Tiered one-shot log for the two-state decision.

    A fused selection logs once at info; an eager fallback caused by a missing
    op at a supported geometry warns once. An intentional eager fallback
    (``hc_mult != 4`` or a non-NPU host) stays silent, matching the existing
    conventions.
    """
    if mode == _HC_PRE_MODE_FUSED:
        _log_fused_hc_pre_active()
        return
    _log_fused_hc_pre_fallback(dtype, device, hc_mult)


# --- Merged attention-input quant (always on) -------------------------------------
#
# q_a_proj and kv_proj are W8A8DynamicLinears over the SAME hidden rows;
# quantizing hidden once and feeding both projections forward_quantized is
# bitwise-identical (same op, same input).


def _merged_attn_quant_supported(x: torch.Tensor) -> bool:
    """Guards for the merged attention-input quant."""
    return x.device.type in ("npu", "privateuseone") and x.dim() == 2


class DeepseekV41HyperConnection(nn.Module):
    """Single-pass Mega-mHC hyper-connections (semantics doc section 7).

    Pure-torch counterpart of the reference ``Block`` mHC pieces
    (model.py:948-966): ``hc_mixes`` projects the flattened hc*dim stream
    (normalized AFTER projection), ``hc_pre`` collapses the hc copies with a
    per-token coefficient vector, and ``hc_post`` expands a sublayer output
    back mixing the residual through the doubly-stochastic ``comb``.
    """

    def __init__(self, cfg: DeepseekV41Config, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.hc_mult = cfg.hc_mult
        self.hc_eps = cfg.hc_eps
        self.norm_eps = cfg.rms_norm_eps
        self.sinkhorn_iters = cfg.hc_sinkhorn_iters
        # Two-state hc_pre decision is static (hc_mult + dtype/device +
        # op registration) and taken once here so the forward stays
        # branch-free with respect to tensor values.
        self._hc_pre_mode = _hc_pre_mode_decision(dtype, device, cfg.hc_mult)
        _log_hc_pre_mode(dtype, device, cfg.hc_mult, self._hc_pre_mode)
        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult
        hc_dim = cfg.hc_mult * cfg.hidden_size
        # hc_fn/scale/base per sub-block, fp32 as in the checkpoint.
        for part in ("attn", "ffn"):
            self.register_parameter(
                f"hc_{part}_fn",
                nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_base",
                nn.Parameter(torch.empty(mix_hc, dtype=torch.float32, device=device)),
            )
            self.register_parameter(
                f"hc_{part}_scale",
                nn.Parameter(torch.empty(3, dtype=torch.float32, device=device)),
            )

    def _project_mixes(self, flattened: torch.Tensor, hc_fn: torch.Tensor) -> torch.Tensor:
        """Keep the small HC projection's FP32 GEMM geometry fixed across chunks."""
        output = flattened.new_empty((flattened.size(0), hc_fn.size(0)))
        if flattened.size(0) == 0:
            return output
        # Reuse one call-local tile; every GEMM retains the same 128-row shape.
        # Padding rows are independent and their outputs are discarded.
        tile = flattened.new_empty((128, flattened.size(1)))
        for start in range(0, flattened.size(0), 128):
            count = min(128, flattened.size(0) - start)
            tile[:count].copy_(flattened[start : start + count])
            output[start : start + count].copy_(F.linear(tile, hc_fn)[:count])
        return output

    def hc_mixes(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x ``[T, hc, d]`` -> (pre, post, comb) per token."""
        flattened = x.flatten(-2).float()
        rsqrt = torch.rsqrt(flattened.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = self._project_mixes(flattened, hc_fn) * rsqrt
        return _hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.sinkhorn_iters, self.hc_eps)

    def hc_pre_fused(
        self,
        x: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        pre_mix: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-op mHC pre: external-``pre_mix`` collapse plus (pre, post, comb).

        Returns ``(y, pre, post, comb)``: the collapsed sub-block input ``y``,
        this block's own ``pre`` (which the NEXT sub-block feeds as its external
        mix), and the two recombination coefficients. ``pre_mix=None``
        materialises the one-hot first-block mix here so callers never carry an
        optional-tensor branch into the forward.
        """
        if pre_mix is None:
            pre_mix = self.make_identity_pre_mix(x, self.hc_mult)
        # All nine operands go positionally: the op's PrivateUse1 dispatch
        # drops keyword arguments, so a keyword tail would silently lose the
        # geometry/eps knobs (see ``kernels_npu.hyper_connection``).
        y, post, comb, pre = _get_hc_pre_fused_op()(
            x,
            hc_fn,
            hc_scale,
            hc_base,
            pre_mix,
            self.hc_mult,
            self.sinkhorn_iters,
            self.norm_eps,
            self.hc_eps,
        )
        return y, pre, post, comb

    def hc_pre(self, x: torch.Tensor, pre_mix: torch.Tensor) -> torch.Tensor:
        """Collapse the hc copies: ``sum_j pre[j] * x[j]`` (model.py:957-960)."""
        # A fixed FP32 order avoids token-count-dependent BF16 rounding.
        contributions = (pre_mix.unsqueeze(-1) * x.float()).unbind(dim=-2)
        y = contributions[0]
        for contribution in contributions[1:]:
            y = y + contribution
        return y.to(x.dtype)

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        post: torch.Tensor,
        comb: torch.Tensor,
    ) -> torch.Tensor:
        """Expand back: ``out[j] = post[j] * x + sum_k comb[k, j] * res[k]``
        (i.e. ``out = x * post + comb^T @ residual``)."""
        if _hc_post_fused_supported(x, residual, post, comb, self.hc_mult):
            # The kernel expects [bs, seq, ...] 4D operands. It consumes the
            # natural [out, in] comb as comb^T @ residual, so no transpose is
            # needed before the call.
            comb4d = comb.contiguous().unsqueeze(0)
            out = _get_hc_post_op()(
                x.unsqueeze(0),
                residual.unsqueeze(0),
                post.unsqueeze(0),
                comb4d,
            )
            _log_fused_hc_post_active()
            return out.squeeze(0)
        _log_fused_hc_post_fallback(x, residual, post, comb, self.hc_mult)
        y = post.unsqueeze(-1) * x.float().unsqueeze(-2)
        y = y + torch.einsum("tkj,tkd->tjd", comb.float(), residual.float())
        return y.to(x.dtype)

    @staticmethod
    def make_identity_pre_mix(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
        """Initial one-hot mix on copy 0 (model.py:1159-1163)."""
        pre_mix = torch.zeros(x.size(0), hc_mult, dtype=torch.float32, device=x.device)
        pre_mix[:, 0] = 1.0
        return pre_mix


# ---------------------------------------------------------------------------
# Indexer weights
# ---------------------------------------------------------------------------


class DeepseekV41Indexer(nn.Module):
    """DeepSeek-V4.1 CSA2 indexer weights (semantics doc section 5).

    ``wq_b`` projects the shared q-lora latent; ``weights_proj`` produces the
    per-head score weights; kv-source layers additionally own the key path
    ``wk`` + ``k_norm`` -- ``wk`` acts on the 512-d compressor latent, NOT on
    hidden states (checkpoint ``layers.N.attn.indexer.*``). There is no
    Hadamard transform in V4.1: the index plane is INT8-only -- Q/K are
    per-row symmetric INT8-quantized (FP16 scales) and the scoring runs
    through einsum in the backend.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str,
        owns_k: bool,
    ) -> None:
        super().__init__()
        self.owns_k = owns_k
        self.n_heads = cfg.index_n_heads
        self.head_dim = cfg.index_head_dim
        self.wq_b = _make_v41_linear(
            cfg.q_lora_rank,
            self.n_heads * self.head_dim,
            quant_mode,
            dtype,
            device,
        )
        self.weights_proj = nn.Linear(cfg.hidden_size, self.n_heads, bias=False, dtype=dtype, device=device)
        if owns_k:
            # wk: head_dim (512) -> index_head_dim (128); one shared key per
            # compressed position (model.py:518).
            self.wk = nn.Linear(cfg.head_dim, self.head_dim, bias=False, dtype=dtype, device=device)
            self.k_norm = DeepseekV41RMSNorm(self.head_dim, cfg.rms_norm_eps, dtype, device)

    def process_weights_after_loading(self) -> None:
        if isinstance(self.wq_b, W8A8DynamicLinear):
            self.wq_b.process_weights_after_loading()


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class DeepseekV41Attention(Attention):
    """DeepSeek-V4.1 CSA2 attention (semantics doc sections 4-6).

    Owns the reference weight set: latent q/kv projections with
    interleaved-pair RoPE on the last ``qk_rope_head_dim`` dims, the
    attention sink, the compressor weights (Full layers) and the indexer
    weights (Full/Reindex layers), plus the grouped ``wo_a``/``wo_b`` output
    projection. Cache orchestration and the sink softmax live in
    :class:`Csa2AttentionBackend`.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
    ) -> None:
        tp = cfg.tp_size
        if cfg.enable_dsa_cp and cfg.cp_size > 1:
            # V4.1's separate PCP path is also unsupported by Model.forward.
            raise ValueError("enable_dsa_cp cannot be combined with cp_size > 1")
        self.enable_dsa_cp = cfg.enable_dsa_cp and tp > 1
        if self.enable_dsa_cp and cfg.n_heads % tp:
            raise ValueError("enable_dsa_cp requires n_heads divisible by tp_size")
        num_heads = cfg.n_heads if self.enable_dsa_cp else cfg.n_heads // tp
        head_dim = cfg.head_dim
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=1,
            head_dim=head_dim,
            scale=head_dim**-0.5,
            sliding_window=cfg.window_size,
            layer_id=layer_id,
        )
        self.cfg = cfg
        self.layer_id = layer_id
        self.mode = mode
        self.quant_mode = quant_mode
        self.num_heads_local = num_heads
        self.head_dim = head_dim
        # Attention sink (learnable logit bias per head); contributes to the
        # softmax normalizer only.
        self.register_parameter(
            "attn_sink",
            nn.Parameter(torch.zeros(num_heads, dtype=torch.float32, device=device)),
        )
        self.attn_sink_loaded = False
        # Q/KV projections.
        self.q_a_proj = _make_v41_linear(cfg.hidden_size, cfg.q_lora_rank, quant_mode, dtype, device)
        self.q_a_layernorm = DeepseekV41RMSNorm(cfg.q_lora_rank, cfg.rms_norm_eps, dtype, device)
        self.q_b_proj = _make_v41_linear(cfg.q_lora_rank, num_heads * head_dim, quant_mode, dtype, device)
        self.kv_proj = _make_v41_linear(cfg.hidden_size, head_dim, quant_mode, dtype, device)
        self.kv_a_layernorm = DeepseekV41RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        # Grouped output projection: o_a (block-diagonal over groups, bf16
        # after the official conversion) -> o_b.
        assert cfg.o_groups % tp == 0
        self.n_local_groups = cfg.o_groups // tp
        self.o_lora_rank = cfg.o_lora_rank
        o_a_in = (cfg.n_heads * head_dim) // cfg.o_groups
        if quant_mode == V41_QUANT_ASCEND:
            self.o_a_proj = ColumnParallelLinear(
                o_a_in,
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                tp,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = RowParallelLinear(
                (cfg.o_groups * cfg.o_lora_rank) // tp,
                cfg.hidden_size,
                tp,
                dtype=dtype,
                device=device,
                use_checkpoint_layout=True,
            )
        else:
            # TP-sharded like the ascend branch: each rank owns the head
            # groups matching its q head slice; the o_b output is partial and
            # all-reduced across the TP group in forward.
            self.o_a_proj = nn.Linear(
                o_a_in,
                self.n_local_groups * cfg.o_lora_rank,
                bias=False,
                dtype=dtype,
                device=device,
            )
            self.o_b_proj = nn.Linear(
                self.n_local_groups * cfg.o_lora_rank,
                cfg.hidden_size,
                bias=False,
                dtype=dtype,
                device=device,
            )
        # Main-KV compressor (Full layers only). Ratio 2 runs the gated
        # pooling in fp32, so its wkv/wgate are fp32; ratio 1 is a plain bf16
        # projection with no gate and no state (model.py:437-456).
        if mode.is_full:
            ratio = mode.compress_ratio
            cmp_dtype = torch.float32 if ratio > 1 else dtype
            self.cmp_wkv = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=cmp_dtype, device=device)
            if ratio > 1:
                self.cmp_wgate = nn.Linear(cfg.hidden_size, head_dim, bias=False, dtype=torch.float32, device=device)
            self.cmp_norm = DeepseekV41RMSNorm(head_dim, cfg.rms_norm_eps, dtype, device)
        # Indexer: Full layers write the shared K and select; Reindex layers
        # re-select against it; Reuse/SWA layers own no indexer weights.
        self.indexer: DeepseekV41Indexer | None = (
            DeepseekV41Indexer(cfg, dtype, device, quant_mode, owns_k=mode.is_full)
            if mode.is_full or mode.is_reindex
            else None
        )
        # Per-forward hand-off to the backend.
        self._csa2_ctx: Csa2LayerContext | None = None
        self._csa2_used_topk: torch.Tensor | None = None

    def process_weights_after_loading(self) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            for module in (self.q_a_proj, self.q_b_proj, self.kv_proj):
                module.process_weights_after_loading()
            if hasattr(self.o_a_proj, "process_weights_after_loading"):
                self.o_a_proj.process_weights_after_loading()
            if self.indexer is not None:
                self.indexer.process_weights_after_loading()

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
    ) -> torch.Tensor:
        # The CSA2 layer context / RoPE helpers live in the attention backend
        # module (csa2_attention); the import is deferred so the model file
        # itself has no backend dependency at import time.
        from xllm.python.attention.csa2_attention import (
            Csa2LayerContext,
            apply_interleaved_rope,
        )

        backend = get_forward_context().attention_backend
        execute_csa2 = getattr(backend, "execute_csa2_layer", None)
        if not callable(execute_csa2):
            raise RuntimeError("DeepSeek-V4.1 requires the CSA2 attention backend")
        num_tokens = hidden.size(0)
        cp_context = None
        query_positions = positions
        if self.enable_dsa_cp:
            from xllm.python.model_executor.dsa_cp import get_dsa_cp_runtime

            runtime = get_dsa_cp_runtime(num_tokens, self.cfg.tp_size, self.cfg.tp_rank)
            cp_context = runtime.plan
            query_positions = cp_context.shard(positions)
        # Q path: qr = q_norm(wq_a(x)); q = wq_b(qr) (no norm after wq_b);
        # RoPE on the last rope dims at token positions (model.py:770-772).
        # MERGED_ATTN_QUANT (default on): q_a_proj and kv_proj both consume the
        # SAME hidden rows, so the W8A8 dynamic quant of hidden runs once and
        # feeds both projections (bitwise-identical to the per-projection
        # quant; saves 40 dynamic_quant calls per prefill step, each over
        # [tokens, hidden]).
        if self.quant_mode == V41_QUANT_ASCEND and _merged_attn_quant_supported(hidden):
            from xllm.python import kernels as _kernels

            hidden_int8, hidden_pertoken = _kernels.dynamic_quant(hidden)
            q_a = self.q_a_proj.forward_quantized(hidden_int8, hidden_pertoken)
            kv = self.kv_a_layernorm(self.kv_proj.forward_quantized(hidden_int8, hidden_pertoken))
        else:
            q_a = self.q_a_proj(hidden)
            kv = self.kv_a_layernorm(self.kv_proj(hidden))
        # q norm only: the per-token INT8 quant path is disabled, so
        # ``_q_norm_quant`` returns the eagerly normed bf16 latent with a None
        # scale and both consumers (q_b_proj here, the indexer via ctx.qr_scale)
        # run on the unquantized latent.
        qr, qr_scale = self._q_norm_quant(q_a)
        query_qr = cp_context.shard(qr) if cp_context is not None else qr
        if query_qr.size(0) == 0:
            q = hidden.new_empty((0, self.num_heads_local, self.head_dim))
        else:
            q = self._q_b_proj_forward(query_qr, qr_scale).view(-1, self.num_heads_local, self.head_dim)
            q = apply_interleaved_rope(q, query_positions, cos_sin_cache)
        # SWA KV path: kv_norm -> RoPE over the whole post-RoPE vector
        # (model.py:700-720). The KV stays in bf16, aligned with vllm-ascend.
        kv = apply_interleaved_rope(kv, positions, cos_sin_cache)
        self._csa2_ctx = Csa2LayerContext(
            x=hidden,
            qr=qr,
            positions=positions,
            cos_sin_cache=cos_sin_cache,
            qr_scale=qr_scale,
        )
        try:
            attn_out = execute_csa2(q, kv, self)
        finally:
            self._csa2_ctx = None
        # De-rotate the rope dims by the query's own rotation (conjugate)
        # BEFORE the o path (model.py:781).
        if attn_out.size(0) > 0:
            attn_out = apply_interleaved_rope(attn_out, query_positions, cos_sin_cache, inverse=True)
        if cp_context is not None:
            attn_out = cp_context.restore_tp_heads(attn_out)
        # O path: grouped LoRA projection (model.py:783-789).
        out = attn_out.view(num_tokens, self.n_local_groups, -1)
        wo_a = self.o_a_proj.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o_low = torch.einsum("tgd,grd->tgr", out, wo_a)
        o = self.o_b_proj(o_low.reshape(num_tokens, -1))
        if self.cfg.tp_size > 1 and self.quant_mode != V41_QUANT_ASCEND:
            # Row-parallel partial: sum the per-rank head-group contributions.
            from xllm.python import distributed as _distributed

            _distributed.tp_all_reduce(o)
        return o

    # -- weight helpers used by the backend -------------------------------------

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compressor projections: fp32 (kv, score) for ratio > 1, (kv, None) for ratio 1."""
        if self.mode.compress_ratio > 1:
            xf = x.float()
            return self.cmp_wkv(xf), self.cmp_wgate(xf)
        return self.cmp_wkv(x), None

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        """RMSNorm AFTER pooling (ratio > 1) / after the projection (ratio 1)."""
        return _rms_norm(v, self.cmp_norm.weight, self.cfg.rms_norm_eps)

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        """Indexer K from the pre-RoPE compressor latent: k_norm(wk(latent))."""
        k = self.indexer.wk(latent)
        return _rms_norm(k, self.indexer.k_norm.weight, self.cfg.rms_norm_eps)

    def _q_norm_quant(self, q_a: torch.Tensor) -> tuple[torch.Tensor, None]:
        """q_a -> (normed bf16 latent, None); the fused INT8 path is disabled."""
        return self.q_a_layernorm(q_a), None

    def _q_b_proj_forward(self, qr: torch.Tensor, qr_scale: torch.Tensor | None) -> torch.Tensor:
        """q_b_proj over the shared latent, reusing fused codes when present."""
        if qr_scale is not None:
            return self.q_b_proj.forward_quantized(qr, qr_scale)
        return self.q_b_proj(qr)

    def indexer_query(self, qr: torch.Tensor, qr_scale: torch.Tensor | None = None) -> torch.Tensor:
        """Indexer Q from the shared q-lora latent: wq_b(qr).

        Under the fused q norm+quant path the latent arrives already
        per-token INT8-quantized (``qr_scale`` present); the indexer then
        reuses those codes/scales through ``forward_quantized`` instead of
        re-quantizing -- the DSV4 production shape (deepseek_v4.py:826-830).
        """
        if qr_scale is not None:
            q = self.indexer.wq_b.forward_quantized(qr, qr_scale)
        else:
            q = self.indexer.wq_b(qr)
        return q.view(-1, self.indexer.n_heads, self.indexer.head_dim)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Per-head scoring weights: weights_proj(x) * (ihd^-0.5 * nh^-0.5)."""
        scale = self.indexer.head_dim**-0.5 * self.indexer.n_heads**-0.5
        return self.indexer.weights_proj(x) * scale


# ---------------------------------------------------------------------------
# MoE (torch) + dense MLP + W8A8 MoE (ascend)
# ---------------------------------------------------------------------------


class DeepseekV41TorchMLP(nn.Module):
    """Dense SwiGLU FFN with the reference clamps (fp8/none checkpoint modes)."""

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.swiglu_limit = cfg.swiglu_limit
        inter = cfg.moe_intermediate_size
        self.w1 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w3 = nn.Linear(cfg.hidden_size, inter, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(inter, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.w1(x), self.w3(x), self.w2, self.swiglu_limit, None)


def _swiglu_clamped(
    x: torch.Tensor,
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    w2: nn.Module,
    limit: float,
    route_weight: torch.Tensor | None,
) -> torch.Tensor:
    """Reference Expert.forward (model.py:841-851).

    The up branch clamps both sides, the gate branch only from above; the
    routing weight (when given) multiplies the hidden before w2.
    """
    gate = gate_proj.float()
    up = up_proj.float()
    if 0.0 < limit < 1_000_000.0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    hidden = F.silu(gate) * up
    if route_weight is not None:
        hidden = route_weight.unsqueeze(-1) * hidden
    return w2(hidden.to(x.dtype))


class DeepseekV41MoE(nn.Module):
    """DeepSeek-V4.1 MoE for the fp8/none checkpoint modes (pure torch).

    Routing (model.py:809-827): fp32 ``softplus(x @ W).sqrt()`` scores, top-k
    of ``scores + bias`` where image-span tokens use ``gate.bias_vl``
    (selection only), weights gathered from the UNBIASED scores, normalized
    (``sum + 1e-20``) and scaled by ``routed_scaling_factor``. Experts are
    the clamped SwiGLU with fp32 routed accumulation plus one shared expert.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_id = layer_id
        self.topk = cfg.n_activated_experts
        self.n_routed_experts = cfg.n_routed_experts
        self.routed_scaling = cfg.routed_scaling_factor
        self.norm_topk_prob = cfg.norm_topk_prob
        self.swiglu_limit = cfg.swiglu_limit
        self.image_token_id = cfg.image_token_id
        self.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False, dtype=dtype, device=device)
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl = nn.Parameter(
            torch.zeros(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False
        inter = cfg.moe_intermediate_size
        hidden = cfg.hidden_size
        # EP: every rank holds only its slice of the routed experts
        # (mirrors C++ FusedMoEImpl and DeepseekV4MoE). The forward
        # accumulates only the local expert slice and all-reduces the routed
        # partial across the EP group, so the summed output is exact.
        ep_size = cfg.ep_size if cfg.ep_size > 0 else 1
        ep_rank = cfg.ep_rank if cfg.ep_size > 0 else 0
        if cfg.n_routed_experts % ep_size != 0:
            raise ValueError(
                f"deepseek v4.1 torch MoE requires n_routed_experts divisible by ep_size, "
                f"got {cfg.n_routed_experts} experts and ep_size={ep_size}"
            )
        self.ep_size = ep_size
        self.num_experts_per_rank = cfg.n_routed_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank
        # Stacked dequantized experts (fp8 mode dequantizes at load).
        self.experts_w1 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w2 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, hidden, inter, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.experts_w3 = nn.Parameter(
            torch.empty(self.num_experts_per_rank, inter, hidden, dtype=dtype, device=device),
            requires_grad=False,
        )
        # One shared expert every token goes through (no routing weight).
        self.shared_w1 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)
        self.shared_w2 = nn.Linear(inter, hidden, bias=False, dtype=dtype, device=device)
        self.shared_w3 = nn.Linear(hidden, inter, bias=False, dtype=dtype, device=device)

    def _selection_logits(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """fp32 sqrtsoftplus scores and the bias-steered selection logits.

        The selection bias picks experts but never scales them (noaux_tc);
        tokens inside image spans use ``gate.bias_vl`` instead of
        ``gate.bias`` -- a per-token modality switch (model.py:809-823).
        """
        scores = F.linear(x.float(), self.gate.weight.float())
        scores = F.softplus(scores).sqrt()
        bias = self.e_score_correction_bias
        if input_ids is not None and self.gate_bias_vl_loaded:
            image_mask = input_ids.reshape(-1) == self.image_token_id
            bias = torch.where(image_mask.unsqueeze(-1), self.gate_bias_vl, self.e_score_correction_bias)
        return scores + bias, scores

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        # The decoder layer delegates the FFN RMSNorm here when the MoE tier is
        # used (``post_norm``), so the norm runs exactly once and the layer no
        # longer pre-normalises. This path never gathers across DP, so the eager
        # norm suffices: ``_selection_logits`` already re-derives the fp32
        # routing input from the normalised activation.
        if post_norm is not None:
            hidden = post_norm(hidden)
        shape = hidden.shape
        x = hidden.reshape(-1, self.cfg.hidden_size)
        logits, scores = self._selection_logits(x, input_ids)
        # The bias picks experts but does not scale them.
        indices = logits.topk(self.topk, dim=-1).indices
        weights = scores.gather(1, indices)
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.routed_scaling
        # This is the fp8/none serving path, not just a CPU reference.
        # Sort local routes once instead of scanning all routes for each expert.
        # Only active experts launch GEMMs; keep ascending expert order for the
        # fp32 accumulation, and stable token order within each expert.
        flat_experts = indices.reshape(-1)
        local_routes = torch.where(
            (flat_experts >= self.start_expert_id) & (flat_experts < self.start_expert_id + self.num_experts_per_rank)
        )[0]
        sorted_experts, permutation = flat_experts[local_routes].sort(stable=True)
        sorted_routes = local_routes[permutation]
        sorted_rows = sorted_routes // self.topk
        sorted_weights = weights.reshape(-1)[sorted_routes]
        active_experts, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
        # Interim eager implementation: copy segment metadata to the host once,
        # rather than synchronizing once per expert. Active experts still run
        # serially; this is not graph-capturable or a final fused MoE path.
        # A grouped GEMM replacement must preserve fp32 asymmetric SwiGLU and
        # apply routing weights BEFORE the down projection (not at combine).
        segments = torch.stack((active_experts - self.start_expert_id, counts), dim=1).cpu().tolist()
        y = torch.zeros_like(x, dtype=torch.float32)
        offset = 0
        for local_slot, count in segments:
            rows = sorted_rows[offset : offset + count]
            expert_input = x[rows]
            expert_out = _swiglu_clamped(
                expert_input,
                F.linear(expert_input, self.experts_w1[local_slot]),
                F.linear(expert_input, self.experts_w3[local_slot]),
                _StackedLinear(self.experts_w2, local_slot),
                self.swiglu_limit,
                sorted_weights[offset : offset + count],
            )
            y.index_add_(0, rows, expert_out.float())
            offset += count
        if self.ep_size > 1:
            from xllm.python import distributed as _distributed

            _distributed.moe_ep_all_reduce(y)
        y += self._shared_forward(x).float()
        return y.type_as(x).view(shape)

    def _shared_forward(self, x: torch.Tensor) -> torch.Tensor:
        return _swiglu_clamped(x, self.shared_w1(x), self.shared_w3(x), self.shared_w2, self.swiglu_limit, None)


class _StackedLinear(nn.Module):
    """Adapter so a stacked expert weight row can be used as the w2 module."""

    def __init__(self, stacked: nn.Parameter, index: int) -> None:
        super().__init__()
        self.weight = stacked[index]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight)


def _swiglu_with_clamp(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Eager fp32-upcast clamped SwiGLU (historical DSv3.x/V4.1 semantics).

    The dsv41 shared-activation tests pin the eager clamp contract against
    this reference (spy + golden reference).
    """
    gate, up = x.chunk(2, dim=-1)
    if 0.0 < limit < 1_000_000.0:
        gate = gate.float().clamp_max(limit)
        up = up.float().clamp(min=-limit, max=limit)
        return (torch.nn.functional.silu(gate) * up).to(x.dtype)
    return kernels.silu_and_mul(x)


class DeepseekV41SharedMLP(DeepseekV3MLP):
    """V4.1 shared expert: W8A8 fused MLP whose operator tail can carry the clamp.

    ``forward_dequant_swiglu_quant`` overrides the base so a clamped call can
    take the operator's GROUPED branch -- the DENSE branch ignores
    ``clamp_limit`` on device, so the clamp is only effective with the
    ``group_index`` this override builds when ``grouped_activation=True``.
    """

    def forward_dequant_swiglu_quant(
        self,
        x: torch.Tensor,
        tp_reduce_add: torch.Tensor | None = None,
        *,
        clamp_limit: float = 0.0,
        grouped_activation: bool = False,
    ) -> torch.Tensor:
        x_int8, pertoken = kernels.dynamic_quant(x)
        gate_up = self.gate_up_proj.forward_accumulated(x_int8)
        if grouped_activation:
            # One group spanning every row: ``group_index[i]`` is the row count
            # of group i, int64 and 1-D, with the scale as ``[groups, 2H]``.
            # CAPTURE SAFETY: a host->device copy must not happen inside an ACL
            # graph capture; park the descriptor in a graph-owned buffer via
            # ``get_execution_buffer`` (pre-capture warmup runs the factory once,
            # capture and replays read the resident storage). Eager without an
            # execution state builds it inline.
            rows = gate_up.shape[0]
            if get_forward_context_or_none() is None:
                group_index = torch.tensor([rows], dtype=torch.int64, device=gate_up.device)
            else:
                group_index = get_execution_buffer(
                    ("SHARED_ACT_GROUP_INDEX", gate_up.device, rows),
                    lambda: torch.tensor([rows], dtype=torch.int64, device=gate_up.device),
                )
            act_int8, act_scale = kernels.dequant_swiglu_quant(
                gate_up,
                self.gate_up_proj.weight_scale.reshape(1, -1),
                pertoken,
                group_index=group_index,
                clamp_limit=clamp_limit,
            )
        else:
            act_int8, act_scale = kernels.dequant_swiglu_quant(
                gate_up,
                self.gate_up_proj.weight_scale,
                pertoken,
                clamp_limit=clamp_limit,
            )
        out = self.down_proj.forward_quantized(act_int8, act_scale)
        return self._reduce_output(out, tp_reduce_add)


class DeepseekV41W8A8MoE(DeepseekV4MoE):
    """Ascend W8A8-export MoE (existing V4 kernel path, NPU-only).

    Keeps the bias-based ``moe_gating_top_k_hash`` routing and the grouped
    W8A8 expert GEMMs; the visual-token ``bias_vl`` selection bias of the
    official checkpoint is loaded but only applied by the torch MoE (the
    multimodal path lands with the vision phase).

    The gate does not consume ``input_ids``: ``_gate_logits`` deletes them and
    this class rejects hash layers.
    """

    # Shared-expert MLP carries the grouped-clamped fused activation
    # (``DeepseekV41SharedMLP``); the base class keeps ``DeepseekV3MLP``.
    _SHARED_MLP_CLS = DeepseekV41SharedMLP

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if cfg.n_hash_layers != 0:
            raise ValueError("DeepSeek-V4.1 removed hash routing; n_hash_layers must be 0")
        super().__init__(cfg, layer_id, dtype, device)
        # ``--enable_fused_mc2`` (PyCausalLM::build_config_dict): gates the
        # decode MC2 dispatch/combine path; guard declines log one-shot.
        self.enable_fused_mc2 = cfg.enable_fused_mc2
        # Deferred until the vision phase: the W8A8 expert path never reads
        # image_token_id or gate_bias_vl; both are staged for the torch-MoE
        # multimodal path only (see the class docstring).
        self.image_token_id = cfg.image_token_id
        self.gate_bias_vl = nn.Parameter(
            torch.empty(cfg.n_routed_experts, dtype=torch.float32, device=device),
            requires_grad=False,
        )
        self.gate_bias_vl_loaded = False

    def _gate_logits(
        self,
        gate_input: torch.Tensor,
        gate_input_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        del gate_input_ids
        return self.gate(gate_input)

    def _apply_post_norm(
        self,
        hidden: torch.Tensor,
        post_norm: nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fused RMSNorm+cast on the post-gather MoE input, eager if unguarded.

        Runs AFTER ``DeepseekV4MoE.forward``'s DP pad+all-gather, so the op's
        widened result describes the exact rows routed and consumed here (this
        is what makes the fusion valid under ``dp_size > 1``). When the guard
        declines -- a missing op or an unsupported dtype/device/shape --
        the base eager norm runs once and the gate widens it, matching the
        pre-fusion behaviour bit for bit.
        """
        weight = post_norm.weight
        if _rms_norm_cast_supported(hidden, weight):
            _log_rms_norm_cast_active()
            return _get_rms_norm_cast_op()(hidden, weight, post_norm.eps)
        _log_rms_norm_cast_fallback(hidden, weight)
        return super()._apply_post_norm(hidden, post_norm)

    def _run_shared_experts(self, hidden: torch.Tensor) -> torch.Tensor:
        """Shared expert with the fused clamped SwiGLU tail, eager if unguarded.

        The fused clamped SwiGLU tail routes the shared expert through the
        existing W8A8 fused MLP (``DeepseekV41SharedMLP.forward_dequant_swiglu_quant``,
        the same tail the V3.2 EP shared expert runs) and passes the model's
        ``swiglu_limit`` as the operator's ``clamp_limit`` -- the operator's
        default is ``0.0`` (no clamp), which is the gap this fusion closes. When
        the guard declines (an unsupported module/shape/dtype/device/
        limit) the base eager ``DeepseekV3MLP`` forward runs unchanged, so the
        fallback is bit-for-bit the pre-fusion behaviour.

        The call is GROUPED (``grouped_activation=True``) because that is the only
        branch of ``aclnnDequantSwigluQuantV2`` that honours ``clamp_limit`` on
        device: the dense call was measured to return the unclamped reference bit
        for bit, so a dense call with a limit set would silently drop the clamp --
        the very defect this fusion must not paper over.
        """
        shared = self.shared_experts
        limit = float(self.cfg.swiglu_limit)
        if _fused_shared_act_supported(hidden, shared, limit):
            _log_fused_shared_act_active()
            # ``forward_dequant_swiglu_quant`` calls the operator with
            # ``swiglu_mode=1`` (the wrapper default). Keep it: that is what makes
            # a V2-less device ABORT on the clamp contract instead of silently
            # dropping the clamp (dequant_swiglu_quant.cpp:124).
            return shared.forward_dequant_swiglu_quant(
                hidden,
                clamp_limit=limit,
                grouped_activation=True,
            )
        _log_fused_shared_act_fallback(hidden, shared, limit)
        return super()._run_shared_experts(hidden)

    def _mc2_ep_size(self) -> int:
        return self.cfg.ep_size if self.cfg.ep_size > 0 else self.cfg.tp_size

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """Decode-time MC2 expert path, or the portable base forward.

        ``--enable_fused_mc2`` (the C++ gflag bridged by
        ``PyCausalLM::build_config_dict``; default off) replaces ONLY the decode
        expert step with the fused all-to-all dispatch/combine kernels and folds
        the EP reduction into ``combine_v2``. Every guard failure falls back to
        ``DeepseekV4MoE.forward`` verbatim, so OFF and every unsupported
        configuration stay bit-for-bit identical to the base path.

        Under ``dp_size > 1`` the override runs on this replica's OWN rows, like
        the C++ mode-0 EP2 branch (``fused_moe.cpp:2853-2874``): no DP
        pad/all-gather and no slice back. The fused post-norm, gate, dispatch
        and shared expert all see the local row set, and ``combine_v2`` scatters
        each source rank's partial back to the target rank (it is not an EP
        all-reduce). ``global_bs=0`` matches C++ for ``dp>1`` and no
        ``x_active_mask`` is passed, because every dispatched row is a real local
        token (there is no fake pad row to drop).

        The guard is evaluated identically on every rank (it keys on host-planned
        metadata and static module state, never a rank-local tensor value), which
        the MC2 kernels require: dispatch and combine are collectives over the
        whole ``moe_ep`` group.
        """
        if self.enable_fused_mc2:
            reason = self._mc2_guard_reason(hidden)
            if reason is None:
                ep_rank = self._mc2_ep_rank(hidden)
                _log_moe_mc2_collective(ep_rank, self._mc2_ep_size())
                _log_moe_mc2_active(ep_rank, self._mc2_ep_size())
                return self._forward_mc2_decode(hidden, post_norm)
            # Pure logging: the draft owns its own one-shot latch so Fix-1's exit
            # is observable even after the target's latch was consumed. The
            # branch keys on the same predicate the guard used, so it never
            # changes which path runs.
            if _moe_mc2_draft_veto_applies(self):
                _log_moe_mc2_draft_fallback(reason, self.enable_fused_mc2)
            else:
                _log_moe_mc2_fallback(reason, self.enable_fused_mc2)
        return super().forward(hidden, input_ids, post_norm)

    def _mc2_ep_rank(self, hidden: torch.Tensor) -> int:
        from xllm.python import distributed as _distributed

        return _distributed.moe_ep_rank(hidden.device)

    def _mc2_guard_reason(self, hidden: torch.Tensor) -> str | None:
        """Return why the MC2 decode path cannot run, or ``None`` when it can.

        Every clause is either static module/config state or host-planned
        metadata that is identical on all ranks -- never a rank-local tensor
        value -- so the decision is graph-capture safe and collective-symmetric.
        ``dispatch_v2``/``combine_v2`` are collectives over the whole ``moe_ep``
        group: a rank that skips a layer hangs every peer, so this must not
        diverge. The step/module clauses precede the platform and
        communicator probes so their reasons stay observable on CPU-only hosts.

        Under ``dp > 1`` the step clause is the shared ``all_dp_ranks_are_decode``
        rather than the rank-local ``_moe_mc2_decode_step``: an empty DP rank is
        dummy locally (``q_max_seq_len == 0`` -> ``is_dummy``) but C++ forces it
        to ``dp_is_decode=1`` (``llm_engine.cpp:1460-1470``) so it runs in
        lockstep with its peers. The DP dispatch no longer pads a fake row: every
        rank dispatches exactly its own rows, and an empty DP rank is rejected up
        front because the operator cannot take ``bs == 0`` (see the token-count
        clause). The host-planned flag keeps all ranks on the same branch. Letting
        the local ``is_dummy`` decide would make that rank fall back while its
        peers dispatch/combine -- a collective split and a hang.
        """
        # Static module state, all-rank identical: the DSpark draft's row
        # counts are uneven by construction (busy rank = one sequence's whole
        # block; empty DP shard = one dummy row), but ``global_bs=0`` asserts
        # equal rows per rank, so each rank would infer a different window slot
        # stride/capacity and the cross-rank writes corrupt memory (UB in the
        # downstream grouped ops). The draft instance is marked at construction
        # (``DeepseekV41DSparkModel``); the target never is. Kept ahead of every
        # probe so the reason is observable on a CPU-only host.
        if _moe_mc2_draft_veto_applies(self):
            return "DSpark draft MoE (uneven per-rank rows break global_bs=0)"
        ctx = get_forward_context_or_none()
        metadata = ctx.metadata if ctx is not None else None
        if self.dp_size > 1:
            if not _moe_mc2_all_dp_ranks_decode(metadata):
                return "not all DP ranks are decode"
        elif not _moe_mc2_decode_step():
            return "not a (non-dummy) decode step"
        if self.hash_layer:
            return "hash-routed layer"
        if self.w4a8_dynamic:
            return "W4A8 expert path"
        if self.moe_tp_size != 1:
            return f"moe_tp_size={self.moe_tp_size} (MC2 requires 1)"
        ep_size = self._mc2_ep_size()
        if ep_size <= 1:
            return f"ep_size={ep_size} (MC2 needs a multi-rank EP group)"
        if self.dp_size > 1:
            # The MC2 all-to-all replaces the base EP all-reduce, so every DP
            # rank must reach it on the same layer. ``all_dp_ranks_are_decode``
            # is the C++ symmetric predicate: uneven batches can leave an empty
            # DP rank on a dummy step, so the local step kind alone is not
            # enough. ``dp_execution_token_counts`` is the host-planned DP layout
            # (identical on every rank); its presence proves the scheduler
            # published the collective-symmetric DP metadata.
            token_counts = getattr(metadata, "dp_execution_token_counts", None)
            if not token_counts or len(token_counts) != self.dp_size:
                return "DP token counts unavailable"
            # P0: the operator infershape rejects ``bs <= 0``
            # (``moe_distribute_dispatch_v2_infershape.cpp:283-289``) and its A2
            # tiling requires ``bs in [1, 256]``
            # (``op_tiling/arch22/moe_distribute_dispatch_v2_tiling_arch22.cpp:507-510``).
            # The base path pads ``max(1, max(token_counts))`` fake rows exactly
            # to dodge the zero, so an uneven DP batch with an empty replica must
            # fall back. ``token_counts`` is the host-planned DP layout, identical
            # on every rank, so this clause is collective-symmetric: all ranks
            # see the same zero row and fall back together.
            if any(int(count) <= 0 for count in token_counts):
                return "an empty DP rank is present"
            # P1-b: same tiling bound; a rank above 256 tokens would be rejected
            # on device, so decline on ALL ranks together (single-replica DP == 1
            # keeps the established explicit ``global_bs`` call).
            if any(int(count) > _MOE_MC2_MAX_LOCAL_TOKENS for count in token_counts):
                return f"a DP rank exceeds the {_MOE_MC2_MAX_LOCAL_TOKENS}-token operator limit"
            # Defensive: ``global_bs=0`` (auto) tells the operators every DP rank
            # holds the SAME row count. The normal target path is uniform by
            # construction -- the ACL-graph runner publishes
            # ``dp_execution_token_counts=(padded_batch_size,) * dp_size`` and
            # states "variable per-rank counts cannot be baked into a graph"
            # (``decode_acl_graph.py:1558-1560``), so the tested c=1/4/8/.../64
            # configurations never trip this clause. A non-uniform layout reaching
            # here would make each rank infer a different window stride/capacity
            # and corrupt the cross-rank writes. ``token_counts`` is the shared
            # host-planned layout (identical on every rank), so all ranks decline
            # together and fall back to the portable path -- safe, merely no MC2.
            row_max = max(int(count) for count in token_counts)
            row_min = min(int(count) for count in token_counts)
            if row_max != row_min:
                return f"uneven DP rows ({row_max} vs {row_min}) break global_bs=0"
        if hidden.dim() != 2:
            return f"hidden shape {tuple(hidden.shape)}"
        if self.dp_size == 1 and hidden.numel() == 0:
            # A single-replica empty step is not a real decode batch. With DP the
            # row count comes from the host-planned metadata, and an empty replica
            # is already rejected by the token-count clause above (the operator
            # needs ``bs >= 1``); a rank-local ``numel`` read must never gate the
            # decision, so it is only consulted here for the DP == 1 case.
            return f"hidden shape {tuple(hidden.shape)}"
        if hidden.dtype != torch.bfloat16:
            return f"hidden dtype {hidden.dtype}"
        if not _moe_mc2_device_supported(hidden):
            return f"device {hidden.device.type}"
        if not _moe_mc2_ops_available():
            return "npu_moe_distribute_{dispatch,combine}_v2 unavailable"
        from xllm.python import distributed as _distributed

        if _distributed.moe_ep_comm_name(hidden.device) is None:
            return "moe_ep communicator name unavailable"
        # P1-a: the communicator name and this rank's dispatch target must come
        # from the SAME source. ``moe_ep_rank`` silently degrades to the Python
        # ``_groups`` rank and then to ``0``; on the V4 family the EP split lives
        # in C++, so a fallback ``0`` would dispatch every rank to rank 0. If the
        # native rank is unavailable we cannot trust the target, so decline.
        if _distributed.moe_ep_rank_from_native(hidden.device) is None:
            return "moe_ep rank unavailable from native source"
        return None

    def _forward_mc2_decode(
        self,
        hidden: torch.Tensor,
        post_norm: nn.Module | None = None,
    ) -> torch.Tensor:
        """Decode expert step through the MC2 all-to-all dispatch/combine kernels.

        The local expert math is the SAME W8A8 chain the portable path runs
        (``kernels_npu/moe.py:_grouped_moe_with_selected_experts_impl``) --
        ``dynamic_quant`` -> int32 gate-up grouped GEMM -> fused
        ``dequant_swiglu_quant`` -> grouped down GEMM -- but the token routing is
        the ``*_distribute_*_v2`` pair instead of
        ``npu_moe_init_routing_v2``/``npu_moe_token_unpermute``:

          * dispatch gathers each ``(token, expert)`` pair to the rank that owns
            the expert (global expert ids, no per-rank masking);
          * combine scatters each source rank's expert partial back to the row
            owner and accumulates them there (a scatter + ``CopyAdd``, NOT an EP
            all-reduce of full rows), so this path must NOT also call
            ``moe_ep_all_reduce``.

        The dispatch/combine kwargs mirror the validated reference chain in
        ``xllm_ops/test/python_test/test_dispatch_gmm_combine_decode.py`` and the
        independently verified probe (``quant_mode=0``: the dispatch carries
        bf16 activations and the explicit ``dynamic_quant`` below produces the
        per-token scale), with ``group_tp=""``/``tp_world_size=1`` because MC2
        requires ``moe_tp_size == 1``.
        """
        from xllm.python import distributed as _distributed
        from xllm.python import kernels

        device = hidden.device
        ep_size = self._mc2_ep_size()
        ep_rank = _distributed.moe_ep_rank(device)
        group_ep = _distributed.moe_ep_comm_name(device)
        if group_ep is None:
            raise RuntimeError("MC2 decode path entered without a moe_ep communicator name")

        # 0) Batch layout. ``dp == 1`` keeps the established single-replica call:
        #    the whole batch is local and C++ derives the global batch as
        #    ``local_bs * ep_world_size`` (an explicit value, ``fused_moe.cpp``
        #    :2652). Under ``dp > 1`` we do NOT pad/all-gather: like the C++
        #    mode-0 EP2 branch (``fused_moe.cpp:2853-2874``) each rank dispatches
        #    exactly its own rows, and ``global_bs = 0`` (auto) matches C++
        #    (``fused_moe.cpp:2653-2659``), which prefers the operator's inferred
        #    global batch over an explicit sum that is wrong for uneven DP
        #    batches. No ``x_active_mask`` is passed: every dispatched row is a
        #    real local token, and C++ only sets the mask for EPLB
        #    (``fused_moe.cpp:2635-2645``), which this path does not use.
        global_bs = hidden.shape[0] * ep_size if self.dp_size == 1 else 0
        x_active_mask: torch.Tensor | None = None

        # 1) FFN RMSNorm on this rank's rows (``_apply_post_norm`` is the V4.1
        #    fused RMSNorm+cast; the guard already vetted the input). The base
        #    class runs it after its DP gather, but the norm is row-wise, so on
        #    the local rows it is identical while feeding the dispatch directly.
        if post_norm is not None:
            hidden, gate_input = self._apply_post_norm(hidden, post_norm)
        else:
            gate_input = hidden.to(torch.float32)

        # 2) Gate + top-k exactly as the portable path. There is NO EP-local
        #    weighting: dispatch routes unconsumed (token, expert) pairs away.
        logits = self._gate_logits(gate_input, None)
        norm_type = {"softmax": 0, "sigmoid": 1, "sqrtsoftplus": 2}.get(self.scoring_func, 2)
        renorm = 0 if norm_type == 2 else 1
        bias = getattr(self, "e_score_correction_bias", None)
        topk_weights, topk_idx, _ = kernels.moe_gating_top_k_hash(
            x=logits,
            k=self.topk,
            bias=bias,
            input_ids=None,
            tid2eid=None,
            k_group=1,
            group_count=1,
            routed_scaling_factor=self.routed_scaling,
            eps=1e-20,
            group_select_mode=1,
            renorm=renorm,
            norm_type=norm_type,
            out_flag=False,
        )
        expert_ids = topk_idx.to(torch.int32)
        expert_scales = topk_weights.to(torch.float32)

        # 3) Dispatch -> local grouped W8A8 experts -> combine, on this rank's
        #    own rows. ``expert_token_nums`` is the per-expert received count the
        #    operators derive themselves, so it stays consistent with
        #    ``global_bs=0`` without any Python capacity arithmetic; ``combine_v2``
        #    scatters each source rank's partial back to the row owner and
        #    accumulates it there. Both ops are fed the SAME host ``global_bs``
        #    (explicit ``local_bs * ep_size`` at dp == 1, the auto ``0`` at
        #    dp > 1); there is no second independent source, so no runtime
        #    dispatch/combine agreement check is possible.
        (
            expand_x,
            dynamic_scales,
            assist_info_for_combine,
            expert_token_nums,
            ep_send_counts,
            tp_send_counts,
            expand_scales,
        ) = _moe_mc2_dispatch(
            x=hidden,
            expert_ids=expert_ids,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=group_ep,
            ep_world_size=ep_size,
            ep_rank_id=ep_rank,
            moe_expert_num=self.num_total_experts,
            group_tp="",
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=int(self.cfg.n_shared_experts),
            shared_expert_rank_num=0,
            quant_mode=0,
            global_bs=global_bs,
            expert_token_nums_type=1,
        )
        del dynamic_scales  # unused: quant_mode=0 leaves the dispatch in bf16.
        sorted_hidden_i8, pertoken_scale = kernels.dynamic_quant(expand_x)
        group_list = expert_token_nums[: self.num_experts_per_rank].to(torch.int64)
        # M4: the same shape contract the portable chain enforces in
        # ``kernels_npu/moe.py:_grouped_moe_with_selected_experts_impl``. The
        # dispatch and this rank's expert shard must agree, and the routed rows
        # must fit the dispatched buffer; otherwise the grouped GEMM reads past
        # its slice (out-of-bounds device read). Fail loud with the shapes rather
        # than silently corrupting memory.
        #
        # PREMISE: EPLB is OFF. ``num_experts_per_rank`` pins the physical local
        # expert shard only without expert-level balancing; with EPLB (redundant
        # experts, a drifting per-rank expert count) both clauses below would need
        # an EPLB short-circuit or an EPLB-aware expert window. This path does not
        # enable EPLB (no ``ep_use_ep_level_balance``; ``global_bs``/mask are the
        # non-EPLB values set above), so the shard stays static.
        #
        # Capture safety: this decode chain is recorded by the ACL-graph runner
        # (``decode_acl_graph``), so any sync in it is illegal. The ``experts_w13``
        # shape check is pure Python over tensor-shape metadata (no device op, no
        # device->host read), so it stays active inside the capture. The
        # ``group_list.sum()`` value check DOES read tensor content -- a D2H sync
        # that is both illegal in the capture region and would bake the reduction
        # into the graph -- so it is gated on the existing capture predicate
        # (``capturing_acl_graph``). It still fires on every non-capture execution
        # (eager and the pre-capture warmup); replay re-runs no Python, so the
        # captured graph never carries the read.
        if self.experts_w13.shape[0] != self.num_experts_per_rank:
            raise RuntimeError(
                f"MC2 decode: experts_w13 has {self.experts_w13.shape[0]} local experts but "
                f"num_experts_per_rank={self.num_experts_per_rank} "
                f"(experts_w13 shape={tuple(self.experts_w13.shape)})"
            )
        if not capturing_acl_graph():
            grouped_rows = int(group_list.sum())
            if grouped_rows > expand_x.shape[0]:
                raise RuntimeError(
                    f"MC2 decode: expert groups cover {grouped_rows} rows but dispatch produced "
                    f"expand_x shape={tuple(expand_x.shape)} "
                    f"(num_experts_per_rank={self.num_experts_per_rank}); the grouped GEMM would "
                    "read past the dispatched rows"
                )
        gemm1_out = _moe_mc2_grouped_matmul(
            x=[sorted_hidden_i8],
            weight=[self.experts_w13],
            scale=None,
            per_token_scale=None,
            split_item=2,
            group_type=0,
            group_list_type=1,
            group_list=group_list,
            output_dtype=torch.int32,
        )
        act_i8, act_pt = kernels.dequant_swiglu_quant(
            x=gemm1_out,
            weight_scale=self.experts_w13_scale,
            activation_scale=pertoken_scale,
            bias=None,
            quant_scale=None,
            quant_offset=None,
            group_index=group_list,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=1,
            clamp_limit=self.cfg.swiglu_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )
        local_out = _moe_mc2_grouped_matmul(
            x=[act_i8],
            weight=[self.experts_w2],
            scale=[self.experts_w2_scale],
            per_token_scale=[act_pt],
            split_item=2,
            group_type=0,
            group_list_type=1,
            group_list=group_list,
            output_dtype=hidden.dtype,
        )
        routed_out = _moe_mc2_combine(
            expand_x=local_out,
            expert_ids=expert_ids,
            assist_info_for_combine=assist_info_for_combine,
            ep_send_counts=ep_send_counts,
            expert_scales=expert_scales,
            x_active_mask=x_active_mask,
            group_ep=group_ep,
            ep_world_size=ep_size,
            ep_rank_id=ep_rank,
            moe_expert_num=self.num_total_experts,
            tp_send_counts=tp_send_counts,
            expand_scales=expand_scales,
            group_tp="",
            tp_world_size=1,
            tp_rank_id=0,
            expert_shard_type=0,
            shared_expert_num=int(self.cfg.n_shared_experts),
            shared_expert_rank_num=0,
            global_bs=global_bs,
        )

        # 4) Shared expert on the same local (post-norm) rows, then add. The
        #    shared expert is per-row, so this matches the base class; the routed
        #    partials were already scattered-and-accumulated by ``combine_v2`` and
        #    the guard pins ``moe_tp_size == 1``, so no explicit reduction runs
        #    here (doing so would double the routed contribution).
        shared_out = self._run_shared_experts(hidden)
        if self.shared_expert_gate_is_loaded:
            shared_gate = torch.sigmoid(self.shared_expert_gate(hidden))
            shared_out = shared_gate * shared_out
        return routed_out + shared_out


# ---------------------------------------------------------------------------
# Decoder layer + model
# ---------------------------------------------------------------------------


class DeepseekV41DecoderLayer(nn.Module):
    """DeepSeek-V4.1 decoder layer with the single-pass mHC shift.

    Attention collapses with the PREVIOUS block's ``ffn_pre`` and the FFN
    with THIS block's ``attn_pre``; the block returns its ``ffn_pre`` for the
    next block's attention (model.py:968-994).
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        layer_id: int,
        dtype: torch.dtype,
        device: torch.device,
        mode: CSA2LayerMode,
        quant_mode: str = V41_QUANT_NONE,
        engram_layout: EngramLayout | None = None,
        engram_storage: str = ENGRAM_STORAGE_FP8,
        engram_gate_unrotate: torch.Tensor | None = None,
        engram_offload: bool = False,
        engram_dp_shared_memory: bool = False,
        engram_dp_group: Any | None = None,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.mode = mode
        self.hc = DeepseekV41HyperConnection(cfg, dtype, device)
        self.input_layernorm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        self.self_attn = DeepseekV41Attention(cfg, layer_id, dtype, device, mode, quant_mode)
        self.post_attention_layernorm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # Engram: gated n-gram lookup on the configured backbone layers only.
        self.engram: Engram | None = None
        if engram_layout is not None and layer_id in engram_layout.layer_ids:
            self.engram = Engram(
                engram_layout,
                engram_layout.layer_ids.index(layer_id),
                cfg.hidden_size,
                cfg.hc_mult,
                cfg.rms_norm_eps,
                cfg.tp_size,
                cfg.tp_rank,
                dtype,
                device,
                storage=engram_storage,
                gate_unrotate=engram_gate_unrotate,
                offload=engram_offload,
                dp_shared_memory=engram_dp_shared_memory,
                dp_group=engram_dp_group,
            )
        # Dense vs MoE by first_k_dense_replace / moe_layer_freq.
        is_dense = (layer_id < cfg.first_k_dense_replace) or (
            cfg.moe_layer_freq > 1 and layer_id % cfg.moe_layer_freq != 0
        )
        if is_dense:
            if quant_mode == V41_QUANT_ASCEND:
                self.mlp: nn.Module = DeepseekV3MLP(
                    cfg,
                    cfg.moe_intermediate_size,
                    dtype,
                    device,
                    swiglu_limit=cfg.swiglu_limit,
                )
            else:
                self.mlp = DeepseekV41TorchMLP(cfg, dtype, device)
        elif quant_mode == V41_QUANT_ASCEND:
            self.mlp = DeepseekV41W8A8MoE(cfg, layer_id, dtype, device)
        else:
            self.mlp = DeepseekV41MoE(cfg, layer_id, dtype, device)

    def forward(
        self,
        hidden: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        pre_mix: torch.Tensor | None = None,
        engram_hashes: torch.Tensor | None = None,
        engram_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wait_for_layer_load(self.layer_id)
        if pre_mix is None:
            pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.hc.hc_mult)
        # Engram injection happens between the previous sublayer's post and
        # this block's pre, on the full hc stream, so the mix coefficients
        # see the injected stream (reference model.py:336-360).
        # ``engram_hashes`` is None on an offloaded graph capture/replay, where
        # the rows were staged by the pre-graph host pass and only the gate
        # mask is computed in-graph (engram.embed(None) reads the staged rows).
        if self.engram is not None and (engram_hashes is not None or engram_mask is not None):
            hidden = self.engram(
                hidden,
                None if engram_hashes is None else engram_hashes[:, self.engram.layer_hash_index],
                engram_mask,
            )
        residual = hidden
        if self.hc._hc_pre_mode == _HC_PRE_MODE_FUSED:
            # One op call replaces the hc_mixes + hc_pre collapse pair; the op
            # collapses with the external pre_mix and returns this block's pre.
            x, attn_pre, attn_post, attn_comb = self.hc.hc_pre_fused(
                hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base, pre_mix
            )
        else:
            attn_pre, attn_post, attn_comb = self.hc.hc_mixes(
                hidden, self.hc.hc_attn_fn, self.hc.hc_attn_scale, self.hc.hc_attn_base
            )
            x = self.hc.hc_pre(hidden, pre_mix)
        x = self.input_layernorm(x)
        x = self.self_attn(x, positions, cos_sin_cache)
        hidden = self.hc.hc_post(x, residual, attn_post, attn_comb)

        residual = hidden
        if self.hc._hc_pre_mode == _HC_PRE_MODE_FUSED:
            x, ffn_pre, ffn_post, ffn_comb = self.hc.hc_pre_fused(
                hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base, attn_pre
            )
        else:
            ffn_pre, ffn_post, ffn_comb = self.hc.hc_mixes(
                hidden, self.hc.hc_ffn_fn, self.hc.hc_ffn_scale, self.hc.hc_ffn_base
            )
            x = self.hc.hc_pre(hidden, attn_pre)
        post_norm = self.post_attention_layernorm
        if isinstance(self.mlp, (DeepseekV41MoE, DeepseekV41W8A8MoE)):
            # Delegate the FFN RMSNorm to the MoE. ``DeepseekV4MoE.forward``
            # pads+all-gathers ``hidden`` across DP BEFORE it takes the
            # gate/expert input, so the norm must run there for the fused op's
            # fp32 routing input to describe the gathered rows. The MoE applies
            # it exactly once (fused RMSNorm+cast when available, eager norm
            # otherwise) and never double-normalises.
            x = self.mlp(x, input_ids, post_norm=post_norm)
        else:
            # Dense FFN keeps the eager RMSNorm and its single-argument MLP.
            x = post_norm(x)
            x = self.mlp(x)
        hidden = self.hc.hc_post(x, residual, ffn_post, ffn_comb)
        return hidden, ffn_pre


@dataclass
class _EngramHashGraph:
    """Hash workload and request storage retained for one target graph entry."""

    metadata: Any
    dsa_metadata: Any
    input_ids: torch.Tensor
    positions: torch.Tensor
    graph: Any
    hashes: torch.Tensor
    gate_mask: torch.Tensor
    ready_event: Any


class DeepseekV41Model(nn.Module):
    """DeepSeek-V4.1 transformer body.

    The layer loop threads the mHC ``pre_mix``: the residual stream starts
    as ``hc_mult`` identical copies of the embedding with a one-hot mix on
    copy 0, and after the last block the final merge is the learned
    ``hc_pre(h, last_ffn_pre)`` weighted sum (NOT a mean; model.py:1268-1269).
    Each layer selects its RoPE table by its CSA2 mode: ratio > 0 layers use
    the compress table (theta 160000 + YaRN), ratio-0 layers the plain
    theta-10000 table with YaRN off.
    """

    def __init__(
        self,
        cfg: DeepseekV41Config,
        dtype: torch.dtype,
        device: torch.device,
        quant_mode: str = V41_QUANT_NONE,
        config: dict | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.quant_mode = quant_mode
        self._engram_staging_executor: ThreadPoolExecutor | None = None
        self._engram_staging_stream: Any = None
        self._engram_hashes_host: torch.Tensor | None = None
        self._engram_hash_graphs: dict[int, _EngramHashGraph] = {}
        self._engram_hash_capture_stream: Any = None
        self._engram_rows_ready: dict[int, Any] = {}
        self.csa2_plan = CSA2LayerPlan.build(
            cfg.n_layers,
            cfg.compress_ratios,
            cfg.kv_source_layer_ids,
            cfg.index_source_layer_ids,
        )
        validate_swa_bounded_replay(cfg, self.csa2_plan)
        # Engram layout (None when the checkpoint has no engram layers).
        self.engram_layout: EngramLayout | None = None
        self.engram_hash: EngramHashState | None = None
        if cfg.engram_layer_ids:
            if cfg.engram_pad_token_id < 0:
                raise ValueError(
                    "DeepSeek-V4.1 engram is enabled (engram_layer_ids="
                    f"{cfg.engram_layer_ids}) but engram_pad_token_id is unset"
                )
            self.engram_layout = EngramLayout(
                cfg.engram_layer_ids,
                cfg.engram_num_embeddings,
                cfg.engram_max_ngram_size,
                cfg.engram_n_heads,
                cfg.engram_head_dim,
                cfg.engram_vocab_size,
                cfg.engram_compressed_vocab_size,
                cfg.engram_pad_token_id,
            )
            # The sidecar-derived buffers (token_map / primes / offsets /
            # multipliers) load on CPU; move them onto the model device.
            self.engram_hash = EngramHashState.from_config(
                self.engram_layout,
                {**(config or {}), "vocab_size": cfg.vocab_size},
            ).to(device)
        # Engram runtime format per checkpoint family: the ascend (W8A8)
        # export stores the tables as symmetric int8 + fp32 group scales and
        # rotates the residual stream (QuaRot), so the gate dot needs the
        # un-rotated stream; the official FP8/bf16 releases need neither.
        self.engram_storage = ENGRAM_STORAGE_INT8 if quant_mode == V41_QUANT_ASCEND else ENGRAM_STORAGE_FP8
        # Host offload (engram_config.cpu_offload): the int8 tables stay in
        # pageable CPU host memory instead of resident HBM (~27.7GB/rank); only
        # the per-rank staging/scratch buffers are pinned. Only the int8 (W8A8)
        # tables have a CPU decoder; unsupported requests fall back to the HBM
        # path (EngramEmbedding warns), never skipping engram.
        self.engram_offload_active = bool(cfg.engram_layer_ids) and cfg.engram_config.cpu_offload
        if self.engram_offload_active and self.engram_storage != ENGRAM_STORAGE_INT8:
            logger.warning(
                "engram_config.cpu_offload=true requested but engram storage is %r; keeping the table on HBM",
                self.engram_storage,
            )
            self.engram_offload_active = False
        # DP-shared host table: the node-local DP replicas that map the same TP
        # shard share one physical copy. Requires the int8 (W8A8) table and the
        # host-offload path; the gloo sharing group is built before the layers so
        # every engram layer joins the same group. A shared request against
        # non-int8 storage is a hard error, not the HBM-fallback warning above.
        shared_requested = bool(cfg.engram_layer_ids) and cfg.engram_config.dp_shared_memory
        if shared_requested and self.engram_storage != ENGRAM_STORAGE_INT8:
            raise ValueError(
                "engram_config.dp_shared_memory=true requires the int8 (W8A8) engram table "
                f"storage, got {self.engram_storage!r}"
            )
        self.dp_shared_memory = shared_requested and self.engram_offload_active
        self.engram_dp_group: Any | None = None
        if self.dp_shared_memory:
            # Ranks in one DP replica: TP x PP x CP (DCP reuses the TP ranks and
            # adds none), matching vllm's tp*pp*pcp replica size.
            replica_size = cfg.tp_size * cfg.pp_size * cfg.cp_size
            shard_size = engram_dp_shard_size(cfg.dp_size, replica_size)
            if shard_size <= 1:
                raise ValueError(
                    "engram_config.dp_shared_memory=true requires a node-local DP sharing "
                    f"group with more than one rank; got DP={cfg.dp_size}, shard_size={shard_size}."
                )
            # One aggregate /dev/shm precheck for the whole node (the per-storage
            # check would double-count across layers and TP shards). It is
            # deliberately best-effort and per-rank: a short host raises only on
            # the ranks of the failing node, while the other nodes proceed to
            # ``init_engram_dp_group`` and block on ``new_group`` until they time
            # out. That is bounded to launch time and every rank fails the job
            # anyway, so the cheap local check is kept instead of a global vote;
            # the per-storage ``statvfs`` check at create time stays the
            # authoritative gate.
            ensure_dev_shm_capacity(self._engram_shared_node_bytes(cfg.tp_size, replica_size, shard_size))
            # Pass the shard size already derived above (against the live world)
            # so the distributed helper does not recompute it or repeat this
            # engram_config error text at a lower layer.
            self.engram_dp_group = init_engram_dp_group(str(device), cfg.dp_size, shard_size)
        self.engram_gate_unrotate = None
        if cfg.engram_layer_ids and quant_mode == V41_QUANT_ASCEND:
            self.engram_gate_unrotate = load_quarot_gate_unrotate(str((config or {}).get("model_path") or ""), device)
        tp = cfg.tp_size
        self.embed_tokens = HiddenParallelEmbedding(
            cfg.vocab_size, cfg.hidden_size // tp, tp, dtype=dtype, device=device
        )
        self.layers = nn.ModuleList(
            [
                DeepseekV41DecoderLayer(
                    cfg,
                    i,
                    dtype,
                    device,
                    self.csa2_plan.mode(i),
                    quant_mode,
                    self.engram_layout,
                    engram_storage=self.engram_storage,
                    engram_gate_unrotate=self.engram_gate_unrotate,
                    engram_offload=self.engram_offload_active,
                    engram_dp_shared_memory=self.dp_shared_memory,
                    engram_dp_group=self.engram_dp_group,
                )
                for i in range(cfg.n_layers)
            ]
        )
        self.norm = DeepseekV41RMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype, device)
        # RoPE tables: ratio-0 layers use theta 10000 with YaRN OFF; ratio>0
        # layers use the compress theta 160000 + YaRN (factor 16, original
        # 65536, beta 32/1, no temperature correction). Both are fp32,
        # matching the reference's complex64 tables (model.py:680-698).
        self.rotary = _PlainRotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_theta,
            device,
        )
        self.compress_rotary = DeepseekV4RotaryEmbedding(
            cfg.qk_rope_head_dim,
            cfg.max_position_embeddings,
            cfg.rope_scaling_factor,
            cfg.compress_rope_theta,
            cfg.rope_beta_fast,
            cfg.rope_beta_slow,
            cfg.original_max_position_embeddings,
            dtype=torch.float32,
            device=device,
        )
        # DSpark target side: buffer the residual stream of ``layers_to_capture``
        # into the [tokens, n_captured * hidden] tensor the draft's ``main_proj``
        # fuses. Disabled (and free) while the list is empty, which is every
        # non-speculative run. The graph opt-in switches the buffer to the
        # persistent mode the decode graph runner needs; the eager paths then
        # publish it out of band through ``executor.execute``.
        self.aux_hidden_capture = AuxHiddenCapture(
            tuple(cfg.layers_to_capture),
            persistent_buffer=cfg.enable_graph_aux_hidden_states,
        )

    def _layer_rope_cache(self, mode: CSA2LayerMode) -> torch.Tensor:
        """Per-layer RoPE table: compress theta for CSA2 layers, plain otherwise."""
        return self.compress_rotary.cos_sin_cache if mode.is_csa2 else self.rotary.cos_sin_cache

    def restore_engram_history(self, backend: Any, metadata: Any) -> None:
        """Restore PD prompt history before eager execution or graph replay."""
        history = getattr(metadata, "engram_history", ())
        if not history or self.engram_hash is None:
            return
        if len(history) % 2 != 0:
            raise ValueError(f"Engram history must contain [slot, token] pairs, got {len(history)} values")
        cache = backend.swa_history_cache()
        if not self.engram_hash.ensure_cache(cache, cache.shape[1]):
            raise RuntimeError("Cannot restore Engram history before cache binding")
        pairs = torch.tensor(history, dtype=torch.int64).reshape(-1, 2)
        if (
            cache.device.type == "npu"
            and bool(getattr(backend, "supports_draft_to_target_overlap", False))
            and not bool(getattr(metadata, "is_prefill", False))
            and not bool(getattr(metadata, "is_chunked_prefill", False))
        ):
            self.engram_hash.restore_history(pairs, non_blocking=True)
        else:
            self.engram_hash.restore_history(pairs)

    def _compute_engram_hashes(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Hash the token stream once for every engram layer.

        Returns ``(hashes, gate_mask)`` or ``(None, None)`` while the SWA cache
        is unbound (graph memory profiling); the caller then skips engram
        entirely, matching the reference ``ensure_cache`` protocol.
        """
        assert self.engram_hash is not None
        dsa = getattr(metadata, "dsa_metadata", None)
        swa_group = backend.swa_group_metadata(dsa)
        if swa_group is None or not self.engram_hash.ensure_cache(swa_group.cache, swa_group.block_size):
            return None, None
        seq_lens_q = dsa.seq_lens_q.reshape(-1).to(torch.int64)
        q_cu = torch.cat((torch.zeros(1, dtype=torch.int64, device=seq_lens_q.device), seq_lens_q.cumsum(0)))
        dead = image_sentinel_mask(input_ids)
        max_chunk_len = getattr(dsa, "max_query_len", None)
        hashes = self.engram_hash(
            input_ids,
            positions,
            q_cu,
            dead,
            getattr(metadata, "engram_lookback_token_ids", None),
            swa_group.slot_mapping,
            swa_group.block_table,
            swa_group.block_size,
            max_chunk_len=max_chunk_len,
        )
        return hashes, ~dead

    def _replay_engram_hashes(
        self,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self._engram_hash_graphs[id(metadata)]
        if (
            state.metadata is not metadata
            or state.dsa_metadata is not metadata.dsa_metadata
            or state.input_ids.data_ptr() != input_ids.data_ptr()
            or state.positions.data_ptr() != positions.data_ptr()
        ):
            raise RuntimeError("Engram hash graph inputs changed after capture")
        state.graph.replay()
        return state.hashes, state.gate_mask

    def _capture_engram_hashes(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if id(metadata) in self._engram_hash_graphs:
            return self._replay_engram_hashes(metadata, input_ids, positions)
        # Warm up on the first target capture, before asynchronous target input
        # preparation starts. Later steps submit this hash workload with one API call.
        caller_stream = torch.npu.current_stream(input_ids.device)
        if self._engram_hash_capture_stream is None:
            self._engram_hash_capture_stream = torch.npu.Stream(device=input_ids.device)
        stream = self._engram_hash_capture_stream
        stream.wait_stream(caller_stream)
        with torch.npu.stream(stream):
            hashes, gate_mask = self._compute_engram_hashes(backend, metadata, input_ids, positions)
        if hashes is None:
            return None, None
        graph = torch.npu.NPUGraph()
        try:
            with torch.npu.graph(graph, stream=stream):
                hashes, gate_mask = self._compute_engram_hashes(backend, metadata, input_ids, positions)
            # External events allocate and initialize device storage lazily.
            # Warm this event with the graph entry, outside steady-state decode.
            ready_event = torch.npu.ExternalEvent()
        except BaseException:
            # Capture did not publish an entry; release its partial graph while
            # preserving the failure. Do not retry from mutated hash history.
            stream.synchronize()
            graph.reset()
            raise
        caller_stream.wait_stream(stream)
        assert hashes is not None and gate_mask is not None
        self._engram_hash_graphs[id(metadata)] = _EngramHashGraph(
            metadata, metadata.dsa_metadata, input_ids, positions, graph, hashes, gate_mask, ready_event
        )
        return self._replay_engram_hashes(metadata, input_ids, positions)

    def release_graph_inputs(self, metadata: Any) -> None:
        """Discard input graphs owned by an uncached target entry.

        The runner drains target capture work before calling this rollback.
        It never evicts a live target entry while its inputs are in use.
        """
        state = self._engram_hash_graphs.get(id(metadata))
        if state is not None:
            self._engram_hash_capture_stream.synchronize()
            torch.npu.current_stream(state.input_ids.device).synchronize()
            state.graph.reset()
            del self._engram_hash_graphs[id(metadata)]
        self._engram_rows_ready.pop(id(metadata), None)

    def _engram_shared_node_bytes(self, tp_size: int, replica_size: int, shard_size: int, quant_block: int = 32) -> int:
        """Estimate this node's shared host-table footprint (precheck only).

        Summed over every engram layer and every TP shard on the node, then
        scaled by the number of sharing groups that hold a copy of each shard
        (``replicas_per_node // shard_size``). Every TP shard's rows are summed
        explicitly: the head buckets are uneven, so one rank's shard bytes is
        not a sound sample of the others, and ``shard_size < replicas_per_node``
        would under-count the extra per-node copies. Falls back to a
        single-replica footprint when the topology is unavailable; the
        per-storage ``statvfs`` check at create time remains the authoritative
        gate, so an over-estimate here only errs toward a safer precheck.
        """
        assert self.engram_layout is not None
        head_dim = self.engram_layout.head_dim
        per_layer = 0
        for layer_hash_index in range(len(self.engram_layout.layer_ids)):
            head_sizes = self.engram_layout.head_sizes(layer_hash_index)
            part = len(head_sizes) // tp_size
            for shard in range(tp_size):
                start = shard * part
                rows = sum(head_sizes[start : start + part])
                per_layer += estimate_shared_bytes(rows, head_dim, quant_block)
        copies = 1
        topology = world_topology()
        if topology is not None and replica_size > 0 and shard_size > 0:
            my_host = socket.gethostname()
            ranks_on_node = sum(1 for entry in topology if entry.get("hostname") == my_host)
            replicas_per_node = ranks_on_node // replica_size
            if replicas_per_node > 0:
                copies = max(replicas_per_node // shard_size, 1)
        return per_layer * copies

    def _reconcile_engram_offload_active(self) -> bool:
        """All-or-nothing offload gate, run after weight loading.

        A single layer whose host-table load failed demotes the WHOLE model to
        the HBM/device path. A mixed model cannot honor the graph-replay
        contract: the capture branch stages ``hash_ids=None`` for every layer
        and each layer's ``embed(None)`` reads its own staging block, but the
        demoted layer has no offload staging block -- and letting the remaining
        layers keep offloading would make the capture fall back to the
        offload-aware eager path, which does illegal host work inside the
        captured region. Demoting every layer keeps one capture-safe path (the
        all-device baseline). Returns True when a demotion happened.

        A DP-shared table never enters this path: shared layers keep
        ``offload=True`` (``fallback_to_device`` refuses to demote them), so the
        shared load is fail-fast and no per-layer demotion exists to reconcile.
        """
        if not self.engram_offload_active or self.dp_shared_memory:
            return False
        engram_layers = [layer.engram for layer in self.layers if getattr(layer, "engram", None) is not None]
        fell_back = [engram for engram in engram_layers if not engram.embed_tokens.offload]
        if not fell_back:
            return False
        for engram in engram_layers:
            if engram.embed_tokens.offload:
                engram.embed_tokens.fallback_to_device()
        self.engram_offload_active = False
        logger.warning(
            "engram host offload disabled: %d/%d engram layer(s) failed the host-table load; "
            "falling back the whole model to the HBM table path",
            len(fell_back),
            len(engram_layers),
        )
        return True

    def close_host_offload(self) -> None:
        """Release every DP-shared host table this model owns (no-op otherwise).

        Every layer is attempted even when one fails, so a single bad table
        cannot leak the rest; the collected failures are raised together at the
        end (the caller is the C++ destructor, which tolerates the raise).
        """
        executor = getattr(self, "_engram_staging_executor", None)
        if executor is not None:
            executor.shutdown(wait=True)
            self._engram_staging_executor = None
        # Teardown only: finish producer copies and graph consumers before
        # dropping graph-held buffers, external events or host staging storage.
        graphs = getattr(self, "_engram_hash_graphs", {})
        streams = [
            getattr(self, "_engram_staging_stream", None),
            getattr(self, "_engram_hash_capture_stream", None),
        ]
        if graphs:
            state = next(iter(graphs.values()))
            torch.npu.synchronize(state.input_ids.device)
        for stream in streams:
            if stream is not None:
                stream.synchronize()
        for state in graphs.values():
            state.graph.reset()
        graphs.clear()
        getattr(self, "_engram_rows_ready", {}).clear()
        self._engram_staging_stream = None
        self._engram_hash_capture_stream = None
        self._engram_hashes_host = None
        failures: list[str] = []
        for layer_id, layer in enumerate(self.layers):
            engram = getattr(layer, "engram", None)
            if engram is None:
                continue
            try:
                engram.close_host_offload()
            except Exception as exc:  # noqa: BLE001 - aggregated; keep releasing the rest
                failures.append(f"layer {layer_id}: {type(exc).__name__}: {exc}")
        if failures:
            raise RuntimeError("engram host-table close failed for " + "; ".join(failures))

    def configure_engram_graph_capacity(self, max_decode_tokens: int) -> bool:
        """Pre-size every graph-visible engram staging block to the largest bucket.

        The ACL-graph runner calls this at construction with its largest decode
        token-row bucket (``_decode_bucket`` of ``max_seqs_per_batch`` times
        ``num_decoding_tokens``). Capture bakes the staging address into the
        graph, so the block is allocated once and never grown; without this
        floor it would be fixed by the first (smallest) lazily captured bucket,
        and a later larger bucket would raise. No-op unless host offload is
        active. Returns True when at least one layer was configured.
        """
        if not self.engram_offload_active:
            return False
        max_tokens = max(int(max_decode_tokens), 0)
        if max_tokens <= 0:
            return False
        configured = False
        for layer in self.layers:
            engram = getattr(layer, "engram", None)
            if engram is not None and engram.offload:
                engram.embed_tokens.set_graph_capacity_hint(max_tokens)
                configured = True
        return configured

    def engram_offload_warning_fields(self) -> tuple[int, list[Any]]:
        """(engram layer count, graph-capacity hints) for the engram-skip warnings.

        Method form of the module-level ``engram_offload_warning_fields``
        helper (:func:`deepseek_v41_engram.engram_offload_warning_fields`) so the
        decode-graph runner can read these fields off the model by duck typing
        (``getattr(model, "engram_offload_warning_fields", None)``) without the
        runner importing this module -- the runner must not depend on
        ``xllm.python.models``. Delegates to that helper so the model-side and
        runner-side diagnostics cannot drift apart.
        """
        # The bare name resolves to the imported module-level helper, not to
        # this method (a method body never sees class attributes as globals), so
        # this is a delegation, not a recursive call.
        return engram_offload_warning_fields(self)

    def _build_dsa_metadata_for_forward(
        self,
        backend: Any,
        metadata: Any,
        positions: torch.Tensor,
    ) -> None:
        """Publish this forward's DSA metadata (the graph warmup build path).

        Mirrors the ``forward`` branch that runs once per request and is then
        reused across ACL-graph warmup/capture/replay (``pass`` once
        ``metadata.dsa_metadata`` is set). Extracted so the offload staging pass
        can build the metadata before the first capture's warmup (see
        ``stage_engram_rows_outside_graph``).
        """
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", False))
        backend.reset_forward(metadata)
        metadata.dsa_graph_mode = graph_mode
        # Keep the Int64+contiguous normalization here. ``forward`` normalizes
        # ``positions`` the same way before calling this, so on the forward path
        # this line is a no-op and both call sites agree. The pre-graph staging
        # path (``stage_engram_rows_outside_graph``) can reach here before any
        # ``forward`` has run, so this is the only point that guarantees
        # ``dsa_positions`` is int64 and contiguous for that path.
        metadata.dsa_positions = positions.to(torch.int64).contiguous()
        # DeepseekV41Model requires a backend with prepare_dsa_metadata_for_forward;
        # call it directly so a wrong backend fails with a clear AttributeError
        # naming the missing method, rather than a hand-rolled getattr+RuntimeError.
        backend.prepare_dsa_metadata_for_forward(metadata)
        if graph_mode:
            metadata.dsa_graph_mode = True

    def _engram_capture_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Gate mask for the offload capture branch.

        The runner's first-capture gate (``_engram_first_capture_gate``) refuses
        to capture unless the pre-graph staging pass staged this step's rows
        while engram is offloading, so by the time this capture branch runs the
        graph-visible staging block is populated. The mask is therefore always
        ``~image_sentinel_mask(input_ids)``: the capture never skips engram, and
        the fail-fast on a failed staging pass lives in the runner.
        """
        return ~image_sentinel_mask(input_ids)

    def stage_engram_rows_outside_graph(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        """Host-offload staging pass, to be called OUTSIDE the ACL graph.

        Computes the n-gram hashes and gathers every engram layer's rows from
        the pageable host tables into their persistent device staging buffers.
        ACL-graph replay does not re-run Python, so the decode-graph runner
        calls this once per step before ``replay``; the graph then only reads
        the staged rows (see ``DeepseekV41Model.forward``). A no-op for every
        non-offload configuration. The returned bool records that staging
        succeeded this step; the runner's first-capture gate reads it and refuses
        the capture when staging could not complete (e.g. the SWA cache is unbound
        during graph memory profiling), so the capture branch always sees staged
        rows.
        """
        if not self.engram_offload_active or self.engram_hash is None:
            return False
        # The DSA metadata this hashing needs is published by the model
        # forward, but on the FIRST graph capture that forward runs inside
        # ``_capture`` -- after this staging pass. Build it here, exactly as the
        # forward would, so the first capture still gathers the engram rows.
        # Otherwise staging would early-return (``dsa`` None -> no SWA group),
        # the capture branch would bake an engram-less graph, and every replay
        # would reuse it (ACL replay does not re-run Python). ``metadata`` is
        # None only in unit stubs, where there is nothing to build.
        if metadata is not None and getattr(metadata, "dsa_metadata", None) is None:
            self._build_dsa_metadata_for_forward(backend, metadata, positions)
        overlap_graph = (
            bool(getattr(backend, "supports_draft_to_target_overlap", False))
            and input_ids.device.type == "npu"
            and bool(getattr(metadata, "dsa_graph_mode", False))
        )
        if overlap_graph:
            hashes, _ = self._capture_engram_hashes(backend, metadata, input_ids, positions)
        else:
            hashes, _ = self._compute_engram_hashes(backend, metadata, input_ids, positions)
        if hashes is None:
            return False
        # ``engram_offload_active`` is True here (the guard above returned
        # otherwise), so every engram layer takes the host path. Stage all of
        # them through the batched two-phase helper: it queues every layer's
        # D2H copies, drains the device ONCE, then runs each layer's CPU gather +
        # H2D. A per-layer ``prepare_embeddings`` would drain the whole device
        # once per engram layer, so an N-layer pass paid N blocking syncs where
        # N-1 were redundant. The graph-visible staging pass is used because a
        # captured ACL graph reads this block's address on every replay, so it
        # must never move (see ``EngramEmbedding._ensure_graph_staging``).
        pairs = [
            (engram, hashes[:, engram.layer_hash_index].to(torch.int64))
            for layer in self.layers
            if (engram := getattr(layer, "engram", None)) is not None
        ]
        stage_offload_rows(
            [engram for engram, _ in pairs],
            [ids for _, ids in pairs],
            graph_visible=True,
        )
        if overlap_graph:
            event = self._engram_rows_ready.get(id(metadata))
            if event is None:
                event = torch.npu.ExternalEvent()
                self._engram_rows_ready[id(metadata)] = event
            event.record(torch.npu.current_stream(input_ids.device))
            self._engram_hash_graphs[id(metadata)].ready_event.record(torch.npu.current_stream(input_ids.device))
        return True

    def stage_engram_rows_outside_graph_async(
        self,
        backend: Any,
        metadata: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        prepare_metadata: Callable[[], None],
        prepare_history: Callable[[], None] | None = None,
    ) -> GraphInputPreparation | None:
        """Submit target replay while the preceding draft feeds host staging."""
        if not self.engram_offload_active or self.engram_hash is None or input_ids.device.type != "npu":
            return None
        engrams = [layer.engram for layer in self.layers if getattr(layer, "engram", None) is not None]
        device = input_ids.device
        caller_stream = torch.npu.current_stream(device)
        ready = self._engram_hash_graphs[id(metadata)].ready_event
        # The caller stream follows the previous graph consumer. Reset there
        # before publishing inputs_ready, so neither the producer's new record
        # nor the target's wait can observe the preceding replay's ready flag.
        ready.reset(caller_stream)
        inputs_ready = caller_stream.record_event()
        rows_ready = self._engram_rows_ready[id(metadata)]
        if self._engram_staging_executor is None:
            self._engram_staging_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-staging")
            self._engram_staging_stream = torch.npu.Stream(device=device)

        def finish() -> None:
            with (
                torch.inference_mode(),
                torch.npu.device(device),
                torch.npu.stream(self._engram_staging_stream),
            ):
                metadata_ready = False
                try:
                    # Input writes precede this event, including writes queued
                    # behind the preceding draft. Keep metadata and hash
                    # preparation off the replay submission thread.
                    inputs_ready.wait(self._engram_staging_stream)
                    # The preceding draft is ordered after the previous target,
                    # so its input event also protects the captured consumer
                    # wait before this persistent event is reset for reuse.
                    rows_ready.reset(self._engram_staging_stream)
                    prepare_metadata()
                    # The target prefix needs only token/position and DSA.
                    # Its first Engram consumer has a separate captured wait.
                    ready.record(self._engram_staging_stream)
                    metadata_ready = True
                    # History affects Engram hashes, while the target prefix
                    # can start as soon as its input and DSA metadata are ready.
                    if prepare_history is not None:
                        prepare_history()
                    hashes, _ = self._replay_engram_hashes(metadata, input_ids, positions)
                    if hashes is None:
                        raise RuntimeError("Engram hash metadata is missing for target graph replay")
                    # Copy the shared integer hashes once. Ownership and local
                    # row arithmetic can run on the CPU that performs lookup,
                    # avoiding two D2H copies and several device launches per
                    # layer on the draft-to-target dependency chain.
                    if self._engram_hashes_host is None or self._engram_hashes_host.shape != hashes.shape:
                        self._engram_hashes_host = torch.empty_like(hashes, device="cpu", pin_memory=True)
                    host_hashes = self._engram_hashes_host
                    host_hashes.copy_(hashes, non_blocking=True)
                    copied = self._engram_staging_stream.record_event()
                    # Wait only for the queued hash-ID copies. The target has
                    # already been submitted with an external readiness wait;
                    # a whole-device synchronize here would wait on itself.
                    copied.synchronize()
                    for engram in engrams:
                        engram.begin_stage_offload(
                            host_hashes[:, engram.layer_hash_index].to(torch.int64), graph_visible=True
                        )
                        engram.finish_stage_offload(graph_visible=True)
                finally:
                    # Release the device wait even when lookup fails. The
                    # runner joins this future before returning any output and
                    # propagates that failure to the model execution boundary.
                    rows_ready.record(self._engram_staging_stream)
                    if not metadata_ready:
                        ready.record(self._engram_staging_stream)

        # Hash-cache updates use the forward context to preserve the graph
        # path's duplicate-slot semantics in this worker thread.
        context = copy_context()
        # The target captures its metadata wait once, just like its Engram-row
        # wait. Reissuing a Python wait before the next producer record is not
        # supported by value-backed ExternalEvent's host-side bookkeeping.
        return GraphInputPreparation(None, self._engram_staging_executor.submit(context.run, finish))

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.cp_size > 1:
            raise NotImplementedError(
                "DeepSeek-V4.1 context parallelism is scheduled with the Phase-6 distributed work"
            )
        context = get_forward_context()
        backend = context.attention_backend
        metadata = context.metadata
        draft_to_target_overlap = bool(getattr(backend, "supports_draft_to_target_overlap", False))
        if draft_to_target_overlap and self.engram_offload_active and capturing_acl_graph():
            self._engram_hash_graphs[id(metadata)].ready_event.wait(torch.npu.current_stream(input_ids.device))
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            hidden = self.embed_tokens(input_ids)
        positions = positions.to(torch.int64).contiguous()
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", False))
        if graph_mode and getattr(metadata, "dsa_metadata", None) is not None:
            # ACL graph replay reuses the metadata captured during warmup.
            pass
        else:
            self._build_dsa_metadata_for_forward(backend, metadata, positions)

        if draft_to_target_overlap and graph_mode:
            backend.prepare_graph_forward(metadata)

        # Engram: compute the n-gram hashes once, on the full token stream,
        # before the layer loop; each engram layer then pre-gathers its rows.
        engram_hashes: torch.Tensor | None = None
        engram_mask: torch.Tensor | None = None
        if self.engram_hash is not None:
            if self.engram_offload_active and capturing_acl_graph():
                # Offload contract: ACL-graph replay does not re-run Python, so
                # the host gather ran in the runner's pre-graph staging pass
                # (``stage_engram_rows_outside_graph``). Only the gate mask and
                # the staged rows enter the graph; the layer reads the rows
                # through ``engram.embed(None)``. The runner's first-capture gate
                # refuses the capture unless that pass staged, so the mask here
                # is always real -- running the device path would do illegal host
                # work inside the captured region.
                engram_mask = self._engram_capture_mask(input_ids)
            else:
                engram_hashes, engram_mask = self._compute_engram_hashes(backend, metadata, input_ids, positions)
                if engram_hashes is not None:
                    # Any pass under the graph runner (warmup before capture
                    # included, i.e. ``in_acl_graph()``) must stage into the
                    # address-stable graph block, because the captured graph
                    # reads whatever ``_staged_local`` this pass leaves behind.
                    # A true eager prefill (no execution state) uses the
                    # growable eager block instead.
                    # NOTE(architecture debt): design doc
                    # docs/src/content/docs/en/design/python_execution_context_design.md rule 7
                    # forbids model code calling in_acl_graph()/get_execution_buffer() to choose an
                    # address source; the compliant end-state is an EngramExecutionMetadataBuilder
                    # owning the staging block (allocate_persistent/update_persistent).
                    # follow-up: migrate to an EngramExecutionMetadataBuilder.
                    # Until that migration (the doc takes an incremental approach and keeps
                    # DeepSeek-V4.x on the legacy path), this probe is the minimal mechanism that
                    # keeps eager/graph staging address-correct: get_execution_buffer is NOT a
                    # drop-in replacement (per-entry buffers => memory blow-up, no shape-drift
                    # guard => silent corruption instead of the loud capacity error, and eager
                    # per-call allocation instead of reuse).
                    graph_visible = in_acl_graph()
                    for layer in self.layers:
                        engram = getattr(layer, "engram", None)
                        if engram is not None:
                            engram.prepare_embeddings(
                                engram_hashes[:, engram.layer_hash_index].to(torch.int64),
                                graph_visible=graph_visible,
                            )

        # Expand to hc_mult identical copies; the first pre_mix is one-hot on
        # copy 0 (model.py:1258-1260).
        hidden = hidden.unsqueeze(1).expand(-1, self.cfg.hc_mult, -1).contiguous()
        pre_mix = DeepseekV41HyperConnection.make_identity_pre_mix(hidden, self.cfg.hc_mult)
        # DSpark: buffer the per-layer residual stream the draft's main_proj
        # fuses, collapsing hc_mult with ``hidden.mean(dim=1)``. A V4.1 target
        # layer id names the stream ENTERING that layer, where a V4 one names
        # the layer output, so the capture runs before the layer call. None
        # while capture is disabled, so the mean and the buffer allocation are
        # skipped entirely off the speculative path.
        aux_capture = self.aux_hidden_capture
        aux_hidden_buffer = aux_capture.create_buffer(hidden.mean(dim=1)) if aux_capture.enabled else None
        for layer_id, layer in enumerate(self.layers):
            if (
                draft_to_target_overlap
                and self.engram_offload_active
                and layer.engram is not None
                and layer_id == self.cfg.engram_layer_ids[0]
                and capturing_acl_graph()
            ):
                self._engram_rows_ready[id(metadata)].wait(torch.npu.current_stream(input_ids.device))
            if layer_id == SWA_BOUNDED_FIRST_CONSUMER and should_run_swa_bounded_replay(
                self.cfg.enable_swa_bounded_replay, context
            ):
                return run_swa_bounded_consumers(
                    self, context, hidden, pre_mix, positions, input_ids, aux_hidden_buffer
                )
            if aux_hidden_buffer is not None and aux_capture.should_capture(layer_id):
                aux_capture.capture_layer(layer_id, hidden.mean(dim=1), None, aux_hidden_buffer)
            layer_cos_sin_cache = self._layer_rope_cache(self.csa2_plan.mode(layer_id))
            hidden, pre_mix = layer(
                hidden,
                positions,
                layer_cos_sin_cache,
                input_ids,
                pre_mix,
                engram_hashes,
                engram_mask,
            )
            record_layer_event(layer_id)
        # Final merge: the learned weighted sum with the LAST block's ffn_pre.
        merged = self.layers[-1].hc.hc_pre(hidden, pre_mix)
        return aux_capture.finalize(self.norm(merged), aux_hidden_buffer)


class _PlainRotaryEmbedding(nn.Module):
    """Plain (no YaRN) RoPE table for the ratio-0 layers."""

    def __init__(
        self,
        rotary_dim: int,
        max_positions: int,
        theta: float,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "cos_sin_cache",
            _build_plain_cos_sin_cache(rotary_dim, max_positions, theta, device),
            persistent=False,
        )


# ---------------------------------------------------------------------------
# Causal LM: mode-aware construction + weight loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Vision tower (DeepSeek-V4.1-Flash; multimodal Phase 4)
# ---------------------------------------------------------------------------
# Ported from the vllm DSV4 vision reference
# (vllm/models/deepseek_v4/common/vision.py): patch-embed → bidirectional ViT
# with 2D RoPE (theta 10000) → spatial-merge aligner projecting vision features
# back to the language width. The tower + aligner are replicated across TP ranks
# (no sharding) and are bf16 (not quantized) in both the fp8 and ascend exports.
# Sentinel injection is handled by ``prepare_multimodal_inputs`` via
# ``merge_multimodal_embeddings`` once ``encode_multimodal`` has produced rows.


@dataclass
class DeepseekV41VisionConfig:
    n_layers: int = 0
    hidden_size: int = 1024
    num_heads: int = 16
    intermediate_size: int = 2816
    patch_size: int = 14
    num_channels: int = 3
    out_hidden_size: int = 5120
    rope_theta: float = 10000.0
    downsample_ratio: int = 3
    head_dim: int = 64
    norm_eps: float = 1e-6

    @property
    def enabled(self) -> bool:
        return self.n_layers > 0 and self.hidden_size > 0 and self.num_heads > 0

    @classmethod
    def from_dict(cls, config: dict) -> DeepseekV41VisionConfig:
        vc = config.get("vision_config") or {}

        def pick(mm_key: str, vc_key: str, default: Any) -> Any:
            # Service path: flat ``mm_*`` keys from the C++ ModelArgs reflection.
            # Standalone tests may pass the nested HF ``vision_config.*`` layout.
            for source, key in ((config, mm_key), (vc, vc_key)):
                value = source.get(key)
                if value is not None:
                    return value
            return default

        hidden = int(pick("mm_hidden_size", "hidden_size", 1024))
        heads = int(pick("mm_num_attention_heads", "num_attention_heads", 16))
        head_dim = int(pick("mm_head_dim", "head_dim", 0)) or (hidden // heads if heads else 0)
        return cls(
            n_layers=int(pick("mm_num_hidden_layers", "num_hidden_layers", 0)),
            hidden_size=hidden,
            num_heads=heads,
            intermediate_size=int(pick("mm_intermediate_size", "intermediate_size", 2816)),
            patch_size=int(pick("mm_patch_size", "patch_size", 14)),
            num_channels=int(pick("mm_num_channels", "num_channels", 3)),
            out_hidden_size=int(pick("mm_projection_dim", "out_hidden_size", 5120)),
            rope_theta=float(pick("mm_rope_theta", "rope_theta", 10000.0)),
            downsample_ratio=int(pick("mm_downsample_ratio", "downsample_ratio", 3)),
            head_dim=head_dim,
        )


class DeepseekV41VisionNorm(nn.Module):
    """RMSNorm with fp32 weight and the vision eps (1e-6, not the text 1e-20)."""

    def __init__(self, dim: int, eps: float = 1e-6, device=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


def _vision_cos_sin(
    n_h: int, n_w: int, dim: int, theta: float, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    hpos = torch.arange(n_h, device=device).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w, device=device).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float()
    freqs = (freqs * inv_freq).flatten(1)
    # Keep the rotary basis in float32: ``_apply_vision_rotary`` upcasts ``x`` to
    # float32 before multiplying, so rounding cos/sin to the model dtype first
    # would only lose precision for nothing.
    return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)


def _apply_vision_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x1, x2 = x.float().chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1).to(dtype)


class DeepseekV41VisionAttention(nn.Module):
    def __init__(self, cfg: DeepseekV41VisionConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.head_dim
        self.wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.num_heads * cfg.head_dim, bias=True, dtype=dtype, device=device)
        self.wo = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.hidden_size, bias=True, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        q, k, v = (t.view(n, self.num_heads, self.head_dim) for t in self.wqkv(x).chunk(3, -1))
        q = _apply_vision_rotary(q, cos, sin)
        k = _apply_vision_rotary(k, cos, sin)
        # One image per call: dense bidirectional attention, no varlen metadata.
        q, k, v = (t.transpose(0, 1) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v)
        return self.wo(o.transpose(0, 1).reshape(n, -1))


class DeepseekV41VisionMLP(nn.Module):
    def __init__(self, cfg: DeepseekV41VisionConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        # w1 is the fused gate|up projection (checkpoint shape [2*inter, hidden]).
        self.w1 = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=False, dtype=dtype, device=device)
        self.w2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.w1(x)
        gate, up = gate_up.chunk(2, -1)
        return self.w2(F.silu(gate) * up)


class DeepseekV41VisionBlock(nn.Module):
    def __init__(self, cfg: DeepseekV41VisionConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.norm1 = DeepseekV41VisionNorm(cfg.hidden_size, cfg.norm_eps, device)
        self.attn = DeepseekV41VisionAttention(cfg, dtype, device)
        self.norm2 = DeepseekV41VisionNorm(cfg.hidden_size, cfg.norm_eps, device)
        self.mlp = DeepseekV41VisionMLP(cfg, dtype, device)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        return x + self.mlp(self.norm2(x))


class DeepseekV41VisionTower(nn.Module):
    """ViT: patch-embed + N bidirectional blocks + final norm (replicated)."""

    def __init__(self, cfg: DeepseekV41VisionConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.cfg = cfg
        self.rope_dim = cfg.head_dim // 2
        self.patch_embed = nn.Linear(
            cfg.num_channels * cfg.patch_size**2, cfg.hidden_size, bias=True, dtype=dtype, device=device
        )
        self.blocks = nn.ModuleList(DeepseekV41VisionBlock(cfg, dtype, device) for _ in range(cfg.n_layers))
        self.norm = DeepseekV41VisionNorm(cfg.hidden_size, cfg.norm_eps, device)

    def forward(self, patches: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        x = self.patch_embed(patches.flatten(1))
        cos, sin = _vision_cos_sin(n_h, n_w, self.rope_dim, self.cfg.rope_theta, x.device)
        for block in self.blocks:
            x = block(x, cos, sin)
        return self.norm(x)


class DeepseekV41Aligner(nn.Module):
    """Spatial merge (downsample_ratio × downsample_ratio) + 2-layer MLP."""

    def __init__(self, cfg: DeepseekV41VisionConfig, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.r = cfg.downsample_ratio
        self.in_dim = cfg.hidden_size * self.r**2
        self.out_dim = cfg.out_hidden_size
        self.w1 = nn.Linear(self.in_dim, self.out_dim, bias=True, dtype=dtype, device=device)
        self.w2 = nn.Linear(self.out_dim, self.out_dim, bias=True, dtype=dtype, device=device)

    def forward(self, x: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        r = self.r
        x = x.view(n_h, n_w, -1).permute(2, 0, 1)  # (C, H, W)
        x = F.pad(x, (0, -n_w % r, 0, -n_h % r))
        x = F.unfold(x.unsqueeze(0), r, stride=r).squeeze(0).transpose(0, 1)  # (L, C*r*r)
        return self.w2(F.gelu(self.w1(x)))


# Image-block layout (vllm deepseek_v4/common/mm_preprocess.py:54-237, ported).
# Roles are laid out in the "N order" the attention compressor expects; the
# xllm convention folds every non-pad role onto the single sentinel id 129264
# and leaves 129265 for the compressor-alignment pads (deepseek_v41_engram.py).
#
# LOCKED PAIR: this layout is implemented twice — here (Python, consumed by
# `encode`) and in C++ `DeepseekV41PromptProcessor::image_block_roles`
# (xllm/processors/deepseek_v41_prompt_processor.cpp, consumed by `process` /
# `adjust_token_layout` / `find_mm_spans`). The two MUST stay byte-for-byte
# consistent: a mismatch shifts token positions vs embed rows silently.
# tests/python/test_deepseek_v41_vision_layout.py and
# tests/processors/deepseek_v41_prompt_processor_test.cpp pin
# both sides to the same golden role sequences — update them together.
_IMAGE_START, _IMAGE_PAD, _IMAGE, _IMAGE_NEWLINE, _IMAGE_END = range(5)


def _vision_block_slots(n_llm_h: int, n_llm_w: int, compress_pad: int = 0) -> list[tuple[int, int]]:
    """Return ``[(role, feature_row_index), ...]`` for one image's LLM block.

    ``feature_row_index`` indexes the aligner rows (``n_llm_h * n_llm_w``);
    pad roles carry ``-1``. The image feature slots are interleaved row-pair
    (``order``) exactly as the reference so the compressor sees them correctly;
    ``compress_pad`` leading pad slots are position-dependent and added by the
    prompt processor at splice time.
    """
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    pre = torch.tensor(
        ([_IMAGE] * n_llm_w + [_IMAGE_NEWLINE]) * n_llm_h + [_IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), _IMAGE_PAD, dtype=torch.int64),
            torch.tensor([_IMAGE_START], dtype=torch.int64),
            pre[order],
            torch.full((pad_last,), _IMAGE_PAD, dtype=torch.int64),
            torch.tensor([_IMAGE_END], dtype=torch.int64),
        ]
    )
    slots = []
    feat = 0
    for role in types.tolist():
        if role == _IMAGE:
            slots.append((_IMAGE, int(perm[feat])))
            feat += 1
        else:
            slots.append((role, -1))
    return slots


class DeepseekV41ForCausalLM(PyModelBase):
    """DeepSeek-V4.1 causal LM driven by the C++ PyCausalLM bridge."""

    # Selected by the C++ PyMultimodalExecutor: image encoder outputs are
    # stateless per request, so the shared encoder cache may retain them.
    cache_multimodal_embeddings: bool = True

    # ``_load_dsv4_moe`` is borrowed from ``DeepseekV4ForCausalLM`` by calling
    # it unbound with this instance (V4.1 does not inherit the V4 LM), and it
    # now dispatches its routing half through ``self._load_dsv4_moe_routing``.
    # Re-expose that static helper so the borrowed loader resolves it.
    _load_dsv4_moe_routing = staticmethod(DeepseekV4ForCausalLM._load_dsv4_moe_routing)

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.cfg = DeepseekV41Config.from_dict(config)
        self.quant_mode = _detect_v41_quant_mode(config)
        dtype = self.resolve_dtype(config.get("dtype") or config.get("torch_dtype"))
        device = torch.device(config.get("device", "npu:0"))
        self.model = DeepseekV41Model(self.cfg, dtype, device, self.quant_mode, config)
        tp = self.cfg.tp_size
        self.lm_head = ColumnParallelLinear(
            self.cfg.hidden_size,
            self.cfg.vocab_size // tp,
            tp,
            gather_output=True,
            dtype=dtype,
            device=device,
        )

        # Vision tower (multimodal Phase 4). Built only when the checkpoint
        # carries a vision_config; text-only runs keep these as None and load
        # no vision tensors.
        self.vision_cfg = DeepseekV41VisionConfig.from_dict(config)
        self.vision_model: Optional[DeepseekV41VisionTower] = None
        self.aligner: Optional[DeepseekV41Aligner] = None
        if self.vision_cfg.enabled:
            self.vision_model = DeepseekV41VisionTower(self.vision_cfg, dtype, device)
            self.aligner = DeepseekV41Aligner(self.vision_cfg, dtype, device)
            out_dim = self.vision_cfg.out_hidden_size
            for name in ("image_start", "image_end", "image_newline"):
                self.register_parameter(
                    name,
                    nn.Parameter(torch.zeros(out_dim, dtype=dtype, device=device), requires_grad=False),
                )
            logger.info(
                "DeepSeek-V4.1 vision tower enabled: layers=%d hidden=%d heads=%d patch=%d downsample=%d out=%d",
                self.vision_cfg.n_layers,
                self.vision_cfg.hidden_size,
                self.vision_cfg.num_heads,
                self.vision_cfg.patch_size,
                self.vision_cfg.downsample_ratio,
                out_dim,
            )

    # -- multimodal encode ------------------------------------------------------

    def encode(self, pixel_values: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Encode images into the concatenated image-block embeddings.

        Emits one row per **sentinel (129264)** position of every image block,
        in prompt order: ``image_start`` → the aligner feature rows (row-pair
        interleaved, N order) each followed by ``image_newline`` → ``image_end``.
        Compressor-alignment pads (129265) are not emitted here;
        ``prepare_multimodal_inputs`` scatters these rows at the matching
        id-129264 positions (``merge_multimodal_embeddings``) and the C++ bridge
        slices them per chunk.

        ``pixel_values`` is the flattened patch batch (``(N, C, p, p)`` or
        ``(N, C*p*p)``); ``grid`` is ``(num_images, 2)`` ``[n_vit_h, n_vit_w]``,
        or a single flat row that is promoted to ``(1, -1)``. A row may carry a
        leading THW column; only its last two entries (``n_vit_h, n_vit_w``) are
        used.
        """
        if self.vision_model is None or self.aligner is None:
            return torch.empty(0)
        ref = next(self.vision_model.parameters())
        patches = pixel_values.to(device=ref.device, dtype=ref.dtype)
        if patches.dim() == 2:
            patches = patches.reshape(
                -1, self.vision_cfg.num_channels, self.vision_cfg.patch_size, self.vision_cfg.patch_size
            )
        if not torch.is_tensor(grid):
            grid = torch.tensor(grid, device=ref.device)
        grid = grid.to(device=ref.device, dtype=torch.int64)
        if grid.dim() == 1:
            # A single flat [n_vit_h, n_vit_w] (or THW) row: promote it to
            # (1, -1) so the per-image loop below can index it.
            grid = grid.reshape(1, -1)
        if grid.dim() != 2:
            raise ValueError(
                f"DeepSeek-V4.1 image_grid_thw must be 1D or 2D, got {grid.dim()}D with shape {tuple(grid.shape)}",
            )
        # Each row is read from its trailing [n_vit_h, n_vit_w]; a leading THW
        # column is tolerated and ignored. Reject short rows here instead of
        # letting `row[-2]` silently read across rows below.
        if grid.size(1) < 2:
            raise ValueError(
                "DeepSeek-V4.1 image_grid_thw rows need at least 2 columns "
                f"[n_vit_h, n_vit_w], got shape {tuple(grid.shape)}"
            )
        r = self.vision_cfg.downsample_ratio
        embeds = []
        offset = 0
        for row in grid.tolist():
            n_vit_h, n_vit_w = int(row[-2]), int(row[-1])
            n_patches = n_vit_h * n_vit_w
            image = patches[offset : offset + n_patches]
            offset += n_patches
            vit = self.vision_model(image, n_vit_h, n_vit_w)
            feats = self.aligner(vit, n_vit_h, n_vit_w)
            n_llm_h = -(-n_vit_h // r)
            n_llm_w = -(-n_vit_w // r)
            rows = []
            for role, feat_idx in _vision_block_slots(n_llm_h, n_llm_w):
                if role == _IMAGE_PAD:
                    continue
                if role == _IMAGE_START:
                    rows.append(self.image_start)
                elif role == _IMAGE_END:
                    rows.append(self.image_end)
                elif role == _IMAGE_NEWLINE:
                    rows.append(self.image_newline)
                else:
                    rows.append(feats[feat_idx])
            embeds.append(torch.stack(rows))
        return torch.cat(embeds, dim=0) if embeds else torch.empty(0)

    # -- SupportsMultimodal protocol (xllm.python.models.multimodal) -------------

    def encode_multimodal(self, data: MultimodalData, token_counts: EncoderTokenCounts) -> EncoderOutputs:
        """Encode every uncached image item into full image-block rows."""
        counts = token_counts.get("image", [])
        if not counts:
            return {}
        if self.vision_model is None or self.aligner is None:
            raise ValueError("DeepSeek-V4.1 vision checkpoint is not loaded")
        pixels = data.get("pixel_values")
        grids = data.get("image_grid_thw")
        if not isinstance(pixels, torch.Tensor) or not isinstance(grids, torch.Tensor):
            raise TypeError("DeepSeek-V4.1 pixel_values and image_grid_thw must be tensors")
        if any(count < 0 for count in counts):
            raise ValueError("DeepSeek-V4.1 image token counts must not be negative")
        # ``counts`` is one entry per image (consumed as ``features.split``
        # sections), so it must line up with the ``image_grid_thw`` rows; a
        # mismatch would mis-group the rows even if the totals happened to agree.
        # Resolve the row count up front and reject a malformed grid here, so the
        # check never silently skips for a 0-dim or 3D+ ``image_grid_thw``.
        if grids.dim() not in (1, 2):
            raise ValueError(
                f"DeepSeek-V4.1 image_grid_thw must be 1D or 2D, got {grids.dim()}D with shape {tuple(grids.shape)}"
            )
        num_images = 1 if grids.dim() == 1 else grids.shape[0]
        if len(counts) != num_images:
            raise ValueError(
                f"DeepSeek-V4.1 image token counts {len(counts)} do not match "
                f"the {num_images} image(s) in image_grid_thw"
            )
        features = self.encode(pixels, grids)
        if features.ndim != 2 or features.shape[0] != sum(counts):
            raise ValueError(
                f"DeepSeek-V4.1 encoder rows {features.shape[0]} do not match the "
                f"scheduled image token counts {sum(counts)}"
            )
        return {"image|embedding": list(features.split(counts, dim=0))}

    def prepare_multimodal_inputs(self, input_ids: torch.Tensor, data: MultimodalData) -> MultimodalInputs:
        """Scatter the scheduled image rows into the text embeddings."""
        inputs_embeds = self.model.embed_tokens(input_ids)
        if "image|embedding" in data:
            merge_multimodal_embeddings(inputs_embeds, data)
        return MultimodalInputs(inputs_embeds=inputs_embeds)

    def _load_vision_weights(self, loader: W8A8WeightLoader) -> None:
        """Load vision.* + aligner.* + image_* (bf16, replicated) from the checkpoint."""
        if self.vision_model is None or self.aligner is None:
            return
        copy = loader.copy_in
        load = loader.load_tensor
        # The checkpoint stores the patch embedding as a Conv2d weight
        # [hidden, C, patch, patch]; the tower keeps it as an nn.Linear, so
        # flatten the patch block to [hidden, C*patch*patch] (copy_ matches
        # shapes only, it does not flatten).
        patch_area = self.vision_cfg.num_channels * self.vision_cfg.patch_size**2
        copy(
            "vision_model.patch_embed.weight",
            load("vision.patch_embed.proj.weight").reshape(self.vision_cfg.hidden_size, patch_area),
        )
        copy("vision_model.patch_embed.bias", load("vision.patch_embed.proj.bias"))
        for i in range(self.vision_cfg.n_layers):
            ck, pm = f"vision.blocks.{i}", f"vision_model.blocks.{i}"
            for suffix in (
                "attn.wqkv.weight",
                "attn.wqkv.bias",
                "attn.wo.weight",
                "attn.wo.bias",
                "mlp.w1.weight",
                "mlp.w2.weight",
                "norm1.weight",
                "norm2.weight",
            ):
                copy(f"{pm}.{suffix}", load(f"{ck}.{suffix}"))
        copy("vision_model.norm.weight", load("vision.norm.weight"))
        for name in ("w1", "w2"):
            copy(f"aligner.{name}.weight", load(f"aligner.{name}.weight"))
            copy(f"aligner.{name}.bias", load(f"aligner.{name}.bias"))
        for name in ("image_start", "image_end", "image_newline"):
            copy(name, load(name))

    # -- weight loading ---------------------------------------------------------

    def load_weights(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        if self.quant_mode == V41_QUANT_ASCEND:
            self._load_weights_ascend(state_dicts, tp_rank, tp_size)
        else:
            self._load_weights_torch(state_dicts, tp_rank, tp_size)
        # Vision tower + aligner are bf16 and identical across quant modes.
        if self.vision_model is not None:
            self._load_vision_weights(W8A8WeightLoader(self, state_dicts, tp_size, tp_rank))
        # All-or-nothing host offload: a per-layer host-table fallback (see
        # ``_load_engram_ascend``) demotes the whole model to the HBM path so the
        # ACL-graph capture never mixes offloaded and device engram layers.
        self.model._reconcile_engram_offload_active()

    def release_graph_inputs(self, metadata: Any) -> None:
        self.model.release_graph_inputs(metadata)

    def close_host_offload(self) -> None:
        """Release DP-shared host tables before this model is dropped.

        Called from the C++ bridge destructor (``PyCausalLM::~PyCausalLM``)
        under the GIL; it runs no collective, so it is safe during teardown.
        """
        self.model.close_host_offload()

    def _load_weights_torch(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        """fp8-block / bf16 loading: dequantized bf16 plain-linear modules."""
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)
        is_fp8 = self.quant_mode == V41_QUANT_FP8

        def _has(name: str) -> bool:
            return loader.has(name)

        def _linear(ckpt_prefix: str, param_prefix: str, shard_dim: int | None = None) -> None:
            if is_fp8:
                weight = loader.load_tensor(ckpt_prefix + ".weight")
                scale = loader.load_tensor(ckpt_prefix + ".scale")
                weight = dequant_fp8_block(weight, scale)
            else:
                weight = loader.load_tensor(ckpt_prefix + ".weight")
            if shard_dim is not None:
                weight = loader.shard(weight, dim=shard_dim)
            loader.copy_in(param_prefix + ".weight", weight)

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            # The five attention projections are constructed unconditionally by
            # DeepseekV41Attention, so a missing checkpoint tensor is a broken
            # export (not an optional module): fail instead of running with
            # uninitialized weights.
            for ckpt_name, param_name, shard_dim in (
                ("attn.wq_a", "self_attn.q_a_proj", None),
                ("attn.wq_b", "self_attn.q_b_proj", None if attn.enable_dsa_cp else 0),
                ("attn.wkv", "self_attn.kv_proj", None),
                ("attn.wo_a", "self_attn.o_a_proj", 0),
                ("attn.wo_b", "self_attn.o_b_proj", 1),
            ):
                weight_key = ck + ckpt_name + ".weight"
                if not _has(weight_key):
                    raise KeyError(
                        f"required DeepSeek-V4.1 attention weight '{weight_key}' "
                        f"is missing from the checkpoint (decoder layer {i})"
                    )
                _linear(ck + ckpt_name, pm + param_name, shard_dim)
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            sink_key = ck + "attn.attn_sink"
            if not _has(sink_key):
                sink_key = ck + "attn.attn_sink.weight"
            if _has(sink_key):
                sink = loader.load_tensor(sink_key)
                if cfg.tp_size > 1 and not attn.enable_dsa_cp:
                    # Per-head sink sliced to this rank's q head range.
                    sink = loader.shard(sink, dim=0)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
                attn.attn_sink_loaded = True
            self._load_common_layer_norms(loader, ck, pm)
            # Main-KV compressor (Full layers; no gate at ratio 1).
            if hasattr(attn, "cmp_wkv") and _has(ck + "attn.compressor.wkv.weight"):
                loader.copy_in(
                    pm + "self_attn.cmp_wkv.weight",
                    loader.load_tensor(ck + "attn.compressor.wkv.weight"),
                )
                if hasattr(attn, "cmp_wgate") and _has(ck + "attn.compressor.wgate.weight"):
                    loader.copy_in(
                        pm + "self_attn.cmp_wgate.weight",
                        loader.load_tensor(ck + "attn.compressor.wgate.weight"),
                    )
                loader.copy_in(
                    pm + "self_attn.cmp_norm.weight",
                    loader.load_tensor(ck + "attn.compressor.norm.weight"),
                )
            # Indexer (Full + Reindex layers; wk/k_norm only on kv sources).
            if attn.indexer is not None and _has(ck + "attn.indexer.wq_b.weight"):
                _linear(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
                loader.copy_in(
                    pm + "self_attn.indexer.weights_proj.weight",
                    loader.load_tensor(ck + "attn.indexer.weights_proj.weight"),
                )
                if attn.indexer.owns_k and _has(ck + "attn.indexer.wk.weight"):
                    loader.copy_in(
                        pm + "self_attn.indexer.wk.weight",
                        loader.load_tensor(ck + "attn.indexer.wk.weight"),
                    )
                    loader.copy_in(
                        pm + "self_attn.indexer.k_norm.weight",
                        loader.load_tensor(ck + "attn.indexer.k_norm.weight"),
                    )
            mlp = self.model.layers[i].mlp
            # Dispatch on the constructed module type, not on checkpoint
            # presence: a MoE layer always requires its gate + experts +
            # shared expert, so a missing key must surface from
            # _load_torch_moe instead of silently leaving the MLP at
            # uninitialized values.
            if isinstance(mlp, DeepseekV41MoE):
                self._load_torch_moe(loader, ck, pm, i, is_fp8)
            elif isinstance(mlp, DeepseekV41TorchMLP):
                for name in ("w1", "w2", "w3"):
                    _linear(ck + f"ffn.{name}", pm + f"mlp.{name}")
            # Engram (only the configured engram layers carry the module).
            if self.model.layers[i].engram is not None:
                self._load_engram_torch(loader, ck, pm, i, is_fp8)
        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_torch_moe(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        """Load gate + bias + stacked experts + shared expert of one torch MoE layer."""
        mlp = self.get_submodule(pm + "mlp")
        loader.copy_in(pm + "mlp.gate.weight", loader.load_tensor(ck + "ffn.gate.weight"))
        loader.copy_in(
            pm + "mlp.e_score_correction_bias",
            loader.load_tensor(ck + "ffn.gate.bias"),
        )
        if loader.has(ck + "ffn.gate.bias_vl"):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.load_tensor(ck + "ffn.gate.bias_vl"))
            mlp.gate_bias_vl_loaded = True
        for local_slot in range(mlp.num_experts_per_rank):
            expert = mlp.start_expert_id + local_slot
            for name, param in (
                ("w1", mlp.experts_w1),
                ("w2", mlp.experts_w2),
                ("w3", mlp.experts_w3),
            ):
                prefix = ck + f"ffn.experts.{expert}.{name}"
                if is_fp8:
                    weight = dequant_fp4_group(
                        loader.load_tensor(prefix + ".weight"),
                        loader.load_tensor(prefix + ".scale"),
                    )
                else:
                    weight = loader.load_tensor(prefix + ".weight")
                param.data[local_slot].copy_(weight.to(param.dtype))
        shared_prefix = ck + "ffn.shared_experts."
        for name, module_name in (
            ("w1", "shared_w1"),
            ("w2", "shared_w2"),
            ("w3", "shared_w3"),
        ):
            prefix = shared_prefix + name
            if is_fp8:
                weight = dequant_fp8_block(
                    loader.load_tensor(prefix + ".weight"),
                    loader.load_tensor(prefix + ".scale"),
                )
            else:
                weight = loader.load_tensor(prefix + ".weight")
            loader.copy_in(pm + f"mlp.{module_name}.weight", weight)

    def _load_engram_torch(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
        is_fp8: bool,
    ) -> None:
        """Load one engram layer (official FP8 / bf16 checkpoint layout).

        The hash table stays quantized in memory: FP8-E4M3 payloads plus
        ue8m0 per-32 scales are narrowed to this rank's head-bucket row range
        (a 3.8e8-row table must not be materialized in bf16), while the small
        ``wkv`` projection is dequantized at load like every other linear.
        """
        eng = self.model.layers[layer_id].engram
        assert eng is not None
        table = loader.load_tensor(ck + "engram.embed.weight")
        table_u8 = table.view(torch.uint8) if table.element_size() == 1 else table
        scale = loader.load_tensor(ck + "engram.embed.scale")
        scale_u8 = scale.view(torch.uint8) if scale.element_size() == 1 else scale
        start = eng.embed_tokens.vocab_start
        rows = eng.embed_tokens.part_num_embeddings
        if table_u8.size(0) < eng.embed_tokens.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table_u8.size(0)} rows, "
                f"config declares {eng.embed_tokens.num_embeddings}"
            )
        if rows > 0:
            # Direct H2D copy_ from the mmap view (no device temporary).
            eng.embed_tokens.weight.data.copy_(table_u8.narrow(0, start, rows))
            eng.embed_tokens.weight_scale.data.copy_(scale_u8.narrow(0, start, rows))
        if is_fp8:
            wkv = dequant_fp8_block(
                loader.load_tensor(ck + "engram.wkv.weight"),
                loader.load_tensor(ck + "engram.wkv.scale"),
            )
        else:
            wkv = loader.load_tensor(ck + "engram.wkv.weight")
        loader.copy_in(pm + "engram.wkv.weight", wkv)
        loader.copy_in(pm + "engram.q_weight", loader.load_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.load_tensor(ck + "engram.k_weight"))

    def _load_weights_ascend(self, state_dicts: list[StateDict], tp_rank: int, tp_size: int) -> None:
        """W8A8 export loading (existing V4 W8A8DynamicLinear path).

        The Eco-Tech W8A8 export ships the engram tables as symmetric int8 +
        fp32 group-32 scales and keeps engram wkv/q/k in bf16; the QuaRot
        trunk rotation is fully folded into the weights, so only the engram
        gate needs the runtime un-rotation (see Engram.forward).
        """
        cfg = self.cfg
        loader = W8A8WeightLoader(self, state_dicts, cfg.tp_size, cfg.tp_rank)

        def _has(name: str) -> bool:
            return loader.has(name)

        def _w8a8(
            ckpt_prefix: str,
            param_prefix: str,
            shard_dims: dict | None = None,
            required: bool = False,
        ) -> None:
            """Load one W8A8-dynamic projection (weight + scale + offset).

            Delegates to :func:`load_w8a8_dynamic_projection` so a missing
            ``weight_offset`` is explicitly zero-filled: the symmetric int8
            export omits it, and ``W8A8DynamicLinear.weight_offset`` otherwise
            stays at its ``torch.empty`` (garbage) initialization, failing the
            all-zero check in ``process_weights_after_loading``. A present
            non-zero offset is copied verbatim and still rejected there.
            ``required`` marks the projections
            :class:`DeepseekV41Attention` constructs unconditionally, so a
            missing ``weight`` / ``weight_scale`` raises.
            """
            load_w8a8_dynamic_projection(
                loader,
                ckpt_prefix,
                param_prefix,
                shard_dims,
                require_weight_and_scale=required,
                fill_missing_offset=True,
                description=f"DeepSeek-V4.1 W8A8 {ckpt_prefix}",
            )

        loader.copy_in(
            "model.embed_tokens.weight",
            loader.shard(loader.load_tensor("embed.weight"), dim=1),
        )
        for i in range(cfg.n_layers):
            ck = f"layers.{i}."
            pm = f"model.layers.{i}."
            attn = self.model.layers[i].self_attn
            _w8a8(ck + "attn.wq_a", pm + "self_attn.q_a_proj", required=True)
            _w8a8(
                ck + "attn.wq_b",
                pm + "self_attn.q_b_proj",
                None if attn.enable_dsa_cp else {"weight": 0, "weight_scale": 0, "weight_offset": 0},
                required=True,
            )
            _w8a8(ck + "attn.wkv", pm + "self_attn.kv_proj", required=True)
            loader.copy_in(
                pm + "self_attn.o_a_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_a.weight"), dim=0),
            )
            loader.copy_in(
                pm + "self_attn.o_b_proj.weight",
                loader.shard(loader.load_tensor(ck + "attn.wo_b.weight"), dim=1),
            )
            loader.copy_in(
                pm + "self_attn.q_a_layernorm.weight",
                loader.load_tensor(ck + "attn.q_norm.weight"),
            )
            loader.copy_in(
                pm + "self_attn.kv_a_layernorm.weight",
                loader.load_tensor(ck + "attn.kv_norm.weight"),
            )
            sink_key = ck + "attn.attn_sink"
            if not _has(sink_key):
                sink_key = ck + "attn.attn_sink.weight"
            if _has(sink_key):
                sink = loader.load_tensor(sink_key)
                if sink.dim() == 1 and sink.size(0) == cfg.n_heads and cfg.tp_size > 1 and not attn.enable_dsa_cp:
                    shard_size = cfg.n_heads // cfg.tp_size
                    sink = sink.narrow(0, cfg.tp_rank * shard_size, shard_size)
                loader.copy_in(pm + "self_attn.attn_sink", sink)
                attn.attn_sink_loaded = True
            self._load_common_layer_norms(loader, ck, pm)
            if hasattr(attn, "cmp_wkv") and _has(ck + "attn.compressor.wkv.weight"):
                loader.copy_in(
                    pm + "self_attn.cmp_wkv.weight",
                    loader.load_tensor(ck + "attn.compressor.wkv.weight"),
                )
                if hasattr(attn, "cmp_wgate") and _has(ck + "attn.compressor.wgate.weight"):
                    loader.copy_in(
                        pm + "self_attn.cmp_wgate.weight",
                        loader.load_tensor(ck + "attn.compressor.wgate.weight"),
                    )
                loader.copy_in(
                    pm + "self_attn.cmp_norm.weight",
                    loader.load_tensor(ck + "attn.compressor.norm.weight"),
                )
            if attn.indexer is not None and _has(ck + "attn.indexer.wq_b.weight"):
                _w8a8(ck + "attn.indexer.wq_b", pm + "self_attn.indexer.wq_b")
                loader.copy_in(
                    pm + "self_attn.indexer.weights_proj.weight",
                    loader.load_tensor(ck + "attn.indexer.weights_proj.weight"),
                )
                if attn.indexer.owns_k and _has(ck + "attn.indexer.wk.weight"):
                    loader.copy_in(
                        pm + "self_attn.indexer.wk.weight",
                        loader.load_tensor(ck + "attn.indexer.wk.weight"),
                    )
                    loader.copy_in(
                        pm + "self_attn.indexer.k_norm.weight",
                        loader.load_tensor(ck + "attn.indexer.k_norm.weight"),
                    )
            attn.process_weights_after_loading()
            mlp = self.model.layers[i].mlp
            # Dispatch on the constructed module type, not on checkpoint
            # presence: the experts are mandatory for a W8A8 MoE layer, so a
            # missing expert tensor must surface from _load_dsv4_moe instead of
            # silently skipping the whole MLP.
            if isinstance(mlp, DeepseekV41W8A8MoE):
                self._load_ascend_moe(loader, ck, pm, i)
                mlp.process_weights_after_loading()
            elif isinstance(mlp, DeepseekV3MLP):
                DeepseekV4ForCausalLM._load_dsv4_dense_mlp(loader, ck, pm, mlp)
            # Engram (Eco-Tech export: int8 table + bf16 wkv/q/k).
            if self.model.layers[i].engram is not None:
                self._load_engram_ascend(loader, ck, pm, i)
        loader.copy_in("model.norm.weight", loader.load_tensor("norm.weight"))
        self._load_lm_head(loader)

    def _load_engram_ascend(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
        layer_id: int,
    ) -> None:
        """Load one engram layer from the Eco-Tech W8A8 export.

        The hash table stays int8 in memory (payload [rows, 256] int8 + one
        fp32 scale per 32-column group) and is narrowed to this rank's
        head-bucket row range; ``wkv`` / ``q_weight`` / ``k_weight`` are bf16
        and copied verbatim (the K half and gate stay in the original basis,
        the V half is pre-folded with the QuaRot rotation).

        With host offload the table copy targets host CPU memory (pageable,
        not pinned); a failure there falls back to the HBM parameter (engram
        stays enabled) instead of silently skipping the layer. The fallback copy
        is a hard error: if the HBM table copy also fails, the layer's table is
        only partially filled and the model cannot serve correctly, so this
        aborts with an explicit error rather than running on a half-loaded table.

        With ``dp_shared_memory`` the table lives in one shared segment per
        node-local group, so only the group leader reads the checkpoint and
        writes its own TP row range; the peers read the shared pages. A leader
        failure is propagated to every group rank as a hard error (fail-fast)
        and never demotes to HBM.
        """
        eng = self.model.layers[layer_id].engram
        assert eng is not None
        embed = eng.embed_tokens
        if embed._shared_storage is not None:
            self._load_engram_ascend_shared(loader, ck, layer_id, embed)
        else:
            self._load_engram_ascend_private(loader, ck, layer_id, embed)
        loader.copy_in(pm + "engram.wkv.weight", loader.load_tensor(ck + "engram.wkv.weight"))
        loader.copy_in(pm + "engram.q_weight", loader.load_tensor(ck + "engram.q_weight"))
        loader.copy_in(pm + "engram.k_weight", loader.load_tensor(ck + "engram.k_weight"))

    @staticmethod
    def _validate_engram_table_dtypes(layer_id: int, table: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Common dtype checks; returns the table normalized to int8."""
        if table.dtype == torch.uint8:
            table = table.view(torch.int8)
        if table.dtype != torch.int8:
            raise NotImplementedError(
                f"engram table for layer {layer_id} has dtype {table.dtype}; the "
                "ascend loader implements the int8_sym export only (fp32 group "
                "scales, see quant_model_description.json optional.embedding_storage)"
            )
        if scale.dtype != torch.float32:
            raise NotImplementedError(
                f"engram table scale for layer {layer_id} has dtype {scale.dtype}; "
                "expected float32 group scales of the int8_sym export"
            )
        return table

    @staticmethod
    def _copy_engram_rows(
        embed: EngramEmbedding, table: torch.Tensor, scale: torch.Tensor, start: int, rows: int
    ) -> None:
        """Copy this rank's narrow row window from the mmap view into ``embed``.

        Shared by the private and DP-shared loaders (and re-run by the private
        host->HBM fallback); a zero-row rank copies nothing.
        """
        if rows <= 0:
            return
        embed.weight.data.copy_(table.narrow(0, start, rows))
        embed.weight_scale.data.copy_(scale.narrow(0, start, rows))

    def _load_engram_ascend_table(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        """Load + validate the int8 table and this rank's ``(start, rows)`` window.

        Both Ascend engram loaders read the same keys, enforce the same dtype and
        row-count contracts and copy the same narrow window; they differ only in
        where the copy lands and how a failure is handled (host->HBM fallback vs.
        group-aggregated abort). Extracted so the shared sequence -- and its
        row-count message -- cannot drift between the two.
        """
        table = loader.load_tensor(ck + "engram.embed.weight")
        scale = loader.load_tensor(ck + "engram.embed.scale")
        table = self._validate_engram_table_dtypes(layer_id, table, scale)
        start = embed.vocab_start
        rows = embed.part_num_embeddings
        if table.size(0) < embed.num_embeddings:
            raise ValueError(
                f"engram table for layer {layer_id} has {table.size(0)} rows, config declares {embed.num_embeddings}"
            )
        return table, scale, start, rows

    def _load_engram_ascend_private(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> None:
        """Per-rank host-table load with the narrow host->HBM fallback."""
        table, scale, start, rows = self._load_engram_ascend_table(loader, ck, layer_id, embed)
        # Direct copy_ from the mmap view; a prior .to(device) would materialize
        # a 12+GB device temporary on top of the parameter. Offload targets the
        # host table (CPU -> CPU); a failed host copy demotes the layer to the
        # HBM path, fail-fast instead of loading a half-populated host table.
        try:
            self._copy_engram_rows(embed, table, scale, start, rows)
        except (RuntimeError, MemoryError) as exc:
            # Narrow on purpose: a host copy fails as RuntimeError (a bad
            # source / shape / pinning) or MemoryError (host OOM), and only
            # those justify the HBM fallback. Anything else is an unexpected
            # defect and must surface instead of being masked as a fallback
            # (``except Exception`` would have swallowed it). ``BaseException``
            # control-flow signals (KeyboardInterrupt / SystemExit) are never
            # caught here either. Because RuntimeError is broader than a host
            # OOM -- a weight-path or shape defect raises it too -- the warning
            # below carries ``exc_info=True`` so the full traceback (which
            # names the failing copy) is always logged, not just the message.
            # Retained as-is on purpose (not narrowed to a specific OOM type):
            # the exception an OOM actually raises on the target stack has not
            # been measured, so narrowing could stop catching the real failure
            # while widening to ``Exception`` would mask unrelated defects. The
            # fallback itself no longer half-commits a failure (see
            # ``EngramEmbedding.fallback_to_device``), so what remains is the
            # documented RuntimeError-is-broader-than-OOM tradeoff above.
            if not embed.offload:
                raise
            logger.warning(
                "engram layer %d host-table load failed (%s); falling back to "
                "the HBM table path for this layer (the model then demotes all "
                "offload layers to HBM)",
                layer_id,
                exc,
                exc_info=True,
            )
            embed.fallback_to_device()
            try:
                self._copy_engram_rows(embed, table, scale, start, rows)
            except (RuntimeError, MemoryError) as exc:
                # The host->HBM fallback is not a success path on its own: if
                # the HBM copy fails too, the layer's table is only partially
                # populated and the model cannot serve correctly
                # (``fallback_to_device`` has already switched ``offload`` off,
                # so the state is half-loaded). Abort explicitly instead of
                # running on a half-loaded table. Narrow on purpose, same as the
                # first catch: only RuntimeError/MemoryError justify this path;
                # anything else surfaces unchanged.
                raise RuntimeError(
                    f"engram layer {layer_id}: HBM table copy failed even "
                    f"after the host->HBM fallback ({exc}); the engram table "
                    "is only partially populated and the model cannot serve "
                    "correctly, aborting instead of running with a "
                    "half-loaded table"
                ) from exc

    def _load_engram_ascend_shared(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        layer_id: int,
        embed: EngramEmbedding,
    ) -> None:
        """Leader-only load into a shared host table; peers read the pages.

        Only group rank 0 reads the checkpoint and writes this rank's TP row
        range (all group members map the same segment and the same TP shard, so
        one writer fills it). The result is propagated as: leader reads/writes
        -> ``engram_all_gather_object`` of the error -> ``engram_barrier`` fences
        every rank (success or failure) before any rank acts -> on failure mark
        FAILED and raise on every rank (fail-fast, never demote); on success mark
        LOADED. The barrier is entered unconditionally so a failing rank cannot
        start unwinding into a later collective while a peer is still completing
        this load (the same choreography ``SharedHostEngramStorage`` uses before
        the leader unlinks).

        The barrier is an ordering rendezvous for the ordinary failure path
        only: ``engram_all_gather_object`` already returns the same error list
        on every rank, and the barrier just guarantees no rank raises or marks
        the table while a peer is still inside this load. It is **not** crash
        tolerance: a failed rank may be noticed by the transport or only surface
        when the fence deadline fires. This one rendezvous runs with the shorter
        candidate deadline ``_ENGRAM_SHARED_LOAD_BARRIER_TIMEOUT`` (via
        ``dist.monitored_barrier``), which shortens that wait from the group's
        full 5-minute Gloo timeout to 30s -- but on the monitor (group rank 0)
        only: ``monitored_barrier`` enforces the per-call deadline on rank 0, and
        a non-zero survivor rank whose peer never reaches the fence still blocks
        until the group's own 5-minute timeout. The ``engram_all_gather_object``
        above is not governed by this deadline at all and is still bounded only
        by that group timeout. The group timeout also governs every other
        collective on the group, so the barrier must still not be read as a
        liveness guarantee.
        """
        storage = embed._shared_storage
        group = embed._dp_group
        assert storage is not None and group is not None
        error: str | None = None
        if group.rank() == 0:
            try:
                table, scale, start, rows = self._load_engram_ascend_table(loader, ck, layer_id, embed)
                self._copy_engram_rows(embed, table, scale, start, rows)
            except Exception as exc:  # noqa: BLE001 - propagated to every rank below
                # Ordinary exception only: a defect (or host OOM) is reported to
                # the whole group, never swallowed into an HBM fallback.
                error = f"{type(exc).__name__}: {exc}"
        errors: list[str | None] = [None] * group.size()
        engram_all_gather_object(errors, error)
        failures = "; ".join(f"rank {rank}: {failure}" for rank, failure in enumerate(errors) if failure is not None)
        # Fence every rank -- success or failure -- before any rank raises or
        # marks the table loaded. The gather already hands every rank the same
        # ``failures`` list, but entering the barrier unconditionally means no
        # rank can begin unwinding (or advance to a later collective) while a
        # peer is still completing this load; mirrors the unconditional fence in
        # ``SharedHostEngramStorage._attach_or_create``. Ordering only, not
        # crash tolerance: a peer that fails mid-collective is not fenced -- the
        # transport may notice it, and otherwise the shorter fence deadline
        # below is what unblocks the survivors (see the docstring).
        try:
            engram_barrier(timeout=_ENGRAM_SHARED_LOAD_BARRIER_TIMEOUT)
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            # The fence itself failed, most likely the shorter candidate timeout
            # firing on a peer that never reached it. Mark this rank FAILED
            # locally and re-raise with the layer/phase/rank context; never
            # retry, never keep serving, and never issue a status gather on a
            # communicator that is already known bad.
            storage.mark_failed()
            raise RuntimeError(
                f"Engram shared host-table load fence failed for layer {layer_id} "
                f"(phase=barrier, group_rank={group.rank()}): {type(exc).__name__}: {exc}"
            ) from exc
        if failures:
            storage.mark_failed()
            raise RuntimeError(f"Engram shared host-table load failed for layer {layer_id}: {failures}")
        storage.mark_loaded()

    def _load_ascend_moe(self, loader: W8A8WeightLoader, ck: str, pm: str, layer_id: int) -> None:
        """Ascend MoE: the V4 W8A8 staging plus the V4.1 ``gate.bias_vl``."""
        # The W8A8 export stores the MoE block under either the native
        # ``ffn.`` or the legacy ``mlp.`` prefix; resolve it once here and
        # share it with the shared V4 loader and the V4.1-only ``bias_vl``.
        source_prefix = _find_checkpoint_prefix(
            loader,
            (ck + "ffn.", ck + "mlp."),
            ("experts.0.w1.weight",),
        )
        if source_prefix is None:
            raise KeyError(f"DeepSeek-V4.1 layer {layer_id} MoE expert weights not found")
        DeepseekV4ForCausalLM._load_dsv4_moe(self, loader, ck, pm, layer_id, source_prefix)
        mlp = self.model.layers[layer_id].mlp
        bias_vl_key = source_prefix + "gate.bias_vl"
        if loader.has(bias_vl_key):
            loader.copy_in(pm + "mlp.gate_bias_vl", loader.load_tensor(bias_vl_key))
            # Deferred until the vision phase; loaded only for checkpoint
            # compatibility. The W8A8 expert path never reads gate_bias_vl.
            mlp.gate_bias_vl_loaded = True

    def _load_common_layer_norms(
        self,
        loader: W8A8WeightLoader,
        ck: str,
        pm: str,
    ) -> None:
        """Layer norms + mHC weights shared by every checkpoint mode."""
        loader.copy_in(
            pm + "input_layernorm.weight",
            loader.load_tensor(ck + "attn_norm.weight"),
        )
        loader.copy_in(
            pm + "post_attention_layernorm.weight",
            loader.load_tensor(ck + "ffn_norm.weight"),
        )
        for part in ("attn", "ffn"):
            for suffix in ("fn", "scale", "base"):
                name = f"hc_{part}_{suffix}"
                loader.copy_in(pm + "hc." + name, loader.load_tensor(ck + name))

    def _load_lm_head(self, loader: W8A8WeightLoader) -> None:
        """Load the output head under whichever alias the export uses."""
        lm_head_key = next(
            (
                name
                for name in ("lm_head.weight", "model.lm_head.weight", "model.head.weight", "head.weight")
                if loader.has(name)
            ),
            None,
        )
        assert lm_head_key is not None, "checkpoint output-head weight not found"
        loader.copy_in(
            "lm_head.weight",
            loader.shard(loader.load_tensor(lm_head_key), dim=0),
        )


# ---------------------------------------------------------------------------
# Pure-torch quantization helpers for the official checkpoint
#
# Reference recipes from docs/deepseek_v41_reference_semantics.md sections
# 2.1-2.5: ue8m0 scale decoding, 32x32-block FP8 E4M3 weight dequant, and MXFP4
# (E2M1 + per-32 ue8m0 group scale) expert dequant. Dequant-only: the dynamic
# FP8/FP4 activation QDQ variants were removed when KV alignment fixed KV to
# bf16 (no activation QDQ remains on the runtime path). Leaf section:
# depends only on torch and the stdlib (imported once at the module head),
# device-agnostic (CPU and NPU).
# ---------------------------------------------------------------------------

# IEEE 754 bits of the float32 subnormal 2**-127 (ue8m0 byte 0x00).
_UE8M0_ZERO_BITS = 0x00400000

_FP4_NIBBLE_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def ue8m0_to_scale(scale_u8: torch.Tensor) -> torch.Tensor:
    """Decodes ue8m0 scale bytes into float32 powers of two.

    Byte ``b`` decodes to ``2 ** (b - 127)``. Byte ``0x00`` is the subnormal
    ``2 ** -127`` (the format has no zero encoding) and byte ``0xFF`` (NaN)
    is invalid.

    Args:
        scale_u8: tensor of raw ue8m0 bytes; uint8, or any 1-byte dtype
            (int8 / float8_e8m0fnu / float4_e2m1fn_x2) reinterpreted as uint8.

    Returns:
        Float32 tensor of the same shape holding ``2 ** (b - 127)``.
    """
    if scale_u8.dtype == torch.uint8:
        raw = scale_u8
    elif scale_u8.element_size() == 1:
        raw = scale_u8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte ue8m0 scale storage, got {scale_u8.dtype}")
    raw_i32 = raw.to(torch.int32)
    if bool(torch.any(raw_i32 == 0xFF)):
        raise ValueError("ue8m0 scale byte 0xFF (NaN) is invalid")
    bits = torch.where(
        raw_i32 == 0,
        torch.full_like(raw_i32, _UE8M0_ZERO_BITS),
        raw_i32 << 23,
    )
    return bits.contiguous().view(torch.float32)


def dequant_fp8_block(
    weight_e4m3: torch.Tensor,
    scale_u8: torch.Tensor,
    block: tuple[int, int] = (32, 32),
) -> torch.Tensor:
    """Dequantizes block-wise FP8 E4M3 weights to bf16.

    ``W[n, k] = fp8_value(weight[n, k]) * scale[n // block_out, k // block_in]``
    with one ue8m0 power-of-two scale per output-by-input block (the official
    checkpoint uses 32x32 blocks and ``[ceil(out/32), ceil(in/32)]`` scales).

    Args:
        weight_e4m3: ``[out, in]`` float8_e4m3fn weight tensor.
        scale_u8: ``[ceil(out / block_out), ceil(in / block_in)]`` ue8m0 bytes.
        block: ``(block_out, block_in)`` block shape.

    Returns:
        Bfloat16 ``[out, in]`` dequantized weight.
    """
    if weight_e4m3.ndim != 2:
        raise ValueError(f"weight must be 2-D [out, in], got ndim={weight_e4m3.ndim}")
    if len(block) != 2 or block[0] <= 0 or block[1] <= 0:
        raise ValueError(f"block must be a pair of positive ints, got {block}")
    block_out, block_in = int(block[0]), int(block[1])
    out_dim, in_dim = weight_e4m3.shape
    expected_shape = (
        (out_dim + block_out - 1) // block_out,
        (in_dim + block_in - 1) // block_in,
    )
    if tuple(scale_u8.shape) != expected_shape:
        raise ValueError(
            f"scale shape {tuple(scale_u8.shape)} does not match expected "
            f"{expected_shape} for weight [{out_dim}, {in_dim}] with block {block}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(block_out, dim=0)
    scale = scale.repeat_interleave(block_in, dim=1)[:out_dim, :in_dim]
    return (_e4m3_to_float(weight_e4m3) * scale).to(torch.bfloat16)


def fp4_nibble_table(device: Optional[Union[torch.device, str]] = None) -> torch.Tensor:
    """Returns the 16-value E2M1 (MXFP4) decode table indexed by nibble.

    Nibble bit 3 is the sign; both ``0x0`` and ``0x8`` decode to (negative)
    zero. There is no Inf or NaN in the finite-only E2M1 format.
    """
    return torch.tensor(_FP4_NIBBLE_VALUES, dtype=torch.float32, device=device)


def unpack_fp4(
    weight_i8: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpacks packed MXFP4 nibbles into ``[N, K]`` values.

    The ``[..., K / 2]`` 1-byte storage (int8 viewed as uint8) holds two E2M1
    nibbles per byte along the last dim: the LOW nibble is element ``2j`` and
    the HIGH nibble is element ``2j + 1`` (matches reference ``convert.py``
    and the torch ``float4_e2m1fn_x2`` layout).

    Args:
        weight_i8: ``[..., K / 2]`` packed fp4 bytes.
        dtype: output dtype (default bf16).

    Returns:
        ``[..., K]`` tensor of decoded E2M1 values.
    """
    if weight_i8.dtype == torch.uint8:
        packed = weight_i8
    elif weight_i8.element_size() == 1:
        packed = weight_i8.view(torch.uint8)
    else:
        raise ValueError(f"expected 1-byte packed fp4 storage, got {weight_i8.dtype}")
    table = fp4_nibble_table(device=packed.device)
    low = table[(packed & 0x0F).to(torch.int64)]
    high = table[(packed >> 4).to(torch.int64)]
    unpacked = torch.stack((low, high), dim=-1).flatten(-2)
    return unpacked.to(dtype)


def dequant_fp4_group(
    weight_i8: torch.Tensor,
    scale_u8: torch.Tensor,
    group: int = 32,
) -> torch.Tensor:
    """Dequantizes MXFP4 expert weights to bf16.

    ``W[n, k] = e2m1(nibble) * 2 ** (S[n, k // group] - 127)`` with one ue8m0
    scale per row per ``group`` consecutive input elements (the official
    checkpoint uses ``[out, in / 32]`` scales).

    Args:
        weight_i8: ``[N, K / 2]`` packed fp4 bytes.
        scale_u8: ``[N, ceil(K / group)]`` ue8m0 scale bytes.
        group: number of input elements sharing one scale.

    Returns:
        Bfloat16 ``[N, K]`` dequantized weight.
    """
    if group <= 0:
        raise ValueError(f"group must be positive, got {group}")
    unpacked = unpack_fp4(weight_i8, dtype=torch.float32)
    in_dim = unpacked.size(-1)
    expected_groups = (in_dim + group - 1) // group
    if scale_u8.size(-1) != expected_groups:
        raise ValueError(
            f"scale has {scale_u8.size(-1)} groups, expected {expected_groups} for K={in_dim} with group={group}"
        )
    scale = ue8m0_to_scale(scale_u8)
    scale = scale.repeat_interleave(group, dim=-1)[..., :in_dim]
    return (unpacked * scale).to(torch.bfloat16)


# Subnormal grid step of the E4M3 format, shared by the arithmetic decode.
_E4M3_SUBNORMAL_STEP = 2.0**-9


def _e4m3_to_float(x_e4m3: torch.Tensor) -> torch.Tensor:
    """Decodes an E4M3 tensor to float32 without native float8 casts.

    Bit-level decode (sign | 4-bit biased exponent, bias 7 | 3-bit mantissa;
    exp 15 + mantissa 7 is NaN, subnormals scale by 2^-9) so weight dequant
    works identically on CPU and NPU.
    """
    bits = x_e4m3.view(torch.uint8).to(torch.int32)
    sign = torch.where(bits >= 128, -1.0, 1.0)
    magnitude_bits = bits & 0x7F
    exponent = (magnitude_bits >> 3) & 0xF
    mantissa = magnitude_bits & 0x7
    mantissa_f = mantissa.to(torch.float32)
    normal = (1.0 + mantissa_f / 8.0) * torch.exp2(exponent.to(torch.float32) - 7.0)
    subnormal = mantissa_f * _E4M3_SUBNORMAL_STEP
    value = torch.where(exponent > 0, normal, subnormal)
    value = torch.where(magnitude_bits == 0x7F, torch.full_like(value, float("nan")), value)
    return value * sign
