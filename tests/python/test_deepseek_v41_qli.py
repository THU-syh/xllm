# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wiring equivalence for the V4.1 quant_lightning_indexer_v2_va.

Extracted from the monolithic ``test_deepseek_v41_fused_ops.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import xllm.python.attention.csa2_attention as csa2
from tests.python._deepseek_v41_fused_testkit import (
    _SpyLogger,
)
from xllm.python.attention.csa2_attention import (
    Csa2AttentionBackend,
    Csa2LayerContext,
    CSA2SharedRuntime,
    _QliGraphState,
    apply_interleaved_rope,
)

# === [8] QLI_V3: quant_lightning_indexer_v2_va ===


# Tiny plan with one ratio-2 and one ratio-1 index layer, no candidate source
# (the production default of ``candidate_source_layer_id`` is -1).
_N_LAYERS = 3
_COMPRESS_RATIOS = [0, 2, 1]
_KV_SOURCE_LAYERS = [1]
_INDEX_SOURCE_LAYERS = [1, 2]
_WINDOW = 128
_HEAD_DIM = 512
_NUM_HEADS = 8
_INDEX_TOPK = 4
_ROPE_HEAD_DIM = 8
_INDEX_HEAD_DIM = 128
_INDEX_N_HEADS = 2
_BLOCK_SIZE = 128
_CAPACITY = 256
_ROWS = 4
_CTX_LENS = [300, 500, 200, 129]
# Physical block pool of the stub index caches the QLI shadow mirrors.
_SHADOW_BLOCKS = 8


class _QliRecorder:
    """Recording stand-ins for the two vendor ops."""

    def __init__(self) -> None:
        self.indexer_calls: list[dict[str, Any]] = []
        self.metadata_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def indexer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.indexer_calls.append(
            {
                "query": query,
                "key": key,
                "weights": weights,
                "query_scale": query_scale,
                "key_scale": key_scale,
                "topk": topk,
                "quant_mode": quant_mode,
                **kwargs,
            }
        )
        rows = query.size(0)
        picks = torch.full((rows, 1, topk), -1, dtype=torch.int32)
        # A deterministic two-pick row; every other row stays padding-only, so
        # the offset/-1 mapping is exercised in one shot.
        if rows:
            picks[0, 0, 0] = 0
            if topk > 1:
                picks[0, 0, 1] = 2
        # Match the op's meta: only ``candidate_mode == 1`` (source) returns a
        # block-index pool ``[rows, 1, candidate_topk_blocks]`` INT32; every
        # other mode returns an empty tensor.
        if int(kwargs.get("candidate_mode", csa2._QLI_V3_CAND_MODE_OFF)) == csa2._QLI_V3_CAND_MODE_SOURCE:
            blocks = int(kwargs.get("candidate_topk_blocks", csa2._QLI_V3_CAND_TOPK_BLOCKS_MAX))
            candidate_out = torch.full((rows, 1, blocks), -1, dtype=torch.int32)
            if rows:
                candidate_out[0, 0, 0] = 0
        else:
            candidate_out = torch.empty(0, dtype=torch.int32)
        return picks, torch.empty(0, dtype=torch.bfloat16), candidate_out

    def metadata(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        self.metadata_calls.append((args, kwargs))
        return torch.zeros(csa2._QLI_V3_METADATA_SIZE, dtype=torch.int32)


def _install_ops(monkeypatch: pytest.MonkeyPatch, recorder: _QliRecorder) -> None:
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: (recorder.indexer, recorder.metadata))


def _reset_log_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(csa2, "_qli_v3_active_logged", False)
    monkeypatch.setattr(csa2, "_qli_v3_fallback_logged", False)
    monkeypatch.setattr(csa2, "_qli_v3_candidate_logged", False)


def _stub_qli(monkeypatch: pytest.MonkeyPatch) -> _QliRecorder:
    """Reset the QLI latches and install a fresh recording-ops stub."""
    _reset_log_state(monkeypatch)
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    return recorder


@pytest.fixture(autouse=True)
def _c1_switches_reset_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset the process-lifetime QLI log latches per test."""
    _reset_log_state(monkeypatch)


def _make_backend(
    *,
    index_head_dim: int = _INDEX_HEAD_DIM,
    candidate_source_layer_id: int = -1,
    n_layers: int = _N_LAYERS,
    compress_ratios: list[int] | None = None,
    kv_source_layer_ids: list[int] | None = None,
    index_source_layer_ids: list[int] | None = None,
    candidate_topk_blocks: int = 2048,
    candidate_block_size: int = 8,
) -> Csa2AttentionBackend:
    return Csa2AttentionBackend(
        compress_ratios=list(_COMPRESS_RATIOS if compress_ratios is None else compress_ratios),
        window_size=_WINDOW,
        n_layers=n_layers,
        num_heads=_NUM_HEADS,
        attn_head_dim=_HEAD_DIM,
        index_topk=_INDEX_TOPK,
        index_n_heads=_INDEX_N_HEADS,
        index_head_dim=index_head_dim,
        rope_head_dim=_ROPE_HEAD_DIM,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        kv_source_layer_ids=list(_KV_SOURCE_LAYERS if kv_source_layer_ids is None else kv_source_layer_ids),
        index_source_layer_ids=list(_INDEX_SOURCE_LAYERS if index_source_layer_ids is None else index_source_layer_ids),
        candidate_source_layer_id=candidate_source_layer_id,
        candidate_topk_blocks=candidate_topk_blocks,
        candidate_block_size=candidate_block_size,
    )


class _StubIndexerLayer:
    """Minimal layer surface ``_run_indexer`` touches on the indexer path."""

    def __init__(self, layer_id: int, ratio: int) -> None:
        self.layer_id = layer_id
        self.mode = SimpleNamespace(compress_ratio=ratio)

    def indexer_query(self, qr: torch.Tensor) -> torch.Tensor:
        return torch.zeros(qr.size(0), _INDEX_N_HEADS, _INDEX_HEAD_DIM, dtype=torch.bfloat16)

    def indexer_weights(self, x: torch.Tensor) -> torch.Tensor:
        return torch.full((x.size(0), _INDEX_N_HEADS), 0.5, dtype=torch.bfloat16)


class _StubWriterLayer:
    """Minimal layer surface ``_run_compressor`` touches on the index-write path.
    Ratio 1 only: one ``compress_kv_score`` projection, no pooling gate and no compressor state, so the index write is a"""

    def __init__(self, layer_id: int, ratio: int = 1) -> None:
        self.layer_id = layer_id
        self.mode = SimpleNamespace(compress_ratio=ratio)
        self.indexer = SimpleNamespace(owns_k=True)

    def compress_kv_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = x.size(0)
        rows = torch.arange(1, n + 1, dtype=torch.float32).unsqueeze(-1)
        cols = torch.arange(1, _INDEX_HEAD_DIM + 1, dtype=torch.float32).unsqueeze(0)
        kv = (rows * cols).to(torch.bfloat16)
        return kv, kv

    def compress_rmsnorm(self, v: torch.Tensor) -> torch.Tensor:
        return v

    def indexer_key(self, latent: torch.Tensor) -> torch.Tensor:
        return latent[:, :_INDEX_HEAD_DIM]


class _StubLayerCache:
    """Index/key/swa cache slots of one writer layer (4-D paged layout).
    The index plane mirrors the NPU platform policy: INT8 codes plus an FP16 per-token scale plane."""

    def __init__(
        self,
        *,
        blocks: int = _SHADOW_BLOCKS,
        block_size: int = _BLOCK_SIZE,
        head_dim: int = _HEAD_DIM,
        index_head_dim: int = _INDEX_HEAD_DIM,
    ) -> None:
        self.key = torch.zeros(blocks, block_size, 1, head_dim, dtype=torch.bfloat16)
        self.index = torch.zeros(blocks, block_size, 1, index_head_dim, dtype=torch.int8)
        self.indexer_scale = torch.zeros(blocks, block_size, 1, dtype=torch.float16)
        self.swa = torch.zeros(blocks, block_size, 1, head_dim, dtype=torch.bfloat16)
        self.compress_kv_state = None
        self.compress_score_state = None


def _make_ctx(rows: int = _ROWS) -> Csa2LayerContext:
    half = _ROPE_HEAD_DIM // 2
    cache = torch.zeros(1024, _ROPE_HEAD_DIM, dtype=torch.float32)
    # cos == 1, sin == 0: the interleaved rotation is the identity, so the
    # quantized query equals the raw projection and the test stays deterministic.
    cache[:, :half] = 1.0
    return Csa2LayerContext(
        x=torch.zeros(rows, 128, dtype=torch.bfloat16),
        qr=torch.zeros(rows, _INDEX_N_HEADS * _INDEX_HEAD_DIM, dtype=torch.bfloat16),
        positions=torch.arange(rows, dtype=torch.int64),
        cos_sin_cache=cache,
    )


def _make_shared(
    ratio: int,
    *,
    index_dtype: torch.dtype = torch.int8,
    with_scale: bool = True,
    key_shape: tuple[int, ...] | None = None,
    scale_shape: tuple[int, ...] | None = None,
    with_block_table: bool = True,
) -> CSA2SharedRuntime:
    # The block table must stay inside the cache: 2 columns per sequence with
    # block ids 0..7 need 8 physical blocks.
    key_shape = (
        key_shape
        if key_shape is not None
        else (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS, _INDEX_HEAD_DIM)
    )
    index_scale = None
    if with_scale:
        index_scale = torch.zeros(
            scale_shape if scale_shape is not None else (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS),
            dtype=torch.float16,
        )
    return CSA2SharedRuntime(
        index_cache=torch.zeros(key_shape, dtype=index_dtype),
        index_block_table=(torch.arange(_ROWS * 2, dtype=torch.int32).view(_ROWS, 2) if with_block_table else None),
        index_scale=index_scale,
        compress_ratio=ratio,
    )


def _seed_index_plane(shared: CSA2SharedRuntime, seed: CSA2SharedRuntime) -> None:
    """Copy the index-plane slots a compressor would publish onto ``shared``.
    ``_shared_runtime`` builds an empty :class:`CSA2SharedRuntime`; this fills the index plane so the read side can run"""
    shared.index_cache = seed.index_cache
    shared.index_block_table = seed.index_block_table
    shared.index_scale = seed.index_scale
    shared.compress_ratio = seed.compress_ratio


def _make_dsa(
    *,
    is_acl_graph: bool = True,
    capacity: int = _CAPACITY,
    seq_lens: list[int] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        is_acl_graph=is_acl_graph,
        graph_token_capacity=capacity,
        seq_lens=torch.tensor(list(_CTX_LENS if seq_lens is None else seq_lens), dtype=torch.int32),
    )


def _drive(
    backend: Csa2AttentionBackend,
    layer: _StubIndexerLayer,
    dsa: SimpleNamespace,
    shared: CSA2SharedRuntime,
    ctx: Csa2LayerContext | None = None,
) -> torch.Tensor:
    return backend._run_indexer(layer, ctx or _make_ctx(), dsa, [1] * _ROWS, dsa.seq_lens, shared)


# ---------------------------------------------------------------------------
# 1) dispatch on/off
# ---------------------------------------------------------------------------


def test_qli_v3_constants() -> None:
    """Pin the operator's static constants."""
    assert csa2._QLI_V3_QUANT_MODE == 2
    assert csa2._QLI_V3_HEAD_DIM == 128
    assert csa2._QLI_V3_KEY_NUM_HEADS == 1
    # The candidate sub-switch and its op mode values.
    assert csa2._QLI_V3_CAND_MODE_SOURCE == 1
    assert csa2._QLI_V3_CAND_MODE_CONSUMER == 2
    assert csa2._QLI_V3_CAND_MODE_OFF == 3


def test_qli_v3_op_missing_stays_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)  # no verified registration
    backend = _make_backend()

    topk = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert recorder.indexer_calls == []
    assert recorder.metadata_calls == []
    assert backend._qli_state == {}
    assert topk.dtype == torch.int32
    assert topk.shape == (_ROWS, 1, _INDEX_TOPK)


def test_qli_v3_dispatch_on_routes_through_the_op(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    shared = _make_shared(1)

    topk = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    assert len(recorder.indexer_calls) == 1
    call = recorder.indexer_calls[0]
    # TND query: [T, index heads, 128] INT8 + per-(token, head) FP16 scale.
    assert call["query"].dtype == torch.int8
    assert call["query"].shape == (_ROWS, _INDEX_N_HEADS, _INDEX_HEAD_DIM)
    assert call["query_scale"].dtype == torch.float16
    assert call["query_scale"].shape == (_ROWS, _INDEX_N_HEADS)
    # PA_BBND key: the main INT8 index plane, [blocks, tokens, 1, head], plus
    # the FP16 per-token scale plane [blocks, tokens, 1].
    num_k_heads = csa2._QLI_V3_KEY_NUM_HEADS
    assert call["key"] is shared.index_cache
    assert call["key"].dtype == torch.int8
    assert call["key"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, num_k_heads, _INDEX_HEAD_DIM)
    assert call["key_scale"] is shared.index_scale
    assert call["key_scale"].dtype == torch.float16
    assert call["key_scale"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, num_k_heads)
    # weights are the per-head scoring coefficients, FP16.
    assert call["weights"].dtype == torch.float16
    assert call["weights"].shape == (_ROWS, _INDEX_N_HEADS)
    assert call["topk"] == _INDEX_TOPK
    assert call["quant_mode"] == 2
    # PA_BBND batch descriptors: the TOKEN-group block table addresses the
    # paged key; cu_seqlens_k must be absent (tiling rejects it).
    assert call["block_table"] is shared.index_block_table
    assert call["block_table"].dtype == torch.int32
    assert call["cu_seqlens_q"].dtype == torch.int32
    assert call["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]
    assert call["cu_seqlens_k"] is None
    assert call["seqused_k"].tolist() == _CTX_LENS
    assert call["cmp_residual_k"] is None  # ratio 1 has no residual semantics
    assert call["metadata"].shape == (csa2._QLI_V3_METADATA_SIZE,) and call["metadata"].dtype == torch.int32
    assert call["layout_q"] == "TND"
    assert call["layout_k"] == "PA_BBND"
    assert call["mask_mode"] == 3
    assert call["cmp_ratio"] == 1
    # No candidate source configured for this backend
    # (``candidate_source_layer_id == -1``): mode 3, and no candidate operands
    # are passed (the historical call). The per-layer mode selection (1/2/3) is
    # pinned by ``test_qli_v3_candidate_mode_is_chosen_per_layer``.
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_OFF
    assert "candidate_topk_index" not in call
    assert "candidate_topk_blocks" not in call
    assert "candidate_block_size" not in call
    assert call["max_seqlen_q"] == 1
    # The op's result is re-based to the eager window-offset contract.
    assert topk.shape == (_ROWS, 1, _INDEX_TOPK)
    assert topk[0, 0].tolist() == [_WINDOW + 1, _WINDOW + 3, -1, -1]
    assert bool((topk[1:] == -1).all())


def test_qli_rebase_filters_invisible_and_sorts() -> None:
    """Picks at/past ``visible`` are dropped; the rest are position-sorted."""
    picks = torch.tensor([[3, 0, -1, 7, 1]], dtype=torch.int32)
    visible = torch.tensor([[4]], dtype=torch.int32)  # rows 0..3 are committed
    out = csa2._rebase_qli_picks(picks, visible, offset=100)
    assert out.tolist() == [[100, 101, 103, -1, -1]]
    assert out.dtype == torch.int32


def test_qli_rebase_boundary_and_all_invalid() -> None:
    """The bound is exclusive (``idxs < visible``) and padding stays last."""
    picks = torch.tensor([[2, 1], [-1, -1]], dtype=torch.int32)
    visible = torch.tensor([[2], [3]], dtype=torch.int32)
    out = csa2._rebase_qli_picks(picks, visible, offset=10)
    # Row 0: 2 == visible is out of range, 1 is valid -> [11, -1].
    # Row 1: no picks at all.
    assert out.tolist() == [[11, -1], [-1, -1]]


def test_qli_rebase_matches_eager_topk_contract() -> None:
    """The helper is the eager ``_indexer_topk`` filter+sort on an already-ranked input."""
    score = torch.tensor([[0.1, 0.9, 0.5, 0.3]], dtype=torch.float32)
    visible = torch.tensor([[3]], dtype=torch.int32)
    offset = 7
    eager = csa2._indexer_topk(score, visible.reshape(-1), index_topk=4, offset=offset)
    # The op pre-selects by score: positions [1, 2, 0] then 3 (invisible).
    picks = torch.tensor([[1, 2, 0, 3]], dtype=torch.int32)
    qli = csa2._rebase_qli_picks(picks, visible, offset=offset)
    assert qli.tolist() == eager.tolist() == [[7, 8, 9, -1]]


def test_qli_rebase_fp32_sort_is_bitwise_equal_to_int32_sort() -> None:
    """The FP32 kernel swap is value-identical over the compressed-index range.
    Covers duplicates (zero-score ties), out-of-range picks, all-padding rows and mixed visibility; the FP32 path must"""
    generator = torch.Generator().manual_seed(11)
    for fraction in (0.0, 0.25, 0.6):
        rows = 64
        picks = torch.randint(-1, 40, (rows, 16), generator=generator, dtype=torch.int32)
        visible = torch.randint(0, 41, (rows, 1), generator=generator, dtype=torch.int32)
        if fraction == 0.0:
            picks = torch.full((rows, 16), -1, dtype=torch.int32)
        int32_out = csa2._rebase_qli_picks(picks, visible, offset=1000, fp32_sort=False)
        fp32_out = csa2._rebase_qli_picks(picks, visible, offset=1000, fp32_sort=True)
        assert torch.equal(fp32_out, int32_out)


def test_qli_rebase_fp32_sort_handles_the_topk_width_boundary() -> None:
    """A row whose picks fill the whole width matches the INT32 sort exactly."""
    picks = torch.tensor([[5, 2, 9, 0, 3, 1, 8, 7]], dtype=torch.int32)
    visible = torch.tensor([[16]], dtype=torch.int32)
    int32_out = csa2._rebase_qli_picks(picks, visible, offset=3, fp32_sort=False)
    fp32_out = csa2._rebase_qli_picks(picks, visible, offset=3, fp32_sort=True)
    # sorted(picks) + offset = [0,1,2,3,5,7,8,9] + 3
    assert int32_out.tolist() == fp32_out.tolist() == [[3, 4, 5, 6, 8, 10, 11, 12]]


def test_qli_v3_filters_out_of_range_picks_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end: an out-of-range op pick never becomes a window row."""
    _reset_log_state(monkeypatch)
    recorder = _QliRecorder()
    real_indexer = recorder.indexer

    def indexer(
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        picks, extra_a, extra_b = real_indexer(query, key, weights, query_scale, key_scale, topk, quant_mode, **kwargs)
        rows = picks.size(0)
        picks = torch.full((rows, 1, topk), -1, dtype=torch.int32)
        if rows:
            # Ratio-2 layer with ctx_lens[0] = 300 -> seqused_k[2] = 150.
            # 150 is at the bound (dropped), 3 is a duplicate, order descends.
            picks[0, 0, 0] = 150
            picks[0, 0, 1] = 3
            picks[0, 0, 2] = 3
            picks[0, 0, 3] = 1
        return picks, extra_a, extra_b

    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: (indexer, recorder.metadata))
    backend = _make_backend()

    topk = _drive(backend, _StubIndexerLayer(1, 2), _make_dsa(), _make_shared(2))

    # 150 was filtered; {1, 3, 3} rebased by window + 1 and sorted ascending.
    assert topk[0, 0].tolist() == [_WINDOW + 2, _WINDOW + 4, _WINDOW + 4, -1]
    assert bool((topk[1:] == -1).all())


def test_qli_v3_ratio2_uses_the_compressed_capacity_and_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()

    _drive(backend, _StubIndexerLayer(1, 2), _make_dsa(), _make_shared(2))

    call = recorder.indexer_calls[0]
    # The paged shadow is addressed by the block table, so its shape does not
    # depend on the ratio; only the per-sequence used lengths do.
    assert call["key"].shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, 1, _INDEX_HEAD_DIM)
    assert call["seqused_k"].tolist() == [length // 2 for length in _CTX_LENS]
    assert call["cmp_residual_k"].tolist() == [length % 2 for length in _CTX_LENS]
    assert call["cmp_ratio"] == 2
    assert call["layout_k"] == "PA_BBND"


# ---------------------------------------------------------------------------
# 2) metadata contract (per-ratio, from the persistent state)
# ---------------------------------------------------------------------------


def test_qli_v3_metadata_kwargs_per_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()

    # A ratio-1 dispatch builds the state, which fills BOTH ratio classes.
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert len(recorder.metadata_calls) == 2
    by_ratio = {kwargs["cmp_ratio"]: (args, kwargs) for args, kwargs in recorder.metadata_calls}
    assert set(by_ratio) == {1, 2}
    for ratio, (args, kwargs) in by_ratio.items():
        assert args[:3] == (_INDEX_N_HEADS, 1, _INDEX_HEAD_DIM)
        assert args[3] == min(_INDEX_TOPK, max(_CAPACITY // ratio, 1))  # topk
        assert args[4] == 2  # quant_mode
        assert kwargs["batch_size"] == _ROWS
        assert kwargs["max_seqlen_q"] == 1
        # The per-bucket per-sequence capacity bound (vllm-ascend's
        # ``max_cache_seq_len`` static upper bound).
        assert kwargs["max_seqlen_k"] == max(_CAPACITY // ratio, 1)
        assert kwargs["layout_q"] == "TND"
        assert kwargs["layout_k"] == "PA_BBND"
        assert kwargs["mask_mode"] == csa2._QLI_V3_MASK_MODE
        # PA_BBND: cu_seqlens_k must be absent, the block table + seq-used
        # address the paged key instead.
        assert kwargs["cu_seqlens_k"] is None
        assert kwargs["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]
        assert kwargs["seqused_k"].tolist() == [length // ratio for length in _CTX_LENS]
        if ratio > 1:
            assert kwargs["cmp_residual_k"].tolist() == [length % ratio for length in _CTX_LENS]
        else:
            # ratio-1 keeps no residual semantics; the vendor metadata host
            # check no longer dereferences ``cmp_residual_k`` before its
            # IsTensorExist guard (fixed in xllm_ops ``...metadata_va_check.h``),
            # so ``None`` is now the canonical value, matching the indexer op.
            assert kwargs["cmp_residual_k"] is None


def test_qli_v3_state_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa()

    _drive(backend, _StubIndexerLayer(2, 1), dsa, _make_shared(1))

    assert set(backend._qli_state) == {(_ROWS, _CAPACITY)}
    state = backend._qli_state[(_ROWS, _CAPACITY)]
    assert isinstance(state, _QliGraphState)
    assert set(state.metadata) == {1, 2}
    assert state.cu_seqlens_q.dtype == torch.int32
    # PA_BBND has no constant cu_seqlens_k operand on the state.
    assert not hasattr(state, "cu_seqlens_k")
    assert torch.equal(state.seqused_k[2], torch.div(dsa.seq_lens, 2, rounding_mode="floor"))
    assert torch.equal(state.cmp_residual[2], torch.remainder(dsa.seq_lens, 2))

    # A replay-like refresh reuses the same buffers (no reallocation) and picks
    # up the new sequence lengths.
    metadata_before = state.metadata[2]
    seqused_before = state.seqused_k[2]
    dsa.seq_lens = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    backend._refresh_qli_state(dsa, state)
    assert state.metadata[2] is metadata_before
    assert state.seqused_k[2] is seqused_before
    assert state.seqused_k[1].tolist() == [10, 11, 12, 13]
    assert state.cmp_residual[2].tolist() == [0, 1, 0, 1]


def test_qli_v3_state_refuses_to_build_under_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing bucket state must fail loudly inside capture, never allocate."""
    _reset_log_state(monkeypatch)
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_backend()
    monkeypatch.setattr(csa2, "_is_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="eager warmup"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))


# ---------------------------------------------------------------------------
# 3) read-side INT8 quantization
# ---------------------------------------------------------------------------


def _reference_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference of the vllm ``quantize_indexer_query`` rounding."""
    absmax = x.float().abs().amax(dim=-1)
    scale = (absmax / 127.0).to(torch.float16).to(torch.float32)
    scale = torch.clamp(scale, min=2.0**-24)
    codes = torch.round(x.float() / scale.unsqueeze(-1)).clamp(-127.0, 127.0).to(torch.int8)
    return codes, scale.to(torch.float16)


def test_quantize_indexer_rows_int8_matches_the_reference() -> None:
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(6, 128, generator=gen) * 3.0).to(torch.bfloat16)
    codes, scale = csa2._quantize_indexer_rows_int8(x)

    ref_codes, ref_scale = _reference_int8(x)
    assert codes.dtype == torch.int8 and codes.shape == x.shape
    assert scale.dtype == torch.float16 and scale.shape == (6,)
    assert torch.equal(codes, ref_codes)
    assert torch.equal(scale, ref_scale)
    # Symmetric range and a non-vacuous scale.
    assert bool((codes >= -127).all()) and bool((codes <= 127).all())
    assert bool((scale > 0).all())

    # A zero row keeps a defined (smallest) scale instead of dividing by zero.
    zeros = torch.zeros(2, 128, dtype=torch.bfloat16)
    zero_codes, zero_scale = csa2._quantize_indexer_rows_int8(zeros)
    assert bool((zero_codes == 0).all())
    assert torch.equal(zero_scale, torch.full((2,), 2.0**-24, dtype=torch.float16))


def test_qli_v3_quantize_indexer_rows_boundaries() -> None:
    """The INT8 index quantizer; pin its boundary semantics.
    Full-scale (absmax == 127 -> unit scale), over-range (clamped to 127) and the all-zero row are defined by the"""
    full = torch.tensor([[127.0, -127.0, 0.0, 63.5]], dtype=torch.bfloat16)
    codes, scale = csa2._quantize_indexer_rows_int8(full)
    ref_codes, ref_scale = _reference_int8(full)
    assert torch.equal(codes, ref_codes) and torch.equal(scale, ref_scale)
    assert scale[0].item() == pytest.approx(1.0)
    assert codes[0].tolist() == [127, -127, 0, 64]

    # absmax 254 -> scale exactly 2.0, both extremes saturate at +-127.
    over = torch.tensor([[254.0, -254.0, 0.0, 0.0]], dtype=torch.bfloat16)
    over_codes, over_scale = csa2._quantize_indexer_rows_int8(over)
    assert over_scale[0].item() == pytest.approx(2.0)
    assert over_codes[0, 0] == 127 and over_codes[0, 1] == -127

    special = torch.tensor(
        [[float("nan")] * 4, [float("inf"), -float("inf"), 0.0, 1.0]],
        dtype=torch.float32,
    ).to(torch.bfloat16)
    special_codes, special_scale = csa2._quantize_indexer_rows_int8(special)
    ref_special_codes, ref_special_scale = _reference_int8(special)
    assert torch.equal(special_codes, ref_special_codes)
    # equal_nan: NaN scale propagates, so plain torch.equal would be False.
    assert torch.allclose(special_scale.float(), ref_special_scale.float(), equal_nan=True)
    # Unspecified-but-observed: NaN scale propagates (documented, not asserted).
    assert torch.isnan(special_scale[0])


# ---------------------------------------------------------------------------
# 4) native INT8 index plane (write side)
# ---------------------------------------------------------------------------


def _make_writer_backend(*, candidate_source_layer_id: int = -1) -> Csa2AttentionBackend:
    """A plan whose layer 1 is a ratio-1 FULL kv source (owns key + index)."""
    return _make_backend(
        n_layers=2,
        compress_ratios=[0, 1],
        kv_source_layer_ids=[1],
        index_source_layer_ids=[1],
        candidate_source_layer_id=candidate_source_layer_id,
    )


def _writer_dsa(rows: int) -> SimpleNamespace:
    # Layer 1 cache slots: [key(TOKEN), index(TOKEN), swa]; one sequence owning
    # physical block 0, so its committed rows land at flat slots 0..rows-1.
    block_zero = torch.zeros(1, 1, dtype=torch.int32)
    return SimpleNamespace(
        slot_mappings=[],
        # Per-layer tables (layer 1 only), slots [key(TOKEN), index(TOKEN), swa].
        block_tables=[[], [block_zero.clone(), block_zero, block_zero.clone()]],
        seq_lens=torch.tensor([rows], dtype=torch.int32),
        is_acl_graph=False,
        graph_token_capacity=0,
    )


def _run_writer(
    backend: Csa2AttentionBackend,
    layer: _StubWriterLayer,
    cache: _StubLayerCache,
    shared: CSA2SharedRuntime,
    *,
    rows: int,
) -> None:
    ctx = _make_ctx(rows)
    mapping = backend._resolve_cache_mapping(layer.layer_id, layer.mode.compress_ratio)
    backend._run_compressor(
        layer,
        ctx,
        _writer_dsa(rows),
        cache,
        mapping,
        [rows],
        torch.tensor([rows], dtype=torch.int32),
        shared,
    )


def test_qli_v3_index_plane_quantizes_and_scatters(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_writer_backend()
    layer = _StubWriterLayer(1)
    cache = _StubLayerCache()
    shared = CSA2SharedRuntime()

    _run_writer(backend, layer, cache, shared, rows=5)

    # The main index plane is the INT8 cache the platform policy allocates.
    assert cache.index.dtype == torch.int8
    assert cache.index.shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS, _INDEX_HEAD_DIM)
    assert cache.indexer_scale.dtype == torch.float16
    assert cache.indexer_scale.shape == (_SHADOW_BLOCKS, _BLOCK_SIZE, csa2._QLI_V3_KEY_NUM_HEADS)
    assert shared.index_cache is cache.index
    assert shared.index_scale is cache.indexer_scale

    # The pre-rounding rope'd rows are per-row INT8 quantized into the index
    # plane and their FP16 scales into the scale plane, at the same TOKEN slots.
    latent = layer.compress_rmsnorm(layer.compress_kv_score(_make_ctx(5).x)[0])
    index_k = apply_interleaved_rope(layer.indexer_key(latent), torch.arange(5), _make_ctx(5).cos_sin_cache)
    expected_codes, expected_scales = csa2._quantize_indexer_rows_int8(index_k)
    # Rows 0..4 land at flat slots 0..4 (block 0).
    assert torch.equal(cache.index.view(-1, _INDEX_HEAD_DIM)[:5], expected_codes)
    assert torch.equal(cache.indexer_scale.view(-1)[:5], expected_scales)

    # A second write reuses the same cache tensors (only the rows change).
    _run_writer(backend, layer, cache, shared, rows=5)
    assert shared.index_cache is cache.index
    assert shared.index_scale is cache.indexer_scale


def test_qli_v3_eager_layers_write_the_int8_plane(monkeypatch) -> None:
    """The native INT8 main plane is written regardless of the fused
    dispatch."""
    _reset_log_state(monkeypatch)
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_writer_backend()
    shared = CSA2SharedRuntime()

    _run_writer(backend, _StubWriterLayer(1), _StubLayerCache(), shared, rows=5)

    assert shared.index_cache is not None and shared.index_cache.dtype == torch.int8
    assert shared.index_scale is not None and shared.index_scale.dtype == torch.float16


def test_qli_v3_index_plane_written_for_candidate_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate-pool layers stay eager and still write the INT8 index plane."""
    _reset_log_state(monkeypatch)
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_writer_backend(candidate_source_layer_id=1)
    layer = _StubWriterLayer(1)
    shared = CSA2SharedRuntime()

    _run_writer(backend, layer, _StubLayerCache(), shared, rows=5)

    assert shared.index_cache is not None
    assert shared.index_scale is not None


def test_qli_v3_mask_mode_is_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both call sites pin ``_QLI_V3_MASK_MODE`` instead of either endpoint default."""
    assert csa2._QLI_V3_MASK_MODE == 3
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()

    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))

    assert recorder.indexer_calls[0]["mask_mode"] == csa2._QLI_V3_MASK_MODE


# ---------------------------------------------------------------------------
# 5) guard branches + one-shot logging
# ---------------------------------------------------------------------------


def _guard_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    backend: Csa2AttentionBackend | None = None,
    layer: _StubIndexerLayer | None = None,
    dsa: SimpleNamespace | None = None,
    q_lens: list[int] | None = None,
    shared: CSA2SharedRuntime | None = None,
) -> tuple[Csa2AttentionBackend, _StubIndexerLayer, SimpleNamespace, list[int], CSA2SharedRuntime]:
    backend = backend or _make_backend()
    return (
        backend,
        layer or _StubIndexerLayer(2, 1),
        dsa or _make_dsa(),
        [1] * _ROWS if q_lens is None else q_lens,
        shared or _make_shared(1),
    )


def test_qli_v3_fallback_reason_reports_each_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_ops(monkeypatch, _QliRecorder())

    backend, layer, dsa, q_lens, shared = _guard_kwargs(monkeypatch)
    assert backend._qli_v3_fallback_reason(layer, dsa, q_lens, shared, shared.index_cache) is None

    off_graph = _make_dsa(is_acl_graph=False)
    assert (
        backend._qli_v3_fallback_reason(layer, off_graph, q_lens, shared, shared.index_cache)
        == "not an ACL-graph decode step"
    )
    assert backend._qli_v3_fallback_reason(layer, dsa, [2, 1, 1, 1], shared, shared.index_cache) == "query length != 1"

    # Static geometry: the op's head width contract.
    narrow = _make_backend(index_head_dim=16)
    assert "unsupported index head dim 16" in narrow._qli_v3_fallback_reason(
        layer, dsa, q_lens, shared, shared.index_cache
    )

    no_capacity = _make_dsa(capacity=0)
    assert (
        backend._qli_v3_fallback_reason(layer, no_capacity, q_lens, shared, shared.index_cache)
        == "missing static graph token capacity"
    )

    # Candidate-pool layers (owner and consumer) with a valid config fall
    # through to the same plane-readiness clauses; an out-of-contract config
    # is pinned by the invalid-config test
    # (``test_qli_v3_candidate_invalid_config_falls_back_with_a_warning``).
    candidate_backend = _make_backend(candidate_source_layer_id=1)
    not_ready = _make_shared(2, with_scale=False)
    assert (
        candidate_backend._qli_v3_fallback_reason(
            _StubIndexerLayer(1, 2), dsa, q_lens, not_ready, not_ready.index_cache
        )
        == "index INT8 plane not ready (scale is None)"
    )

    # Missing / malformed INT8 index plane.
    for bad, expected in (
        (_make_shared(1, with_scale=False), "index INT8 plane not ready"),
        (_make_shared(1, index_dtype=torch.bfloat16), "index plane dtype mismatch"),
        (_make_shared(1, key_shape=(8, 128, 128)), "index key must be"),
        (_make_shared(1, key_shape=(_SHADOW_BLOCKS, _BLOCK_SIZE, 1, 64)), "does not match"),
        (_make_shared(1, key_shape=(4, _BLOCK_SIZE, 1, _INDEX_HEAD_DIM)), "index scale geometry"),
        (_make_shared(1, with_block_table=False), "TOKEN-group block table"),
    ):
        reason = backend._qli_v3_fallback_reason(layer, dsa, q_lens, bad, bad.index_cache)
        assert reason is not None and expected in reason, (bad, reason)

    # A missing op is the last guard and reports itself.
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    reason = backend._qli_v3_fallback_reason(layer, dsa, q_lens, shared, shared.index_cache)
    assert reason is not None and "not registered" in reason


def test_qli_v3_falls_back_and_keeps_the_eager_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every designed fallback must leave the pure-torch result untouched."""
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_backend()

    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    _reset_log_state(monkeypatch)
    eager = _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))
    _install_ops(monkeypatch, recorder)

    # (a) a genuine guard failure (head mismatch) warns once and stays eager.
    _reset_log_state(monkeypatch)
    spy = _SpyLogger()
    monkeypatch.setattr(csa2, "logger", spy)
    narrow = _make_backend(index_head_dim=16)
    assert torch.equal(_drive(narrow, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1)), eager)

    # (b) a designed silent fallback (not ACL graph) does not warn.
    off_graph = _make_dsa(is_acl_graph=False)
    assert torch.equal(_drive(backend, _StubIndexerLayer(2, 1), off_graph, _make_shared(1)), eager)

    # The designed fallbacks (not an ACL-graph step) are silent by contract;
    # the genuine (a) guard failure is the one and only warning, consumed
    # one-shot.
    assert len(spy.warnings) == 1 and "unsupported index head dim 16" in spy.warnings[0]
    assert recorder.indexer_calls == []


def test_qli_v3_candidate_layers_stay_eager(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the fused op unavailable the candidate layers stay eager and the
    source still publishes the eager bool-mask pool."""
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    _reset_log_state(monkeypatch)
    backend = _make_backend(candidate_source_layer_id=1)
    dsa = _make_dsa()

    # The pool owner (== candidate source) takes the eager path and publishes
    # the pool; the consumer after it is a candidate layer too (asserted through
    # the guard, since its eager read needs that pool).
    shared = _make_shared(2)
    _drive(backend, _StubIndexerLayer(1, 2), dsa, shared)
    assert shared.candidates is not None
    consumer = _make_shared(1)
    consumer_reason = backend._qli_v3_fallback_reason(
        _StubIndexerLayer(2, 1), dsa, [1] * _ROWS, consumer, consumer.index_cache
    )
    assert consumer_reason is not None and "not registered" in consumer_reason

    assert shared.candidate_blocks is None


# ---------------------------------------------------------------------------
# 5b) two-level candidate pool (config-driven)
# ---------------------------------------------------------------------------


def _candidate_plan_backend(**kwargs: Any) -> Csa2AttentionBackend:
    """A plan whose layer 1 is the candidate source and layer 2 consumes it.
    Both layers share the ratio-1 index group (the production candidate layers are all ratio 1), so"""
    params: dict[str, Any] = {"compress_ratios": [0, 1, 1], "candidate_source_layer_id": 1}
    params.update(kwargs)
    return _make_backend(**params)


def test_qli_v3_candidate_mode_is_chosen_per_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """The op mode is a function of the layer's role, never a constant 3.
    Production roles: the pre-pool layers {2, 8, 14} stay 3, the candidate source (layer 20) is 1, and the pool consumers"""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    plain = _make_backend(compress_ratios=[0, 1, 1])  # no candidate source
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)  # source -> 1
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)  # consumer -> 2
    _drive(plain, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))  # -> 3

    assert [call["candidate_mode"] for call in recorder.indexer_calls] == [
        csa2._QLI_V3_CAND_MODE_SOURCE,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_OFF,
    ]


def test_qli_v3_candidate_source_publishes_the_block_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """The source layer runs mode 1 and publishes the op's third output."""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)

    call = recorder.indexer_calls[0]
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_SOURCE == 1
    # A source publishes; it never consumes a pool.
    assert "candidate_topk_index" not in call
    assert call["candidate_topk_blocks"] == 2048
    assert call["candidate_block_size"] == 8
    # The op returns ``[rows, 1, candidate_topk_blocks]`` INT32 for mode 1; it
    # is parked on the shared runtime for this forward's consumers.
    assert shared.candidate_blocks is not None
    assert shared.candidate_blocks.dtype == torch.int32
    assert shared.candidate_blocks.shape == (_ROWS, 1, 2048)
    assert shared.candidate_blocks_layer == 1


def test_qli_v3_candidate_consumer_uses_the_published_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """A consumer layer runs mode 2 and feeds the source's pool back to the op."""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    published = shared.candidate_blocks
    assert published is not None
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    call = recorder.indexer_calls[-1]
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_CONSUMER == 2
    assert call["candidate_topk_index"] is published
    assert call["candidate_topk_blocks"] == 2048
    assert call["candidate_block_size"] == 8


def test_qli_v3_candidate_consumer_requires_the_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mode 2 without a source-published pool fails loudly (never a silent wrong pick)."""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()

    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), _make_shared(1))


def test_qli_v3_candidate_layers_fall_back_when_the_op_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the fused op unavailable the candidate layers stay eager, byte for
    byte: the source publishes the eager bool-mask pool, the consumer reads it,
    and no block-index pool is ever published."""
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    _reset_log_state(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)

    for layer_id in (1, 2):
        reason = backend._qli_v3_fallback_reason(
            _StubIndexerLayer(layer_id, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
        )
        assert reason is not None and "not registered" in reason

    # The whole forward stays on the eager path: the source publishes the eager
    # bool-mask pool and the consumer reads it; the fused op is never called.
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)
    assert shared.candidates is not None
    assert shared.candidate_blocks is None


def test_qli_v3_candidate_invalid_config_falls_back_with_a_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An out-of-contract candidate config is a reported fallback, not a silent one."""
    recorder = _stub_qli(monkeypatch)
    spy = _SpyLogger()
    monkeypatch.setattr(csa2, "logger", spy)

    # block_size != 8 is outside the A3 tiling contract.
    bad_block = _candidate_plan_backend(candidate_block_size=16)
    shared = _make_shared(1)
    reason = bad_block._qli_v3_fallback_reason(
        _StubIndexerLayer(1, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
    )
    assert reason is not None and "candidate block size 16" in reason
    _drive(bad_block, _StubIndexerLayer(1, 1), _make_dsa(), _make_shared(1))
    assert recorder.indexer_calls == []
    assert any("candidate block size 16" in message for message in spy.warnings)

    # topk_blocks must be a positive multiple of 64 and at most 2048.
    _reset_log_state(monkeypatch)
    bad_topk = _candidate_plan_backend(candidate_topk_blocks=1000)
    shared = _make_shared(1)
    reason = bad_topk._qli_v3_fallback_reason(
        _StubIndexerLayer(1, 1), _make_dsa(), [1] * _ROWS, shared, shared.index_cache
    )
    assert reason is not None and "candidate topk blocks 1000" in reason


def test_qli_v3_candidate_active_log_fires_when_fused_and_stays_silent_when_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The positive rank-log marker fires on a fused candidate layer, one-shot.
    Acceptance greps the exact string ``QLI v3 candidate indexer active``: it must appear once per rank when a candidate"""
    recorder = _stub_qli(monkeypatch)
    spy = _SpyLogger()
    monkeypatch.setattr(csa2, "logger", spy)

    # The source (mode 1) fires the marker; a repeat is a one-shot no-op.
    _reset_log_state(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)
    source_marker = [m for m in spy.infos if "QLI v3 candidate indexer active" in m]
    assert len(source_marker) == 1
    assert "candidate indexer active" in source_marker[0]

    # The consumer (mode 2) path also reaches the marker when it is the first
    # fused candidate layer (fresh rank state; the source already published).
    _reset_log_state(monkeypatch)
    spy.infos.clear()
    backend = _candidate_plan_backend()
    shared = _make_shared(1)
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), shared)  # publish the pool
    _reset_log_state(monkeypatch)  # a fresh rank's first candidate layer
    spy.infos.clear()
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)  # consumer
    assert any("QLI v3 candidate indexer active" in m for m in spy.infos)


def test_qli_v3_candidate_consumer_rejects_a_pool_whose_marker_mismatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty pool stamped by another layer is rejected, not consumed.
    The consumer's defensive invariant is "the pool exists AND its producer marker is this forward's source layer"."""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()
    shared = _make_shared(1)
    shared.candidate_blocks = torch.zeros(1, 1, 8, dtype=torch.int32)
    shared.candidate_blocks_layer = 99

    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)
    assert recorder.indexer_calls == []


def test_qli_v3_candidate_pool_does_not_leak_across_forwards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A forward's published pool never leaks into the next forward.
    The non-graph path calls ``reset_forward`` between forwards, which drops the shared runtime; the next forward rebinds"""
    recorder = _stub_qli(monkeypatch)
    backend = _candidate_plan_backend()

    first = backend._shared_runtime()
    _seed_index_plane(first, _make_shared(1))
    _drive(backend, _StubIndexerLayer(1, 1), _make_dsa(), first)
    assert first.candidate_blocks is not None
    assert first.candidate_blocks_layer == 1

    backend.reset_forward()
    second = backend._shared_runtime()
    assert second is not first
    assert second.candidate_blocks is None
    assert second.candidate_blocks_layer == -1

    # The next forward's index plane is seeded (as the compressor would), but
    # its consumer still cannot see the previous forward's pool.
    _seed_index_plane(second, _make_shared(1))
    with pytest.raises(RuntimeError, match="requires the candidate pool"):
        _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), second)


def test_qli_v3_fallback_and_active_logging_is_one_shot(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SpyLogger()
    monkeypatch.setattr(csa2, "logger", spy)
    _reset_log_state(monkeypatch)

    # A designed silent fallback: silent, and it must NOT consume the one-shot,
    # so a later genuine guard failure still reports itself.
    for reason in ("not an ACL-graph decode step", "query length != 1"):
        csa2._log_qli_v3_fallback(reason)
        csa2._log_qli_v3_fallback(reason)
    assert spy.warnings == []
    assert csa2._qli_v3_fallback_logged is False

    # a genuine guard failure: exactly one warning.
    csa2._log_qli_v3_fallback("unsupported index head dim 16")
    csa2._log_qli_v3_fallback("unsupported index head dim 16")
    assert len(spy.warnings) == 1 and "QLI v3" in spy.warnings[0]
    assert spy.infos == []

    # active path: exactly one info.
    csa2._log_qli_v3_active()
    csa2._log_qli_v3_active()
    assert len([message for message in spy.infos if "QLI v3" in message]) == 1


# ---------------------------------------------------------------------------
# 5c) native ``xllm_ops`` QLI v3 op resolver
# ---------------------------------------------------------------------------


def test_qli_v3_op_names_are_the_native_va_handles() -> None:
    """The fused QLI v3 path resolves the native "_va" handles only.

    The candidate indexer is the migrated "_va" form of the V2 kernel, so the
    indexer handle is ``quant_lightning_indexer_v2_va`` -- there is no separate
    "v3" kernel and no ``quant_lightning_indexer_v3_va`` handle.
    """
    assert csa2._QLI_V3_OP_NAME == "quant_lightning_indexer_v2_va"
    assert csa2._QLI_V3_METADATA_OP_NAME == "quant_lightning_indexer_v2_metadata_va"
    # The stock (non-_va) metadata producer is NOT accepted by this path.
    assert csa2._QLI_V3_METADATA_OP_NAME != "quant_lightning_indexer_v2_metadata"


def test_qli_v3_ops_resolves_when_both_native_handles_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both native schemas registered in ``xllm_ops`` -> a successful resolution."""
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    indexer = object()
    metadata = object()
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
        raising=False,
    )

    assert csa2._get_qli_v3_ops() == (indexer, metadata)


@pytest.mark.parametrize(
    "present",
    [
        {"quant_lightning_indexer_v2_va": object()},
        {"quant_lightning_indexer_v2_metadata_va": object()},
        {},
    ],
)
def test_qli_v3_ops_requires_both_native_handles(
    monkeypatch: pytest.MonkeyPatch,
    present: dict[str, Any],
) -> None:
    """Only one (or neither) native schema present -> the pair does not resolve."""
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", SimpleNamespace(**present), raising=False)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_returns_none_without_the_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No ``torch.ops.xllm_ops`` namespace at all -> ``None``."""
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", None, raising=False)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_ignores_the_temporary_wheel_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A temporary-wheel registration must NOT yield a false success.

    The retired temporary ``cann_ops_transformer_dsv41`` wheel registered into
    ``torch.ops.cann_ops_transformer``; the fused path no longer accepts that
    source (and the shim module that imported it is gone), so a process holding
    only that namespace -- or the stock non-_va metadata handle -- resolves to
    ``None``.
    """
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(
        torch.ops,
        "cann_ops_transformer",
        SimpleNamespace(quant_lightning_indexer_v3=object()),
        raising=False,
    )
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(quant_lightning_indexer_v2_metadata=object()),
        raising=False,
    )
    assert csa2._get_qli_v3_ops() is None

    # The "_va" metadata handle alone is not enough either: the migrated
    # indexer must be registered by the same extension.
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(quant_lightning_indexer_v2_metadata_va=object()),
        raising=False,
    )
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    assert csa2._get_qli_v3_ops() is None


def test_qli_v3_ops_caches_a_hit_and_not_a_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful resolution is cached; a miss is re-probed on the next call."""
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    indexer = object()
    metadata = object()
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
        raising=False,
    )

    resolved = csa2._get_qli_v3_ops()
    assert resolved == (indexer, metadata)
    # A hit is served from cache even after the namespace is torn down.
    monkeypatch.setattr(torch.ops, "xllm_ops", None, raising=False)
    assert csa2._get_qli_v3_ops() is resolved

    # A miss is not cached: a later registration is picked up.
    monkeypatch.setattr(csa2, "_qli_v3_ops", None)
    monkeypatch.setattr(torch.ops, "xllm_ops", SimpleNamespace(), raising=False)
    assert csa2._get_qli_v3_ops() is None
    monkeypatch.setattr(
        torch.ops,
        "xllm_ops",
        SimpleNamespace(
            quant_lightning_indexer_v2_va=indexer,
            quant_lightning_indexer_v2_metadata_va=metadata,
        ),
        raising=False,
    )
    assert csa2._get_qli_v3_ops() == (indexer, metadata)


# ---------------------------------------------------------------------------
# 6) eager read: dequantize the INT8 index plane with the scale plane
# ---------------------------------------------------------------------------


def test_qli_v3_eager_read_dequantizes_the_int8_index_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The eager indexer gathers INT8 codes and scales them back per row.
    Independent reference of the read-side dequantization: the score operand must equal ``codes.float() * scale.float()``"""
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)  # eager
    _reset_log_state(monkeypatch)
    backend = _make_backend()

    codes = torch.randint(-127, 128, (_SHADOW_BLOCKS, _BLOCK_SIZE, 1, _INDEX_HEAD_DIM), dtype=torch.int8)
    scales = (torch.rand(_SHADOW_BLOCKS, _BLOCK_SIZE, 1) + 0.1).to(torch.float16)
    shared = CSA2SharedRuntime(
        index_cache=codes,
        index_scale=scales,
        index_block_table=torch.arange(_ROWS * 2, dtype=torch.int32).view(_ROWS, 2),
        compress_ratio=1,
    )

    seen: dict[str, torch.Tensor] = {}
    real_score = csa2._indexer_score

    def spy_score(q: torch.Tensor, index_k: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if "k" not in seen:
            seen["k"] = index_k.clone()
        return real_score(q, index_k, weights)

    monkeypatch.setattr(csa2, "_indexer_score", spy_score)
    _drive(backend, _StubIndexerLayer(2, 1), _make_dsa(), shared)

    # Sequence 0's block table is [0, 1], so logical rows 0.._CAPACITY-1 map to
    # flat slots 0.._CAPACITY-1; dequantize them row-wise.
    expected = codes.view(-1, _INDEX_HEAD_DIM)[:_CAPACITY].float() * scales.view(-1)[:_CAPACITY].float().unsqueeze(-1)
    assert torch.allclose(seen["k"], expected)
    assert not torch.equal(seen["k"], codes.view(-1, _INDEX_HEAD_DIM)[:_CAPACITY].float())


# -- QLI picks order: why the sort stays ---------------------------------------
#
# ``_rebase_qli_picks`` always sorts: it mirrors the eager indexer's position
# sort and keeps the picks contract stable for fused consumers, which walk
# just the left-aligned valid prefix of each index row.


def _reference_rebase(
    picks: torch.Tensor,
    visible: torch.Tensor,
    offset: int,
    *,
    sort: bool,
) -> torch.Tensor:
    """The ``_rebase_qli_picks`` filter+rebase with the sort toggled.
    Models the unsorted picks the reverted switch would have produced; the production helper always sorts."""
    sentinel = torch.iinfo(torch.int32).max
    sortable = torch.where(
        (picks >= 0) & (picks < visible),
        picks,
        torch.full_like(picks, sentinel),
    )
    if sort:
        sortable = sortable.sort(dim=-1).values
    picks = torch.where(sortable >= sentinel, torch.full_like(sortable, -1), sortable)
    return torch.where(picks >= 0, picks + offset, torch.full_like(picks, -1)).to(torch.int32)


def test_qli_rebase_always_matches_the_sorted_reference() -> None:
    """The reverted helper sorts unconditionally (bitwise the sorted reference)."""
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        picks = torch.randint(-1, 12, (5, 6), generator=generator, dtype=torch.int32)
        visible = torch.randint(0, 6, (5, 1), generator=generator, dtype=torch.int32)
        assert torch.equal(
            csa2._rebase_qli_picks(picks, visible, offset=128),
            _reference_rebase(picks, visible, 128, sort=True),
        )


def test_qli_rebase_ignores_the_removed_no_sort_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reverted env has no effect, and its accessor is gone (revert guard)."""
    picks = torch.tensor([[3, 0, -1, 7, 1]], dtype=torch.int32)
    visible = torch.tensor([[4]], dtype=torch.int32)
    out = csa2._rebase_qli_picks(picks, visible, offset=100)
    assert out.tolist() == [[100, 101, 103, -1, -1]]  # still sorted
    assert not hasattr(csa2, "_qli_picks_no_sort_enabled")


# ---------------------------------------------------------------------------
# 8) QLI v3 prefill extension (default ON; kept A/B switches)
# ---------------------------------------------------------------------------

# Production index-source layers (config ``index_source_layer_ids`` / the arc-19
# C1 plan), reproduced as a coherent plan so the coverage count is pinned: the
# pinned production ``candidate_source_layer_id`` default is -1, so no index
# layer is a candidate-pool layer and all 8 fuse (the A2 question of the plan).
_PREFILL_INDEX_LAYERS = [1, 2, 3, 4, 5, 6, 7, 8]
_PREFILL_N_LAYERS = 9
_PREFILL_RATIOS = [0] + [1] * 8


def _make_prefill_backend(**kwargs: Any) -> Csa2AttentionBackend:
    return _make_backend(
        n_layers=_PREFILL_N_LAYERS,
        compress_ratios=list(_PREFILL_RATIOS),
        kv_source_layer_ids=[1],
        index_source_layer_ids=list(_PREFILL_INDEX_LAYERS),
        **kwargs,
    )


def _make_prefill_ctx(q_lens: list[int]) -> Csa2LayerContext:
    """A context whose query rows are the packed chunk (``sum(q_lens)`` rows)."""
    return _make_ctx(sum(int(length) for length in q_lens))


def _drive_prefill(
    backend: Csa2AttentionBackend,
    layer: _StubIndexerLayer,
    dsa: SimpleNamespace,
    shared: CSA2SharedRuntime,
    q_lens: list[int],
) -> torch.Tensor:
    return backend._run_indexer(layer, _make_prefill_ctx(q_lens), dsa, list(q_lens), dsa.seq_lens, shared)


def _install_topk_counter(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count the pure-torch ``_indexer_topk`` (the eager Sort producer)."""
    counter = {"calls": 0}
    real = csa2._indexer_topk

    def counted(*args: Any, **kwargs: Any) -> torch.Tensor:
        counter["calls"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(csa2, "_indexer_topk", counted)
    return counter


class _PerSeqPicksRecorder(_QliRecorder):
    """Indexer stub that plants a valid pick at every sequence's first row.
    The base recorder only writes into row 0 of the packed query, which cannot express a per-sequence rebase offset; this"""

    def indexer(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        weights: torch.Tensor,
        query_scale: torch.Tensor,
        key_scale: torch.Tensor,
        topk: int,
        quant_mode: int,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _picks, _values, candidate_out = super().indexer(
            query, key, weights, query_scale, key_scale, topk, quant_mode, **kwargs
        )
        cu_seqlens = kwargs["cu_seqlens_q"].tolist()
        picks = torch.full((query.size(0), 1, topk), -1, dtype=torch.int32)
        for seq in range(len(cu_seqlens) - 1):
            picks[cu_seqlens[seq], 0, 0] = 0
            if topk > 1:
                picks[cu_seqlens[seq], 0, 1] = 2
        return picks, torch.empty(0, dtype=torch.bfloat16), candidate_out


def test_qli_v3_prefill_op_missing_restores_eager_indexer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A prefill with the fused op unavailable stays on the historical
    eager path."""
    monkeypatch.setattr(csa2, "_get_qli_v3_ops", lambda: None)
    _reset_log_state(monkeypatch)
    topk_counter = _install_topk_counter(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10])

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, _make_shared(1), [4])

    assert topk_counter["calls"] == 1
    assert topk.shape == (1, 4, 4)
    assert backend._qli_state == {}


def test_qli_v3_prefill_fuses_every_index_layer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ON: prefill => one fused call per index-source layer, no eager Sort."""
    _reset_log_state(monkeypatch)
    topk_counter = _install_topk_counter(monkeypatch)
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_prefill_backend()
    assert backend.candidate_source_layer_id == -1
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5, 8, 12])
    q_lens = [3, 1, 2, 4]
    shared = _make_shared(1)
    ctx = _make_prefill_ctx(q_lens)

    topk = None
    for layer_id in _PREFILL_INDEX_LAYERS:
        topk = backend._run_indexer(_StubIndexerLayer(layer_id, 1), ctx, dsa, q_lens, dsa.seq_lens, shared)

    assert len(recorder.indexer_calls) == len(_PREFILL_INDEX_LAYERS) == 8
    assert len(recorder.metadata_calls) == 8
    assert topk_counter["calls"] == 0
    assert topk is not None and topk.shape == (4, 4, 4)
    assert shared.topk_idxs is topk


def test_qli_v3_prefill_dispatch_relaxes_for_an_eager_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both decode-shape gates open for an eager prefill, and only a prefill."""
    _install_ops(monkeypatch, _QliRecorder())
    backend, layer, graph_dsa, _q_lens, shared = _guard_kwargs(monkeypatch)
    eager_dsa = _make_dsa(is_acl_graph=False)
    prefill_lens = [3, 1, 1, 1]

    # An eager prefill may fuse ...
    assert backend._qli_v3_fallback_reason(layer, eager_dsa, prefill_lens, shared, shared.index_cache) is None
    # ... but an eager decode still may not.
    assert (
        backend._qli_v3_fallback_reason(layer, eager_dsa, [1] * _ROWS, shared, shared.index_cache)
        == "not an ACL-graph decode step"
    )
    # An ACL-graph step never fuses on a chunk query (single-token by contract):
    # the "query length != 1" reason is preserved, order included.
    assert (
        backend._qli_v3_fallback_reason(layer, graph_dsa, prefill_lens, shared, shared.index_cache)
        == "query length != 1"
    )


def test_qli_v3_prefill_multiseq_descriptor_and_rectangular_handoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5, 8, 12])
    q_lens = [3, 1, 2, 4]
    shared = _make_shared(1)

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, shared, q_lens)

    call = recorder.indexer_calls[0]
    assert call["cu_seqlens_q"].tolist() == [0, 3, 4, 6, 10]
    assert call["seqused_k"].tolist() == [10, 5, 8, 12]  # floor(ctx/ratio), ratio 1
    assert call["cu_seqlens_k"] is None
    assert call["cmp_residual_k"] is None
    assert call["max_seqlen_q"] == 4
    assert call["layout_q"] == "TND"
    assert call["layout_k"] == "PA_BBND"
    assert call["mask_mode"] == 3
    assert call["cmp_ratio"] == 1
    assert call["candidate_mode"] == csa2._QLI_V3_CAND_MODE_OFF
    _args, kwargs = recorder.metadata_calls[0]
    assert kwargs["cu_seqlens_q"].tolist() == [0, 3, 4, 6, 10]
    assert kwargs["seqused_k"].tolist() == [10, 5, 8, 12]
    assert kwargs["max_seqlen_q"] == 4
    assert kwargs["batch_size"] == 4
    assert kwargs["max_seqlen_k"] == 12
    # Rectangular hand-off identical in shape to the pure-torch indexer.
    assert topk.shape == (4, 4, 4)
    assert shared.topk_idxs is topk
    assert shared.topk_layer == 2


def test_qli_v3_prefill_rebases_with_the_per_sequence_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _reset_log_state(monkeypatch)
    recorder = _PerSeqPicksRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 12])
    q_lens = [2, 3]
    shared = _make_shared(1)

    topk = _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, shared, q_lens)

    # window = 128; seq 0 has q_len 2 -> offset 130, seq 1 q_len 3 -> offset 131.
    # Row 0 of each sequence: pick 0 is visible, pick 2 is visible only for the
    # sequence whose first query position is >= 1 (visible = (p + 1) // ratio).
    assert topk[0, 0].tolist() == [130, -1, -1, -1]
    assert topk[1, 0].tolist() == [131, 133, -1, -1]
    # Every other row keeps the -1 padding.
    assert int(topk[0, 1].sum()) == -4
    assert int(topk[1, 1:].sum()) == -8


def _record_prefill_rebase_kernels(
    monkeypatch: pytest.MonkeyPatch,
    backend: Csa2AttentionBackend,
    dsa: SimpleNamespace,
    shared: CSA2SharedRuntime,
    q_lens: list[int],
) -> list[bool]:
    """Drive a prefill and return the ``fp32_sort`` flag each rebase was called with."""
    seen: list[bool] = []
    real = csa2._rebase_qli_picks

    def spy(
        picks: torch.Tensor,
        visible: torch.Tensor,
        offset: int,
        *,
        fp32_sort: bool = False,
    ) -> torch.Tensor:
        seen.append(fp32_sort)
        return real(picks, visible, offset, fp32_sort=fp32_sort)

    monkeypatch.setattr(csa2, "_rebase_qli_picks", spy)
    _drive_prefill(backend, _StubIndexerLayer(2, 1), dsa, shared, q_lens)
    return seen


def test_qli_v3_prefill_rebase_default_is_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default ON: the eager prefill rebase runs the FP32 kernel per sequence."""
    _reset_log_state(monkeypatch)
    _install_ops(monkeypatch, _QliRecorder())
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 12])
    q_lens = [2, 3]

    seen = _record_prefill_rebase_kernels(monkeypatch, backend, dsa, _make_shared(1), q_lens)

    assert seen == [True] * len(q_lens)  # one rebase per sequence, all FP32


def test_qli_v3_prefill_covers_all_eight_layers_with_a_candidate_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The candidate pool stays in scope on prefill: 8/8 still fuse.
    Production plan pins ``candidate_source_layer_id`` from the released config and the launcher defaults"""
    _reset_log_state(monkeypatch)
    topk_counter = _install_topk_counter(monkeypatch)
    recorder = _QliRecorder()
    _install_ops(monkeypatch, recorder)
    backend = _make_prefill_backend(candidate_source_layer_id=5)
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[10, 5])
    q_lens = [3, 2]
    shared = _make_shared(1)
    ctx = _make_prefill_ctx(q_lens)

    for layer_id in _PREFILL_INDEX_LAYERS:
        backend._run_indexer(_StubIndexerLayer(layer_id, 1), ctx, dsa, q_lens, dsa.seq_lens, shared)

    assert len(recorder.indexer_calls) == 8
    assert topk_counter["calls"] == 0
    assert [call["candidate_mode"] for call in recorder.indexer_calls] == [
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_OFF,
        csa2._QLI_V3_CAND_MODE_SOURCE,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
        csa2._QLI_V3_CAND_MODE_CONSUMER,
    ]
    assert "candidate_topk_index" not in recorder.indexer_calls[4]
    assert "candidate_topk_index" in recorder.indexer_calls[-1]


def test_qli_v3_prefill_ratio2_uses_the_compressed_capacity_and_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa(is_acl_graph=False, seq_lens=[11, 7])
    q_lens = [2, 2]
    shared = _make_shared(2)

    _drive_prefill(backend, _StubIndexerLayer(1, 2), dsa, shared, q_lens)

    call = recorder.indexer_calls[0]
    assert call["cmp_ratio"] == 2
    # floor(ctx/ratio): [5, 3]; residual ctx % ratio: [1, 1].
    assert call["seqused_k"].tolist() == [5, 3]
    assert call["cmp_residual_k"].tolist() == [1, 1]
    assert call["max_seqlen_q"] == 2
    _args, kwargs = recorder.metadata_calls[0]
    assert kwargs["max_seqlen_k"] == 5
    assert kwargs["seqused_k"].tolist() == [5, 3]


def test_qli_v3_prefill_leaves_decode_graph_steps_on_the_decode_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prefill extension must not reroute an ACL-graph decode step.
    Discriminates the delegation's ``q_len != 1`` clause: an ACL-graph step always has single-token rows, so delegating"""
    layer = _StubIndexerLayer(2, 1)

    def _boom(*args: Any, **kwargs: Any) -> torch.Tensor:
        raise AssertionError("an ACL-graph decode step must not enter the prefill branch")

    monkeypatch.setattr(Csa2AttentionBackend, "_run_indexer_qli_v3_prefill", _boom)
    recorder = _stub_qli(monkeypatch)
    backend = _make_backend()
    topk = _drive(backend, layer, _make_dsa(), _make_shared(1))

    # The decode path built its persistent graph bucket and used the static graph
    # token capacity; the eager prefill branch would do neither.
    assert backend._qli_state
    assert topk.shape[1] == 1  # one query row per sequence on the decode hand-off
    assert recorder.metadata_calls[0][1]["max_seqlen_q"] == 1
    assert recorder.metadata_calls[0][1]["max_seqlen_k"] == _CAPACITY
    assert recorder.indexer_calls[0]["cu_seqlens_q"].tolist() == [0, 1, 2, 3, 4]


def test_qli_refresh_runs_inside_target_after_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_qli(monkeypatch)
    backend = _make_backend()
    dsa = _make_dsa()
    state = backend._qli_state_for(dsa)
    metadata = SimpleNamespace(dsa_metadata=dsa)
    monkeypatch.setattr(csa2.DsaAttentionBackend, "refresh_dsa_metadata_for_graph_replay", lambda *_: None)
    addresses = {ratio: value.data_ptr() for ratio, value in state.seqused_k.items()}
    backend.prepare_graph_forward(metadata)
    assert dsa.qli_refresh_in_target_graph
    for values in ([0, 0, 0, 0], [127, 128, 8193, 8194]):
        previous = state.seqused_k[2].clone()
        dsa.seq_lens.copy_(torch.tensor(values, dtype=torch.int32))
        backend.refresh_dsa_metadata_for_graph_replay(metadata)
        assert torch.equal(state.seqused_k[2], previous)
        backend.prepare_graph_forward(metadata)
        for ratio, value in state.seqused_k.items():
            assert value.data_ptr() == addresses[ratio]
            assert torch.equal(value, dsa.seq_lens // ratio)
            assert torch.equal(state.cmp_residual[ratio], dsa.seq_lens % ratio)
