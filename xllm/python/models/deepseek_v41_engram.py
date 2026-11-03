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

"""DeepSeek-V4.1 Engram: n-gram hash lookups gated into the mHC residual stream.

Port of the reference ``inference/engram.py`` (vLLM
``vllm/models/deepseek_v4_1/common/engram.py`` is the upstream mirror).
Engram modules live on the backbone layers listed in ``engram_layer_ids``
only; the gate is a normalized dot product of the residual stream against a
per-hc-copy key, signed-sqrt'ed before the sigmoid:

    out[t, j] = h[t, j] + sigmoid(signed_sqrt(|dot|)) * value[t]

Three pieces of derived state are NOT in the checkpoint and are rebuilt at
init (deterministically, exactly as the reference does):

* ``token_map``: token id -> compressed vocab id (normalization-folded).
  Loaded from the ``engram_hash_state.bin`` sidecar
  (:func:`build_engram_hash_state` builds it) or injected for tests.
* ``primes`` / ``offsets``: per-(layer, n-gram size, head) prime-sized
  bucket ranges, drawn in order from ``engram_vocab_size - 1`` upward and
  never reused across layers.
* ``multipliers``: one odd int64 per (layer, lookback shift) from the
  per-layer RNG seeded ``10007 * layer_id`` (numpy PCG64, matching the
  reference bit-for-bit).

The hash computation keeps a slot-keyed rolling store (``hash_cache``) of
compressed ids, one entry per SWA KV slot of the first engram layer: slots
are stable per (request, position), prefix-cache hits reuse both the
physical blocks and the identical token ids, and spec-decode rollbacks
rewrite the same slots, so lookbacks read back exactly what the owning
request wrote. The runner may pass ``lookback_token_ids`` (ids just before
each request's chunk start), which take precedence over the slot cache.

Everything is plain torch with static shapes and no host synchronization,
so the whole module is ACL-graph safe.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
import weakref
from typing import Any, Callable, Sequence

import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import last_ring_writes
from xllm.python.model_executor.forward_context import in_acl_graph
from xllm.python.models.engram_shared_memory import SharedHostEngramStorage

# Cache value for tokens that take no part in an n-gram (image spans).
ENGRAM_DEAD_ID = -1
# Image sentinel / padding token ids that break an n-gram (mm_preprocess.py).
ENGRAM_IMAGE_SENTINEL_BASE_ID = 129264
ENGRAM_IMAGE_PAD_ID = 129265


# Signed-sqrt clamp of the gate input (reference kernel clamp_value).
_ENGRAM_GATE_CLAMP = 1e-6

_WHITESPACE_RE = re.compile(r"[ \t\r\n]+")
# Private-use char so a token that is exactly one space survives Strip().
_SENTINEL = "\ue000"


def image_sentinel_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """True where the token is an image sentinel and must break every n-gram."""
    ids = input_ids.reshape(-1)
    return (ids == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (ids == ENGRAM_IMAGE_PAD_ID)


# ---------------------------------------------------------------------------
# Prime bucket layout
# ---------------------------------------------------------------------------


def _is_prime(n: int) -> bool:
    """Deterministic Miller-Rabin for n < 2**32 (same witnesses as vLLM)."""
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 7, 61):
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _find_next_prime(start: int, seen_primes: set[int]) -> int:
    """The smallest prime above ``start`` that has not been handed out yet."""
    candidate = start + 1
    while not _is_prime(candidate) or candidate in seen_primes:
        candidate += 1
    return candidate


class EngramLayout:
    """Bucket layout of the per-layer n-gram hash tables.

    A position is hashed as ``max_ngram_size - 1`` n-grams (2-gram .. max),
    each split over ``n_heads`` heads. Every (n-gram size, head) pair owns a
    prime-sized bucket range in the layer's table; the primes are drawn in
    order and never reused, keeping the ranges disjoint.
    """

    def __init__(
        self,
        layer_ids: Sequence[int],
        num_embeddings: Sequence[int],
        max_ngram_size: int,
        n_heads: int,
        head_dim: int,
        vocab_size: int,
        compressed_vocab_size: int,
        pad_token_id: int,
    ) -> None:
        self.layer_ids: tuple[int, ...] = tuple(int(i) for i in layer_ids)
        self.num_embeddings: tuple[int, ...] = tuple(int(n) for n in num_embeddings)
        self.max_ngram_size = int(max_ngram_size)
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.vocab_size = int(vocab_size)
        self.compressed_vocab_size = int(compressed_vocab_size)
        self.pad_token_id = int(pad_token_id)
        if len(self.layer_ids) != len(self.num_embeddings):
            raise ValueError(
                f"engram_layer_ids {self.layer_ids} and engram_num_embeddings "
                f"{self.num_embeddings} must have the same length"
            )
        if self.max_ngram_size < 2:
            raise ValueError(f"engram_max_ngram_size must be >= 2, got {max_ngram_size}")

        primes: list[list[tuple[int, ...]]] = []
        seen: set[int] = set()
        for _ in self.layer_ids:
            per_ngram = []
            for _ in range(self.max_ngram_size - 1):
                sizes = []
                current = self.vocab_size - 1
                for _ in range(self.n_heads):
                    current = _find_next_prime(current, seen)
                    seen.add(current)
                    sizes.append(current)
                per_ngram.append(tuple(sizes))
            primes.append(per_ngram)
        self.primes: tuple[tuple[tuple[int, ...], ...], ...] = tuple(tuple(p) for p in primes)
        self.n_hash_cols = (self.max_ngram_size - 1) * self.n_heads
        for layer, layer_primes in enumerate(self.primes):
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            total = sum(flat)
            if total > self.num_embeddings[layer]:
                raise ValueError(
                    f"engram layer {self.layer_ids[layer]} prime buckets sum to {total} rows "
                    f"but engram_num_embeddings only provides {self.num_embeddings[layer]}"
                )
        offsets = []
        for layer_primes in primes:
            flat = [size for per_ngram in layer_primes for size in per_ngram]
            prefix = [0]
            for size in flat[:-1]:
                prefix.append(prefix[-1] + size)
            offsets.append(prefix)
        self.offsets = torch.tensor(offsets, dtype=torch.int64)

    def head_sizes(self, layer_hash_index: int) -> tuple[int, ...]:
        return tuple(size for per_ngram in self.primes[layer_hash_index] for size in per_ngram)


def compute_hash_multipliers(
    layer_ids: Sequence[int],
    max_ngram_size: int,
    compressed_vocab_size: int,
) -> torch.Tensor:
    """One odd int64 multiplier per (layer, lookback shift).

    Must match the reference bit-for-bit: the multipliers were drawn with
    ``numpy.random.default_rng(10007 * layer_id).integers(0, bound, size=(n,))``
    at training time, so the exact PCG64 stream is part of the checkpoint
    contract. Values are doubled-plus-one (odd) and bounded so that
    ``compressed_id * multiplier`` cannot overflow int64.
    """
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "Building DeepSeek-V4.1 engram hash multipliers requires numpy (the "
            "reference RNG is numpy PCG64 and must be reproduced bit-exactly). "
            "Install numpy, or pre-generate an 'engram_hash_state.bin' sidecar "
            "with build_engram_hash_state() and place it next to "
            "the checkpoint."
        ) from exc
    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(
            low=0,
            high=multiplier_bound,
            size=(max_ngram_size,),
            dtype=np.int64,
        )
        rows.append(torch.tensor(values * 2 + 1, dtype=torch.int64))
    return torch.stack(rows)


# ---------------------------------------------------------------------------
# Token map (tokenizer-derived, sidecar-loaded)
# ---------------------------------------------------------------------------


def normalize_token_text(text: str) -> str:
    """The reference normalization chain (engram.py ``build_compressed_token_map``).

    NFKC -> NFD -> strip accents -> lowercase -> fold whitespace runs to one
    space -> protect a lone space with a sentinel -> strip -> restore the
    space. Tokens that normalize alike collapse onto one compressed id.
    """
    s = unicodedata.normalize("NFKC", text)
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = _WHITESPACE_RE.sub(" ", s)
    if s == " ":
        s = _SENTINEL
    s = s.strip()
    return s.replace(_SENTINEL, " ")


def build_compressed_token_map_from_texts(
    token_texts: Sequence[str],
    id_to_token: Callable[[int], str] | None = None,
) -> tuple[list[int], int]:
    """Fold token texts onto compressed ids (first-seen order).

    ``token_texts[i]`` is the decoded text of token id ``i``. Tokens whose
    decoded text contains U+FFFD (partial UTF-8 byte tokens) are keyed by
    their raw vocabulary form via ``id_to_token``.
    """
    key_to_new: dict[str, int] = {}
    lookup: list[int] = []
    for token_id, text in enumerate(token_texts):
        if "\ufffd" in text:
            key = id_to_token(token_id) if id_to_token is not None else text
        else:
            normalized = normalize_token_text(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup.append(new_id)
    return lookup, len(key_to_new)


def _load_engram_hash_state_sidecar(model_path: str) -> dict[str, torch.Tensor] | None:
    """Load the optional ``engram_hash_state.bin`` sidecar.

    Holds the tokenizer-derived state (``token_map`` int32 [vocab],
    ``multipliers`` int64 [n_layers, max_ngram]) so serving environments
    without numpy/tokenizers stay self-contained.
    """
    import os

    sidecar = os.path.join(model_path, "engram_hash_state.bin")
    if not os.path.isfile(sidecar):
        return None
    try:
        return torch.load(sidecar, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(f"failed to load engram hash-state sidecar {sidecar}: {exc}") from exc


def build_engram_hash_state(
    model_path: str,
    tokenizer_path: str | None = None,
    output_path: str | None = None,
) -> str:
    """Generate the ``engram_hash_state.bin`` sidecar next to a checkpoint.

    Writes the tokenizer-derived state :class:`EngramHashState` needs but that
    is not stored in the checkpoint: the compressed token map (every token id
    decoded and folded through :func:`normalize_token_text`) and the numpy
    PCG64 hash multipliers (seed ``10007 * layer_id``). Both are rebuildable
    deterministically from ``config.json`` + ``tokenizer.json``, but the
    multipliers reproduce a specific RNG stream, so they are baked once here
    instead of rebuilt on every model load (the serving path only needs
    ``torch``, not ``numpy``/``tokenizers``).

    Requires the optional ``tokenizers`` and ``numpy`` packages. Refuses to
    write a sidecar whose compressed count diverges from the config's
    ``engram_compressed_vocab_size``: every hash multiplier derives from that
    count, so a mismatch would silently rehash the engram tables.

    Returns the path written.
    """
    model_dir = os.path.abspath(model_path)
    if not os.path.isdir(model_dir):
        model_dir = os.path.dirname(model_dir)
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        top = json.load(handle)
    config = {**top, **top.get("text_config", {})}

    layer_ids = config.get("engram_layer_ids") or []
    if not layer_ids:
        raise ValueError("config.json has no engram_layer_ids; this model does not use engram")
    vocab_size = int(config["vocab_size"])
    max_ngram_size = int(config["engram_max_ngram_size"])
    compressed_vocab = int(config["engram_compressed_vocab_size"])
    pad_token_id = int(config["engram_pad_token_id"])
    if not 0 <= pad_token_id < vocab_size:
        raise ValueError(f"engram_pad_token_id {pad_token_id} is out of vocab range {vocab_size}")

    try:
        from tokenizers import Tokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            "building the engram token map requires the 'tokenizers' package (pip install tokenizers)"
        ) from exc
    tokenizer_file = tokenizer_path or os.path.join(model_dir, "tokenizer.json")
    if not os.path.isfile(tokenizer_file):
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_file}")
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    tokenizer_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    if tokenizer_vocab != vocab_size:
        raise ValueError(f"tokenizer vocab size {tokenizer_vocab} does not match config vocab_size {vocab_size}")

    token_texts = tokenizer.decode_batch([[token_id] for token_id in range(vocab_size)])
    lookup, compressed = build_compressed_token_map_from_texts(token_texts, id_to_token=tokenizer.id_to_token)
    if compressed != compressed_vocab:
        raise ValueError(
            f"the tokenizer folds to {compressed} compressed ids but the config declares "
            f"engram_compressed_vocab_size={compressed_vocab}; every hash multiplier derives "
            "from this count, so the engram tables would be silently rehashed -- refusing to "
            "write the sidecar"
        )

    multipliers = compute_hash_multipliers(layer_ids, max_ngram_size, compressed_vocab)
    output = output_path or os.path.join(model_dir, "engram_hash_state.bin")
    torch.save(
        {
            "token_map": torch.tensor(lookup, dtype=torch.int32),
            "multipliers": multipliers,
        },
        output,
    )
    return output


# ---------------------------------------------------------------------------
# Hash state (per model)
# ---------------------------------------------------------------------------


class EngramHashState(nn.Module):
    """Maps each position to the hash ids of the n-grams ending there.

    Stateless pieces (``token_map`` / ``primes`` / ``offsets`` /
    ``multipliers``) are built once at init; the slot-keyed ``hash_cache``
    is sized lazily from the bound SWA KV cache and is discarded on cache
    rebind (graph memory profiling swaps in a temporary, smaller cache).
    """

    def __init__(
        self,
        layout: EngramLayout,
        token_map: torch.Tensor,
        multipliers: torch.Tensor,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.lookback_depth = layout.max_ngram_size - 1
        self.pad_id = int(token_map[layout.pad_token_id].item())
        self.register_buffer("token_map", token_map.to(torch.int64), persistent=False)
        self.register_buffer(
            "primes",
            torch.tensor(layout.primes, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer("offsets", layout.offsets.to(torch.int64), persistent=False)
        self.register_buffer("multipliers", multipliers.to(torch.int64), persistent=False)
        self._cache: torch.Tensor | None = None
        self._kv_cache_ref: weakref.ReferenceType[torch.Tensor] | None = None

    @classmethod
    def from_config(cls, layout: EngramLayout, config: dict) -> EngramHashState:
        """Build from the model config dict, resolving the token map source.

        Order: explicit ``engram_token_map`` injection (tests) -> the
        ``engram_hash_state.bin`` sidecar next to the checkpoint -> a hard
        error telling the operator to generate the sidecar.
        """
        injected = config.get("engram_token_map")
        if injected is not None:
            token_map = torch.as_tensor(injected, dtype=torch.int64)
            multipliers = compute_hash_multipliers(
                layout.layer_ids, layout.max_ngram_size, layout.compressed_vocab_size
            )
        else:
            sidecar = _load_engram_hash_state_sidecar(str(config.get("model_path") or ""))
            if sidecar is None:
                raise RuntimeError(
                    "DeepSeek-V4.1 engram requires the tokenizer-derived hash state. "
                    "Call build_engram_hash_state(model_path) against the "
                    "checkpoint's tokenizer to write 'engram_hash_state.bin' (or pass "
                    "'engram_token_map' in the config for tests)."
                )
            token_map = sidecar["token_map"]
            multipliers = sidecar["multipliers"]
        # layout.vocab_size is the engram hash-bucket space (engram_vocab_size,
        # e.g. 16M), NOT the tokenizer vocab; validate the map against the
        # model's vocab_size when the caller provides it.
        expected_vocab = int(config.get("vocab_size") or token_map.numel())
        if token_map.numel() != expected_vocab:
            raise ValueError(
                f"engram token map covers {token_map.numel()} ids but model vocab_size is {expected_vocab}"
            )
        compressed = int(token_map.max().item()) + 1
        if compressed != layout.compressed_vocab_size:
            raise ValueError(
                f"engram token map compresses to {compressed} ids but config expects "
                f"{layout.compressed_vocab_size}; every hash multiplier derives from "
                "it, so the engram tables would be silently rehashed"
            )
        return cls(layout, token_map, multipliers)

    def ensure_cache(self, swa_cache: torch.Tensor, block_size: int) -> bool:
        """Lazily size the slot cache from the bound SWA KV cache.

        Returns False while the KV cache is unbound (profile run); the
        caller skips engram hashing then.
        """
        if swa_cache is None or swa_cache.numel() == 0:
            self._cache = None
            self._kv_cache_ref = None
            return False
        if self._kv_cache_ref is not None and self._kv_cache_ref() is swa_cache:
            return True
        flat_rows = swa_cache.numel() // swa_cache.size(-1) if swa_cache.dim() > 1 else swa_cache.numel()
        self._cache = torch.zeros(flat_rows, dtype=torch.int32, device=swa_cache.device)
        self._kv_cache_ref = weakref.ref(swa_cache)
        del block_size
        return True

    def restore_history(
        self,
        history: torch.Tensor,
        *,
        non_blocking: bool = False,
    ) -> None:
        """Seed prompt-tail [receiver-local slot, token] pairs before decode."""
        if self._cache is None:
            raise RuntimeError("Engram history restore requires a bound cache")
        if history.ndim != 2 or history.shape[1] != 2:
            raise ValueError("Engram history must contain [slot, token] pairs")
        deduplicate = not in_acl_graph()
        if non_blocking and history.device.type == "cpu" and self._cache.device.type == "npu":
            # Resolve duplicate receiver slots while the history is still on
            # the host. Device boolean indexing would synchronize the draft
            # even after an asynchronous upload of these fixed history rows.
            slots, tokens = last_ring_writes(history[:, 0], history[:, 1])
            history = torch.stack((slots, tokens), dim=1).to(torch.int64)
            history = history.pin_memory(device=self._cache.device.type)
            deduplicate = False
        history = history.to(device=self._cache.device, dtype=torch.int64, non_blocking=non_blocking)
        slots, tokens = history.unbind(dim=1)
        comp = self.token_map[tokens.clamp(0, self.token_map.numel() - 1)]
        comp = torch.where(
            image_sentinel_mask(tokens).reshape(-1),
            ENGRAM_DEAD_ID,
            comp,
        ).to(torch.int64)
        self._write_hash_cache(
            comp,
            slots,
            deduplicate=deduplicate,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
        dead_mask: torch.Tensor,
        lookback_token_ids: torch.Tensor | None,
        slot_mapping: torch.Tensor | None,
        block_table: torch.Tensor | None,
        swa_block_size: int,
        *,
        max_chunk_len: int | None = None,
    ) -> torch.Tensor:
        """Compute ``[T, n_engram_layers, n_hash_cols]`` int32 n-gram hashes.

        History resolution per (token, shift): the current chunk first, then
        ``lookback_token_ids`` (runner-provided, -1 = unknown), then the
        slot cache. Once a lookback is out of range or dead, every further
        shift pools ``pad_id`` into the rolling hash (the pad rows are
        trained, not skipped).
        """
        num_tokens = input_ids.shape[0]
        num_layers = len(self.layout.layer_ids)
        n_heads = self.layout.n_heads
        max_ngram = self.layout.max_ngram_size
        device = input_ids.device
        output = torch.zeros(num_tokens, num_layers, self.layout.n_hash_cols, dtype=torch.int32, device=device)
        if num_tokens == 0:
            return output
        pos = positions.reshape(-1).to(torch.int64)
        ids = input_ids.reshape(-1).to(torch.int64)
        req = self._sequence_index(pos, query_start_loc)
        chunk_start = pos[query_start_loc.to(torch.int64)[req].clamp(0, num_tokens - 1)]

        comp = self.token_map[ids.clamp(0, self.token_map.numel() - 1)]
        comp = torch.where(dead_mask.reshape(-1), ENGRAM_DEAD_ID, comp).to(torch.int64)
        # Expanded verify presents each candidate as a one-token sequence, so
        # later candidates must see earlier ones through the slot cache. A
        # chunk longer than the ring must not overwrite its own pre-chunk
        # history, so publish before hashing only chunks that fit
        # (length + lookback depth <= ring); the rest publish after the loop.
        early_slots = None
        if slot_mapping is not None and self._cache is not None:
            q_cu = query_start_loc.to(torch.int64).reshape(-1)
            q_lens = q_cu[1:] - q_cu[:-1]
            fits_ring = q_lens[req] + self.lookback_depth <= swa_block_size
            slots = slot_mapping.reshape(-1).to(torch.int64)
            early_slots = torch.where(fits_ring, slots, torch.full_like(slots, -1))
            self._write_hash_cache(
                comp,
                early_slots,
                deduplicate=not in_acl_graph(),
            )

        rows = torch.arange(num_tokens, device=device)
        rolling = torch.zeros(num_tokens, num_layers, dtype=torch.int64, device=device)
        blocked = torch.zeros(num_tokens, dtype=torch.bool, device=device)
        cache = self._cache
        for shift in range(max_ngram):
            lookback = pos - shift
            in_batch = lookback >= chunk_start
            source = self._chunk_source(comp, rows, shift, in_batch, num_tokens)
            # ``known`` marks positions whose explicit history resolved to a
            # real token id; the slot cache must not overwrite them. None means
            # no explicit history was passed, so the cache stays the only
            # history source (unchanged behavior).
            known = None
            if lookback_token_ids is not None:
                source, known = self._lookback_source(
                    source, comp, req, lookback, in_batch, chunk_start, lookback_token_ids
                )
            if cache is not None and block_table is not None:
                source = self._cache_source(source, req, lookback, in_batch, known, block_table, swa_block_size)
            blocked = blocked | (lookback < 0) | (source == ENGRAM_DEAD_ID)
            value = torch.where(blocked, self.pad_id, source)
            rolling ^= value.unsqueeze(1) * self.multipliers[:, shift].unsqueeze(0)
            if shift > 0:
                lo = (shift - 1) * n_heads
                prime = self.primes[:, shift - 1, :]  # [L, n_heads]
                offset = self.offsets[:, lo : lo + n_heads]  # [L, n_heads]
                hashed = rolling.unsqueeze(-1) % prime.unsqueeze(0) + offset.unsqueeze(0)
                output[:, :, lo : lo + n_heads] = hashed.to(torch.int32)
        # The first n-grams in a chunk may still read the previous ring tail.
        # Publish the chunks that do not fit the ring only after those
        # lookbacks have been consumed; the others were published up front so
        # expanded-verify candidates could read each other.
        # ``max_chunk_len`` is the host-side longest chunk
        # (``dsa.max_query_len``); it uses the same criterion as the per-token
        # ``fits_ring`` mask above without a device-to-host sync.
        chunk_len_bound = num_tokens if max_chunk_len is None else max_chunk_len
        all_chunks_fit_ring = chunk_len_bound + self.lookback_depth <= swa_block_size
        if early_slots is not None and not all_chunks_fit_ring:
            slots = slot_mapping.reshape(-1).to(torch.int64)
            late_slots = torch.where(early_slots >= 0, torch.full_like(slots, -1), slots)
            self._write_hash_cache(
                comp,
                late_slots,
                # Eager chunks may revisit a physical ring slot, including
                # mixed batches whose empty rows hide that fact from aggregate
                # token counts. Graph decode/verify uses fixed row-aligned
                # slots: active writes must be distinct; padded rows use -1.
                deduplicate=not in_acl_graph(),
            )
        return output

    def _write_hash_cache(
        self,
        comp: torch.Tensor,
        slot_mapping: torch.Tensor,
        *,
        deduplicate: bool = False,
    ) -> None:
        """Publish resolved ids; graph callers supply distinct nonnegative slots.

        Repeated padding slots are allowed: they write the same preserved
        value to reserved slot 0. Arbitrary conflicting active slots are not
        a supported graph input; eager chunks select their last valid writes.
        """
        assert self._cache is not None
        # Reuse the resolved ids, including clamping and dead-token masking.
        comp = comp.reshape(-1).to(torch.int32)
        slots = slot_mapping.reshape(-1).to(torch.int64)
        if deduplicate:
            slots, comp = last_ring_writes(slots, comp)
        valid = slots >= 0
        safe = torch.where(valid, slots, torch.zeros_like(slots))
        keep = self._cache.index_select(0, safe)
        self._cache.scatter_(0, safe, torch.where(valid, comp, keep))

    def _sequence_index(
        self,
        positions: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token request index via the cumulative chunk boundaries."""
        num_tokens = positions.shape[0]
        qsl = query_start_loc.to(torch.int64).reshape(-1)
        ends = qsl[1:].contiguous()
        req = torch.searchsorted(ends, torch.arange(num_tokens, device=positions.device), right=True)
        return req.clamp(0, max(qsl.numel() - 2, 0))

    def _chunk_source(
        self,
        comp: torch.Tensor,
        rows: torch.Tensor,
        shift: int,
        in_batch: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Compressed id of ``rows - shift`` when it stays inside the chunk."""
        source_row = (rows - shift).clamp_min(0).clamp_max(num_tokens - 1)
        source = comp.index_select(0, source_row)
        return torch.where(in_batch, source, torch.zeros_like(source))

    def _lookback_source(
        self,
        source: torch.Tensor,
        comp: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        chunk_start: torch.Tensor,
        lookback_token_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Explicit-history id where the runner provided one, plus its mask.

        Returns ``(source, known)``: ``known`` is True exactly where the
        position is outside the current chunk and the runner supplied a real
        token id (``>= 0``). Callers use ``known`` to keep explicit history
        ahead of the slot cache.
        """
        depth = lookback_token_ids.shape[-1]
        col = chunk_start - 1 - lookback
        in_window = (~in_batch) & (col >= 0) & (col < depth)
        col_safe = col.clamp(0, depth - 1)
        req_safe = req.clamp(0, lookback_token_ids.shape[0] - 1)
        token = lookback_token_ids.to(torch.int64)[req_safe, col_safe]
        known = in_window & (token >= 0)
        token_safe = token.clamp_min(0)
        mapped = self.token_map[token_safe.clamp(0, self.token_map.numel() - 1)]
        dead = (token_safe == ENGRAM_IMAGE_SENTINEL_BASE_ID) | (token_safe == ENGRAM_IMAGE_PAD_ID)
        mapped = torch.where(dead, ENGRAM_DEAD_ID, mapped)
        return torch.where(known, mapped, source), known

    def _cache_source(
        self,
        source: torch.Tensor,
        req: torch.Tensor,
        lookback: torch.Tensor,
        in_batch: torch.Tensor,
        known: torch.Tensor | None,
        block_table: torch.Tensor,
        swa_block_size: int,
    ) -> torch.Tensor:
        """Slot-cache id, applied only where neither the chunk nor explicit history resolved.

        ``known`` is the ``_lookback_source`` mask (None when no explicit
        history was passed) and takes precedence over the cache, matching the
        documented resolution order: chunk -> explicit lookback -> slot cache.
        """
        assert self._cache is not None
        table_cols = block_table.shape[1]
        # Lookbacks are absolute token positions; clamp only negative padding.
        # An upper clamp would alias later positions to the final table column.
        # Logical SWA tables retain placeholders, so modulo is normally a no-op.
        # Bounded rings must match the C++ writer's rotation and table width.
        clamped = lookback.clamp_min(0)
        col = torch.div(clamped, swa_block_size, rounding_mode="floor") % table_cols
        req_safe = req.clamp(0, block_table.shape[0] - 1)
        block = block_table.to(torch.int64)[req_safe, col]
        missing = block < 0
        safe_block = torch.where(missing, torch.zeros_like(block), block)
        slot = (safe_block * swa_block_size + clamped % swa_block_size).clamp(0, self._cache.numel() - 1)
        fallback = self._cache.index_select(0, slot).to(torch.int64)
        fallback = torch.where(
            missing,
            torch.full_like(fallback, ENGRAM_DEAD_ID),
            fallback,
        )
        resolved = in_batch if known is None else (in_batch | known)
        return torch.where(resolved, source, fallback)


# ---------------------------------------------------------------------------
# Hash table embedding
# ---------------------------------------------------------------------------


# Table storage layouts: the official FP8 checkpoint keeps FP8-E4M3 payloads
# plus ue8m0 per-32 scales as raw bytes; the Eco-Tech ascend (W8A8) export
# requantizes the tables to symmetric int8 with float32 group-32 scales
# (quant_model_description.json optional.embedding_storage, format int8_sym).
ENGRAM_STORAGE_FP8 = "fp8"
ENGRAM_STORAGE_INT8 = "int8"

# Host offloading the int8 tables (W8A8 export) keeps their ~27.7GB/rank
# resident HBM in host memory instead, at the cost of a per-step NPU->CPU id
# sync plus a CPU dequant + H2D of the staged rows. Selected through
# --engram-config (engram_config.cpu_offload); default off so the HBM path
# stays byte-identical.

# Floor for the pinned host staging buffers; only grown (never shrunk) so the
# device staging buffer keeps one stable address for ACL-graph capture.
_ENGRAM_HOST_MIN_CAPACITY = 256


def engram_offload_warning_fields(model: Any) -> tuple[int, list[Any]]:
    """(engram layer count, graph-capacity hints) for the engram-skip warning.

    Feeds the runner-side staging diagnostic
    (``_warn_engram_stage_incomplete``) so the fields it logs cannot drift apart
    from the model. Reads the model generically (``layers`` may be absent on
    test stubs).
    """
    engram_layers = [
        engram
        for layer in getattr(model, "layers", None) or []
        if (engram := getattr(layer, "engram", None)) is not None
    ]
    hints = [getattr(getattr(engram, "embed_tokens", None), "_graph_capacity_hint", None) for engram in engram_layers]
    return len(engram_layers), hints


# Process-level best-effort log-dedup latch: ``_host_buffer`` warns on the
# first pin failure of an episode and a successful pin re-arms it. It carries no
# pinned/buffer *correctness* -- that lives per embedding instance in
# ``EngramEmbedding._ids_host_pinned`` / ``_rows_host_pinned`` plus the failure
# cause ``EngramEmbedding._host_pin_failure`` (a module-global cause would be
# overwritten by a sibling layer's allocation and misreported by
# :meth:`EngramEmbedding._pin_failure_hint`). It is intentionally unlocked: a
# lost update at worst logs one extra warning, and the per-instance flags above
# are what the lookup-op contract reads.
_host_pin_warned = False


def _host_buffer(shape: tuple[int, ...], dtype: torch.dtype) -> tuple[torch.Tensor, bool, str | None]:
    """A CPU staging buffer, whether it is pinned, and the pin failure cause.

    ``pin_memory`` needs a registered accelerator (CUDA/NPU); a host without one
    raises, and this returns the cause (``None`` on success) so the caller can
    record it on the owning embedding. The failure is never silent: the first
    failure of an episode is warned here (a later allocation that pins again
    re-arms the warning, so a host that can never pin logs exactly once), and it
    remains a hard error at the first lookup, where the op contract refuses
    unpinned staging and names the real cause
    (:meth:`EngramEmbedding._engram_lookup_op_contract_error`). The returned
    ``pinned`` bool is the allocation-time outcome, which the caller stores so
    the op contract can rely on it instead of re-probing ``Tensor.is_pinned()``
    later (that probe can misreport on hosts/PyTorch versions without a
    registered accelerator).
    """
    global _host_pin_warned
    try:
        buffer = torch.empty(shape, dtype=dtype, device="cpu", pin_memory=True)
    except (RuntimeError, ValueError) as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if not _host_pin_warned:
            _host_pin_warned = True
            try:
                from scripts.logger import logger

                logger.warning(
                    "engram host offload cannot allocate pinned CPU staging "
                    f"({failure}); the lookup op contract refuses "
                    "unpinned staging, so set engram_config.cpu_offload=false on "
                    "a host that cannot pin host memory"
                )
            except Exception:  # noqa: BLE001 - logging must never mask the fallback
                pass
        return torch.empty(shape, dtype=dtype, device="cpu"), False, failure
    _host_pin_warned = False
    return buffer, True, None


def _sync_accelerator(device: torch.device | None) -> None:
    """Block until all queued accelerator work on this device has completed.

    Deliberately a whole-device synchronize, not a per-stream one: this module
    never creates -- nor keeps a handle to -- a dedicated copy stream, and the
    copies this drains are issued by plain ``Tensor.copy_`` on the ambient
    accelerator stream, so there is no narrower stream to target. Two callers
    rely on the drain: ``_local_block_offload`` must make the id/owned D2H
    visible to the CPU gather that follows, and ``_ensure_host_scratch`` must
    not free a pinned buffer that a still-in-flight non-blocking H2D (the
    staging upload at the end of the previous offload pass) could read. A
    device-wide synchronize is the portable guarantee for both; the alternative
    -- a per-copy stream -- would require threading an explicit copy stream
    through every ``copy_`` here, which the offload path does not use.

    A whole-device synchronize is safe here because this runs only on the
    single-threaded per-rank Python path -- the load-time staging setup and the
    per-step pre-graph host staging pass, never inside an already-captured graph
    -- and the module creates no bypass copy stream, so every drained ``copy_``
    lands on the ambient accelerator stream. There is no other stream running
    concurrently with the one being drained, so the device-wide wait cannot
    deadlock against a stream that is itself waiting on the current stream.
    A per-stream synchronize would only be possible if each ``copy_`` here were
    explicitly threaded through a dedicated copy stream (which this offload path
    does not use), so the whole-device synchronize is deliberate.
    """
    if device is None or device.type == "cpu":
        return
    if device.type == "npu":
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    else:
        # Unknown/other device families: best-effort whole-device synchronize
        # through the device module's own ``synchronize`` if it exposes one; a
        # device type without it is skipped (there is no narrower fallback).
        synchronize = getattr(torch, device.type, None)
        synchronize = getattr(synchronize, "synchronize", None)
        if synchronize is not None:
            synchronize()


# --- CPU int8 lookup op (the default and only offload dequant path) ---------
#
# The host-offload dequant (index_select + repeat_interleave group scale
# + bf16 cast) would otherwise run on the host channel every layer every step.
# The in-repo ``xllm_ops.engram_int8_lookup_cpu`` kernel (built from
# ``core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp``) fuses it into one CPU
# call writing directly into the pinned staging buffer, and is now the only
# dequant path for the offloaded table. (The table is pageable host memory; only
# the staging/scratch buffers are pinned.) It is not graph-safe host work, so
# this stays inside ``_local_block_offload`` (the pre-graph host staging pass)
# and MUST NOT move into ACL-graph capture.

_engram_lookup_op: Any = None


def _get_engram_lookup_op() -> Any:
    """Resolve the in-repo CPU lookup op; raise when it is not registered.

    The op is compiled into the xLLM binary from
    ``core/kernels/npu/xllm_ops/engram_int8_lookup_cpu.cpp`` and registered under
    ``torch.ops.xllm_ops`` on the CPU device key. It is the one and only
    dequantizer for the host-offloaded table, so a missing registration is a
    loud failure (rebuild/reinstall the xLLM binary) rather than a silent
    fallback to an eager host path.
    """
    global _engram_lookup_op
    if _engram_lookup_op is not None:
        return _engram_lookup_op
    try:
        op = torch.ops.xllm_ops.engram_int8_lookup_cpu
    except (AttributeError, RuntimeError) as exc:
        raise RuntimeError(
            "engram host offload requires the in-repo CPU int8 lookup op "
            "torch.ops.xllm_ops.engram_int8_lookup_cpu, which is not registered; "
            "rebuild/reinstall xLLM with core/kernels/npu/xllm_ops/"
            "engram_int8_lookup_cpu.cpp compiled in (the fused op is the only "
            "offload dequant path)"
        ) from exc
    _engram_lookup_op = op
    return op


_engram_lookup_op_active_logged = False


def _log_engram_lookup_op_active() -> None:
    """One-shot info when the CPU lookup op path first engages."""
    global _engram_lookup_op_active_logged
    if _engram_lookup_op_active_logged:
        return
    _engram_lookup_op_active_logged = True
    try:
        from scripts.logger import logger

        logger.info("engram CPU int8 lookup op active (fused offload dequant)")
    except Exception:  # noqa: BLE001 - logging must never block the lookup
        pass


def load_quarot_gate_unrotate(model_path: str, device: torch.device) -> torch.Tensor | None:
    """The QuaRot global rotation transpose used to un-rotate the gate input.

    The Eco-Tech W8A8 export rotates the residual stream (h_rot = h @ Q) and
    folds the rotation into every trunk weight, so the stream reaches the
    engram gate in the rotated basis while q/k/wkv-K stay in the original
    basis. The gate dot therefore needs ``h_rot @ Q.T``; the V half of ``wkv``
    is pre-folded (Q.T @ Wv) and must NOT be rotated again. Returns ``Q.T``
    ([hidden, hidden] fp32) or None when the checkpoint is not rotated. The
    safetensors payload may be saved as F64/F32/F16/BF16; the dtype is read
    from the file header and the result is converted to fp32 on return.
    """
    import json
    import os
    import struct

    config_path = os.path.join(str(model_path), "config.json")
    try:
        with open(config_path, encoding="utf-8") as handle:
            rotation = json.load(handle).get("engram_rotation_config") or {}
    except OSError:
        return None
    if not rotation.get("value_projection_rotated"):
        return None
    quarot_path = os.path.join(str(model_path), "optional", "quarot.safetensors")
    try:
        from safetensors.torch import load_file

        q = load_file(quarot_path, device="cpu")["global_rotation"]
    except ImportError:
        # Manual header parse when the safetensors package is unavailable;
        # real load failures from load_file must propagate.
        dtype_map = {
            "F64": (torch.float64, 8),
            "F32": (torch.float32, 4),
            "F16": (torch.float16, 2),
            "BF16": (torch.bfloat16, 2),
        }
        with open(quarot_path, "rb") as handle:
            header_len = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_len))
        meta = header["global_rotation"]
        if meta["dtype"] not in dtype_map:
            raise ValueError(f"unsupported quarot global_rotation dtype: {meta['dtype']!r}")
        q_dtype, itemsize = dtype_map[meta["dtype"]]
        offset = 8 + header_len + meta["data_offsets"][0]
        numel = 1
        for dim in meta["shape"]:
            numel *= dim
        with open(quarot_path, "rb") as handle:
            handle.seek(offset)
            raw = handle.read(numel * itemsize)
        q = torch.frombuffer(bytearray(raw), dtype=q_dtype).reshape(meta["shape"])
    return q.t().contiguous().to(device=device, dtype=torch.float32)


class EngramEmbedding(nn.Module):
    """The n-gram hash table, sharded by complete hash heads over TP ranks.

    Rows stay quantized in memory and are dequantized to bf16 on lookup (a
    3.8e8-row table must not be materialized in bf16). TP ranks own
    contiguous head-bucket row ranges. ``tp_size`` must divide the hash-column
    count so every rank owns the same number of complete heads
    (``n_hash_cols // tp_size``); a non-dividing shard raises at construction
    instead of being padded, so the all-gather always reassembles exactly the
    real columns.

    ``storage`` selects the on-device layout: ``fp8`` (official FP8 release:
    FP8-E4M3 payload bytes + ue8m0 scale bytes) or ``int8`` (Eco-Tech W8A8
    export: symmetric int8 payload + float32 per-group-32 scales).
    """

    def __init__(
        self,
        num_embeddings: int,
        dim: int,
        head_sizes: Sequence[int],
        tp_size: int,
        tp_rank: int,
        quant_block: int = 32,
        storage: str = ENGRAM_STORAGE_FP8,
        device: torch.device | None = None,
        offload: bool = False,
        layer_id: int = -1,
        dp_shared_memory: bool = False,
        dp_group: Any = None,
    ) -> None:
        super().__init__()
        if not head_sizes or any(size <= 0 for size in head_sizes):
            raise ValueError(f"engram head sizes must be positive, got {head_sizes}")
        if sum(head_sizes) > num_embeddings:
            raise ValueError(f"engram head sizes sum to {sum(head_sizes)} rows but the table only has {num_embeddings}")
        if dim % quant_block != 0:
            raise ValueError(f"engram head dim {dim} is not divisible by {quant_block}")
        self.num_embeddings = int(num_embeddings)
        self.dim = int(dim)
        self.quant_block = int(quant_block)
        self.storage = storage
        self.n_hash_cols = len(head_sizes)
        if self.n_hash_cols % tp_size != 0:
            # Uniform TP head shards are a hard requirement: each rank's block
            # width is n_hash_cols // tp_size, and a short last shard would need
            # padding the (now strict) gather no longer reassembles. Name the
            # supported TP sizes that actually divide the head count.
            supported = sorted(size for size in (1, 2, 4, 8) if self.n_hash_cols % size == 0)
            raise ValueError(
                "Engram requires uniform TP head shards: "
                f"n_hash_cols={self.n_hash_cols} is not divisible by tp_size={tp_size}. "
                f"Use TP from {supported}. dp_shared_memory does not relax this requirement."
            )
        self.part_n_hash_cols = self.n_hash_cols // tp_size
        self.head_start = tp_rank * self.part_n_hash_cols
        head_end = self.head_start + self.part_n_hash_cols
        self.vocab_start = sum(head_sizes[: self.head_start])
        self.vocab_end = sum(head_sizes[:head_end])
        self.part_num_embeddings = self.vocab_end - self.vocab_start
        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        # Model layer id, used to name the layer in the staging-capacity
        # error; -1 when the caller does not supply one.
        self.layer_id = int(layer_id)
        self.setup_device = torch.device(device) if device is not None else torch.device("cpu")
        # Host offload is int8-only (the fp8 path keeps raw byte tables and has
        # no CPU decoder); anything else falls back to the HBM path with a
        # warning rather than silently skipping engram.
        self.offload = bool(offload) and storage == ENGRAM_STORAGE_INT8
        if offload and not self.offload:
            self._warn_offload_unsupported(storage)
        # Direct-construction mirror of the ``EngramConfig`` invariant: a shared
        # host table without the host-offload path has nothing to share --
        # ``self.offload`` would silently AND the request away below -- so reject
        # it here with the same message the config layer uses instead of dropping
        # it.
        if dp_shared_memory and not offload:
            raise ValueError("engram_config.dp_shared_memory=true requires engram_config.cpu_offload=true")
        # DP-shared host tables are only meaningful on the int8 (W8A8) path: the
        # fp8/bf16 tables have no CPU decoder to share. Reject the request
        # outright instead of inheriting the HBM fallback the offload warning
        # above uses.
        if dp_shared_memory and storage != ENGRAM_STORAGE_INT8:
            raise ValueError(
                "engram_config.dp_shared_memory=true requires the int8 (W8A8) table "
                f"storage (the only CPU-shareable format), got storage={storage!r}"
            )
        self.dp_shared_memory = bool(dp_shared_memory) and self.offload
        if self.dp_shared_memory and dp_group is None:
            raise ValueError("engram_config.dp_shared_memory=true requires an initialized node-local DP group")
        self._dp_group = dp_group
        self._shared_storage: SharedHostEngramStorage | None = None
        # Fail fast at construction: the fused CPU int8 lookup op (the only host
        # dequant path) hard-codes a group size of 32, so an offloaded table with
        # any other quant_block can never be served. Without this the mismatch
        # only surfaced at the first lookup (``_engram_lookup_op_contract_error``)
        # -- after the whole ~27.7GB/rank host table had been loaded. The
        # runtime contract check stays as defence in depth.
        if self.offload and self.quant_block != 32:
            raise ValueError(
                "engram host offload requires quant_block=32 (the fused CPU int8 "
                f"lookup op's fixed group size), got {self.quant_block}; set "
                "engram_config.cpu_offload=false to use the HBM table path"
            )
        table_device: torch.device | str | None = device
        if self.offload:
            table_device = "cpu"
            self._ids_host: torch.Tensor | None = None
            self._owned_host: torch.Tensor | None = None
            self._rows_host: torch.Tensor | None = None
            self._host_capacity = 0
            # Allocation-time pin records for the staging operands (see
            # ``_host_buffer``): the op contract reads these instead of
            # re-probing ``Tensor.is_pinned()`` later.
            self._ids_host_pinned = False
            self._rows_host_pinned = False
            # This instance's most recent pinned-allocation failure cause, if any
            # (``_pin_failure_hint`` explains an unpinned operand with it).
            self._host_pin_failure: str | None = None
            # Two device staging blocks. The graph-visible one is baked into a
            # captured ACL graph and must never move (``_ensure_graph_staging``
            # fails loudly instead); the eager one serves prefill / non-graph
            # decode and may grow freely.
            self._graph_staged_device: torch.Tensor | None = None
            self._graph_capacity = 0
            # Runner-supplied floor for the graph-visible block, set before the
            # first capture to the largest decode token bucket (see
            # ``set_graph_capacity_hint``); 0 means "size from the first pass".
            self._graph_capacity_hint = 0
            self._eager_staged_device: torch.Tensor | None = None
            self._eager_capacity = 0
        if storage == ENGRAM_STORAGE_INT8:
            # Symmetric int8 payloads with one float32 scale per 32 columns
            # (Eco-Tech W8A8 embedding_storage format int8_sym). Offloaded
            # tables stay in *pageable* CPU host memory and never reach the
            # device; only the staging/scratch buffers handed out by
            # ``_host_buffer`` are pinned (they are the op's operands and the
            # source of the non-blocking H2D copy).
            if self.dp_shared_memory:
                # One physical host table per node-local DP shard group: the
                # frombuffer views become the parameters, so every reader in the
                # group gathers from the same pages.
                assert self._dp_group is not None
                self._shared_storage = SharedHostEngramStorage(
                    layer_id=self.layer_id,
                    rows=self.part_num_embeddings,
                    dim=self.dim,
                    quant_block=self.quant_block,
                    group=self._dp_group,
                )
                self._shared_storage.create_or_attach()
                self.weight = nn.Parameter(self._shared_storage.codes, requires_grad=False)
                self.weight_scale = nn.Parameter(self._shared_storage.scales, requires_grad=False)
            else:
                self.weight = nn.Parameter(
                    torch.empty(self.part_num_embeddings, self.dim, dtype=torch.int8, device=table_device),
                    requires_grad=False,
                )
                self.weight_scale = nn.Parameter(
                    torch.empty(
                        self.part_num_embeddings,
                        self.dim // self.quant_block,
                        dtype=torch.float32,
                        device=table_device,
                    ),
                    requires_grad=False,
                )
        else:
            # FP8-E4M3 payloads and ue8m0 scales are kept as raw uint8 bytes:
            # index_select stays on a universally supported dtype and the
            # bit-level decoders in deepseek_v41 do the rest.
            self.weight = nn.Parameter(
                torch.empty(self.part_num_embeddings, self.dim, dtype=torch.uint8, device=device),
                requires_grad=False,
            )
            self.weight_scale = nn.Parameter(
                torch.empty(
                    self.part_num_embeddings,
                    self.dim // self.quant_block,
                    dtype=torch.uint8,
                    device=device,
                ),
                requires_grad=False,
            )

    @staticmethod
    def _warn_offload_unsupported(storage: str) -> None:
        try:
            from scripts.logger import logger

            logger.warning(
                "engram_config.cpu_offload=true ignored for engram storage %r; only the "
                "int8 (W8A8) tables have a host decoder, falling back to the HBM path",
                storage,
            )
        except Exception:  # pragma: no cover - logging must never block loading
            pass

    def _owned_local_cols(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """This rank's ``owned`` mask and clamped ``local`` rows for ``indices``.

        ``indices`` holds integer hash ids (an int32/int64 tensor), never floats
        or a Python list -- the int64 cast below is only to feed
        ``index_select`` and to keep the subtraction/clamp in one dtype. Slices
        out this rank's ``part_n_hash_cols`` head columns, marks which hash ids
        fall inside this rank's row range (``owned``) and clamps the rest into a
        valid local row (``local``) so the caller can zero the columns it does
        not own. Shared by the device path and the host-offload staging path so
        the TP sharding rule lives in exactly one place.
        """
        local_cols = indices[:, self.head_start : self.head_start + self.part_n_hash_cols].to(torch.int64)
        owned = (local_cols >= self.vocab_start) & (local_cols < self.vocab_end)
        local = (local_cols - self.vocab_start).clamp(0, self.part_num_embeddings - 1)
        return owned, local

    def _local_block_device(self, indices: torch.Tensor) -> torch.Tensor:
        """This rank's block from the device-resident quantized table.

        Real head columns are dequantized from the quantized table; columns
        this rank does not own (hash ids outside its row range) contribute
        zeros.
        """
        # Deferred: deepseek_v41 imports this module at load time, so the
        # inlined quant decoders are resolved on first use instead.
        from xllm.python.models.deepseek_v41 import _e4m3_to_float, ue8m0_to_scale

        num_tokens = indices.shape[0]
        owned, local = self._owned_local_cols(indices)
        flat = local.reshape(-1)
        payload = self.weight.index_select(0, flat)
        scale = self.weight_scale.index_select(0, flat)
        if self.storage == ENGRAM_STORAGE_INT8:
            values = payload.to(torch.float32)
            scale = scale.repeat_interleave(self.quant_block, dim=-1)
        else:
            values = _e4m3_to_float(payload)
            scale = ue8m0_to_scale(scale).repeat_interleave(self.quant_block, dim=-1)
        rows = (values * scale).to(torch.bfloat16)
        rows = rows.view(num_tokens, self.part_n_hash_cols, self.dim)
        return torch.where(owned.unsqueeze(-1), rows, torch.zeros_like(rows))

    # -- host offload -------------------------------------------------------

    def _ensure_host_buffers(self, num_tokens: int, *, graph_visible: bool = False) -> None:
        """Ensure the host scratch and the staging block for this pass.

        The pinned host scratch (id/owned/rows) is shared and only ever grown.
        The device staging block is chosen by ``graph_visible``:

        * ``True`` -- the runner's pre-graph pass and the graph warmup/capture
          forwards. This block is the address an ACL graph bakes in at capture,
          so it is allocated once and never reallocated.
        * ``False`` -- eager prefill / non-graph decode, served by a separate
          block that may grow without disturbing a captured graph.
        """
        self._ensure_host_scratch(num_tokens)
        if graph_visible:
            self._ensure_graph_staging(num_tokens)
        else:
            self._ensure_eager_staging(num_tokens)

    def _ensure_host_scratch(self, num_tokens: int) -> None:
        """Grow the pinned host scratch (id/owned/rows); never read in-graph."""
        width = self.part_n_hash_cols
        if self._ids_host is not None and self._host_capacity >= num_tokens:
            return
        # A first allocation (``_ids_host is None``) has no old buffer to drop,
        # so no copy can still be in flight and the drain is skipped. When the
        # scratch grows, dropping the old pinned buffers below would free memory
        # an in-flight H2D copy may still be reading (the copy at the end of the
        # previous ``_local_block_offload`` is non-blocking), so drain the
        # accelerator queue first and never recycle a buffer still in use.
        if self._ids_host is not None:
            _sync_accelerator(self.setup_device)
        capacity = max(self._host_capacity, _ENGRAM_HOST_MIN_CAPACITY)
        while capacity < num_tokens:
            capacity *= 2
        self._host_capacity = capacity
        self._ids_host, self._ids_host_pinned, ids_failure = _host_buffer((capacity, width), torch.int64)
        self._owned_host, _, owned_failure = _host_buffer((capacity, width), torch.bool)
        self._rows_host, self._rows_host_pinned, rows_failure = _host_buffer(
            (capacity, width, self.dim), torch.bfloat16
        )
        # Per-instance cause for this (re)allocation episode: a module-global
        # diagnosis would be overwritten by a sibling layer's allocation and
        # misreported by ``_pin_failure_hint``. All three buffers target this
        # host, so the first failure explains every unpinned sibling.
        self._host_pin_failure = ids_failure or owned_failure or rows_failure

    def set_graph_capacity_hint(self, max_tokens: int) -> None:
        """Pre-size the graph-visible staging block to the largest bucket.

        ACL-graph capture bakes this block's address, so it is allocated once
        and never grown. Lazy capture otherwise fixes the size at the FIRST
        captured bucket, which is normally the smallest (buckets warm up in
        ascending order), so any later, larger decode bucket would overflow and
        raise. The runner passes its largest decode token bucket here before
        the first capture, so the first allocation already covers every bucket.
        Only ever raises the floor, and is ignored once the block exists.
        """
        self._graph_capacity_hint = max(self._graph_capacity_hint, int(max_tokens))

    def _ensure_graph_staging(self, num_tokens: int) -> None:
        """Allocate the graph-visible staging block exactly once.

        ACL-graph replay reads this exact address every step, so growth is not
        allowed: a pass that needs more rows than the captured capacity fails
        loudly instead of silently handing the graph a freed/reused address.
        The first allocation is sized to ``_graph_capacity_hint`` (the runner's
        largest decode bucket) so a later bucket never trips the guard.
        """
        if self._graph_staged_device is not None:
            if num_tokens > self._graph_capacity:
                raise RuntimeError(
                    "engram graph-visible staging capacity exceeded: "
                    f"layer={self.layer_id} rank={self.tp_rank} needs {num_tokens} rows "
                    f"but the block was captured at capacity {self._graph_capacity}. "
                    "An ACL graph baked in the staging address during capture; growing "
                    "it would make every replay read freed memory, so this is refused "
                    "rather than silently corrupting the output. The runner pre-sizes this "
                    "block to its largest decode bucket via configure_engram_graph_capacity; "
                    "an overflow means a bucket exceeded that bound."
                )
            return
        capacity = max(num_tokens, self._graph_capacity_hint, _ENGRAM_HOST_MIN_CAPACITY)
        self._graph_capacity = capacity
        self._graph_staged_device = self._alloc_staging(capacity)

    def _ensure_eager_staging(self, num_tokens: int) -> None:
        """Grow the eager staging block; never referenced by a captured graph."""
        if self._eager_staged_device is not None and self._eager_capacity >= num_tokens:
            return
        capacity = max(self._eager_capacity, _ENGRAM_HOST_MIN_CAPACITY)
        while capacity < num_tokens:
            capacity *= 2
        self._eager_capacity = capacity
        self._eager_staged_device = self._alloc_staging(capacity)

    def _alloc_staging(self, capacity: int) -> torch.Tensor:
        """A bf16 ``[capacity, part_n_hash_cols, dim]`` accelerator staging block."""
        return torch.empty(
            (capacity, self.part_n_hash_cols, self.dim),
            dtype=torch.bfloat16,
            device=self.setup_device,
        )

    def _engram_lookup_op_contract_error(
        self,
        ids: torch.Tensor,
        output: torch.Tensor,
    ) -> str:
        """Why the CPU int8 lookup op cannot run; ``""`` when the contract holds.

        The fused op is the only dequant path for the host-offloaded table, so
        every violation here is a hard error (the caller raises loudly). Checks
        the host-offloaded int8 table and the exact
        dtype/shape/pinning/contiguity contract of the op's four operands. The
        offloaded table is pageable host memory by design (the loader copies it
        straight from the mmap, see
        ``DeepseekV41ForCausalLM._load_engram_ascend``), so pinning is required
        only for the ``_host_buffer`` staging operands.
        """
        if not self.offload:
            return "table is not host-offloaded"
        if self.quant_block != 32:
            return f"quant_block={self.quant_block} != 32"
        if self._shared_storage is not None and not self._shared_storage.is_loaded:
            # The shared table is filled by the leader-only load; reading it
            # before that would gather the zero-initialized segment. The load
            # runs before any capture/replay, so this only fires on a wiring
            # bug (a lookup before load_weights).
            return f"shared host table is {self._shared_storage.state}, not loaded"
        weight, scale = self.weight, self.weight_scale
        if weight.dtype != torch.int8 or weight.dim() != 2 or weight.size(1) != self.dim:
            return f"weight is {weight.dtype} {tuple(weight.shape)}"
        if weight.device.type != "cpu" or not weight.is_contiguous():
            return f"weight is on {weight.device.type} contiguous={weight.is_contiguous()}"
        if scale.dtype != torch.float32 or scale.dim() != 2 or scale.size(0) != weight.size(0):
            return f"scale is {scale.dtype} {tuple(scale.shape)}"
        if scale.size(1) != self.dim // 32:
            return f"scale has {scale.size(1)} groups, expected {self.dim // 32}"
        if scale.device.type != "cpu" or not scale.is_contiguous():
            return f"scale is on {scale.device.type} contiguous={scale.is_contiguous()}"
        if ids.dtype != torch.int64 or ids.dim() != 1 or ids.numel() != output.size(0):
            return f"ids is {ids.dtype} {tuple(ids.shape)} for output {tuple(output.shape)}"
        if (
            ids.device.type != "cpu"
            or not ids.is_contiguous()
            or not self._staging_operand_pinned(ids, self._ids_host, self._ids_host_pinned)
        ):
            return (
                f"ids is on {ids.device.type} contiguous={ids.is_contiguous()} "
                f"pinned={self._ids_host_pinned}{self._pin_failure_hint(self._ids_host_pinned)}"
            )
        if output.dtype != torch.bfloat16 or output.dim() != 2 or output.size(1) != self.dim:
            return f"output is {output.dtype} {tuple(output.shape)}"
        if (
            output.device.type != "cpu"
            or not output.is_contiguous()
            or not self._staging_operand_pinned(output, self._rows_host, self._rows_host_pinned)
        ):
            return (
                f"output is on {output.device.type} contiguous={output.is_contiguous()} "
                f"pinned={self._rows_host_pinned}{self._pin_failure_hint(self._rows_host_pinned)}"
            )
        return ""

    @staticmethod
    def _staging_operand_pinned(
        operand: torch.Tensor,
        staging: torch.Tensor | None,
        pinned: bool,
    ) -> bool:
        """Whether ``operand`` is a view of the recorded pinned staging buffer.

        Compared by underlying storage, not by ``data_ptr()``: the op's operands
        are views of the ``_host_buffer`` staging blocks, and a storage-pointer
        compare accepts any contiguous view/slice of the same staging block -- a
        non-zero-offset slice has a different ``data_ptr()`` than the block head,
        so a plain pointer compare would reject a legitimate staging slice. Still
        rejects a foreign (pageable or otherwise unrecorded) tensor, whose
        storage pointer differs. Together with the allocation-time ``pinned``
        record this replaces the ``Tensor.is_pinned()`` re-probe, whose result
        can be wrong on hosts/PyTorch versions without a registered accelerator.
        """
        return (
            pinned
            and staging is not None
            and operand.untyped_storage().data_ptr() == staging.untyped_storage().data_ptr()
        )

    def _pin_failure_hint(self, pinned: bool) -> str:
        """Explain an unpinned staging operand by its real cause, when known.

        ``_host_buffer`` records the ``pin_memory`` failure it swallowed on this
        embedding (``self._host_pin_failure``); without that the caller only sees
        a bare ``pinned=False`` and may misread it as a code defect rather than a
        host that cannot pin host memory at all.
        """
        if pinned or self._host_pin_failure is None:
            return ""
        return (
            f" (this host cannot allocate pinned CPU memory: {self._host_pin_failure}; "
            "the offloaded table needs pinned staging, so set "
            "engram_config.cpu_offload=false)"
        )

    def stage_ids_offload(self, indices: torch.Tensor, *, graph_visible: bool = False) -> None:
        """Queue this rank's async device->host id/owned copies; no drain here.

        The copies are queued on the ambient accelerator stream with
        ``non_blocking=True`` (valid because the destinations are pinned host
        buffers), so neither copy stalls the host; the caller drains the device
        once (:func:`stage_offload_rows` runs :func:`_sync_accelerator`) before
        the CPU gather reads them. No host-side work runs here either -- only the
        queued copies.

        The first half of the offload gather (see :meth:`_local_block_offload`),
        split out so a batch of layers can queue every layer's D2H before a
        single device drain (:func:`stage_offload_rows`). Only the
        ``Tensor.copy_`` calls that queue accelerator->host work run here; the
        CPU gather and the pinned->device upload are deferred to
        :meth:`finish_stage_offload`, which must run after exactly one
        :func:`_sync_accelerator` drain.
        """
        num_tokens = indices.shape[0]
        self._ensure_host_buffers(num_tokens, graph_visible=graph_visible)
        assert self._ids_host is not None
        assert self._owned_host is not None

        owned, local = self._owned_local_cols(indices)

        # Device -> pinned host: one batched copy per tensor, queued here on the
        # current accelerator stream with ``non_blocking=True`` so neither copy
        # stalls the host. ``_ids_host``/``_owned_host`` are pinned host buffers
        # (``_host_buffer``), which is what makes the async D2H valid. Both are
        # part of the same batch, so both must be non-blocking -- making only one
        # of them synchronous would still block on every layer. The caller drains
        # the device once after every layer has queued its copies (see
        # :func:`stage_offload_rows`, which runs :func:`_sync_accelerator`), then
        # the CPU gather reads them; the sync point and its count are unchanged.
        self._ids_host[:num_tokens].copy_(local, non_blocking=True)
        self._owned_host[:num_tokens].copy_(owned, non_blocking=True)

    def finish_stage_offload(self, num_tokens: int, *, graph_visible: bool = False) -> torch.Tensor:
        """CPU gather + pinned->device upload for a staged pass (second half).

        The second half of :meth:`_local_block_offload`; must run after
        :meth:`stage_ids_offload` staged the same ``num_tokens`` and after
        exactly one :func:`_sync_accelerator` drain, so every queued D2H is
        visible to the CPU gather below. ``graph_visible`` selects the same
        destination staging block :meth:`stage_ids_offload` ensured. Returns the
        selected staging slice.
        """
        assert self._ids_host is not None
        assert self._owned_host is not None
        assert self._rows_host is not None
        staging = self._graph_staged_device if graph_visible else self._eager_staged_device
        assert staging is not None

        flat = self._ids_host[:num_tokens].reshape(-1).to(torch.int64)
        # Row-index correspondence for the reset mask below. ``_ids_host`` and
        # ``_owned_host`` are freshly allocated C-contiguous tensors, so the k-th
        # flat element maps to ``(token, col) = (k // width, k % width)`` (C
        # order). ``rows_out = self._rows_host[:num_tokens].view(-1, self.dim)``
        # has its k-th row produced from ``flat[k]``, and
        # ``self._owned_host[:num_tokens].reshape(-1, 1)`` flattens ``owned`` in
        # the same C order, so column k's mask lines up with ``owned[token, col]``
        # and broadcasts over that row's ``dim`` features.
        rows_out = self._rows_host[:num_tokens].view(-1, self.dim)
        reason = self._engram_lookup_op_contract_error(flat, rows_out)
        if reason:
            raise RuntimeError(
                f"engram host offload cannot run the CPU int8 lookup op ({reason}); "
                "the fused op is the only dequant path for the offloaded table"
            )
        # The op dequantizes straight into the pinned staging buffer. It does
        # not take the ownership mask, so this rank's unowned columns are zeroed
        # in place, matching the device path's ``torch.where``. Kept here (not
        # fused into the op, whose contract has no mask operand, nor moved to the
        # device): it is in-place on the CPU staging buffer already resident at
        # this point, adds no device round-trip, and costs the same order as the
        # gather -- not shown to be a hot spot, so an op-contract change is not
        # warranted before a measured gain.
        _get_engram_lookup_op()(self.weight, self.weight_scale, flat, rows_out)
        rows_out.masked_fill_(~self._owned_host[:num_tokens].reshape(-1, 1), 0.0)
        _log_engram_lookup_op_active()

        # Pinned host -> the selected device staging block.
        staging[:num_tokens].copy_(self._rows_host[:num_tokens], non_blocking=True)
        return staging[:num_tokens]

    def _local_block_offload(self, indices: torch.Tensor, *, graph_visible: bool = False) -> torch.Tensor:
        """Gather + dequantize this rank's block from the host-resident table.

        One NPU->CPU id copy (followed by a single device sync) and one
        pinned->NPU staged copy per call. The table itself is pageable host
        memory; only the ``_host_buffer`` staging operands are pinned. The CPU
        dequant always runs the fused in-repo ``xllm_ops.engram_int8_lookup_cpu``
        kernel, which writes the bf16 rows straight into the pinned staging
        buffer and stays bitwise equal to the device path; there is no eager
        host fallback.

        ``graph_visible`` selects the destination staging block: the persistent
        graph-visible one (see :meth:`_ensure_graph_staging`) or the growable
        eager one.

        The single-layer entry point: the two halves
        (:meth:`stage_ids_offload` + :meth:`finish_stage_offload`) are split so a
        batch of layers can share one device drain (:func:`stage_offload_rows`);
        called together here, they are the original one-copy/one-sync/one-upload
        sequence.
        """
        self.stage_ids_offload(indices, graph_visible=graph_visible)
        _sync_accelerator(self.setup_device)
        return self.finish_stage_offload(indices.shape[0], graph_visible=graph_visible)

    def fallback_to_device(self) -> bool:
        """Move an offloaded table onto the accelerator (HBM fallback).

        Used when the host path cannot be set up, the host-table load fails or
        the pinned staging cannot be set up; engram stays enabled on the device
        path instead of being silently skipped. Returns True when a fallback
        happened.

        Contract: this may only run *before* ACL-graph capture. Once a
        graph-visible staging block exists its address is baked into the
        captured graph (see :meth:`_ensure_graph_staging`), so dropping the host
        path here would leave every replay reading a freed/reused address. The
        guard below therefore fail-fasts (raises ``RuntimeError``) rather than
        silently corrupting; it is a safety check, not an unreachable no-op.

        The two load-time call sites (the loading-time ``_load_engram_ascend``
        fallback and the post-load ``_reconcile_engram_offload_active``
        demotion) run before any capture during the initial load, but they are
        not the only path. The in-place level-2 reload (``update_weights`` ->
        ``WorkerImpl::load_model``) re-enters the weight load after capture, so
        this guard *is* reachable post-capture and must fail-fast there rather
        than be read as "cannot happen".

        The two-phase commit below only makes the two accelerator copies atomic
        for this embedding; the surrounding reload / row-copy sequence is not
        transactional, so a post-capture reload still needs its own quiesce
        verification (stop replays / drain the device before reloading). This
        helper makes no safety promise about the reload as a whole.

        Calling convention: load-time only, single-threaded. The
        ``_graph_capacity`` guard below is a pre-capture check, not a lock, so
        the caller must not race it from another thread or overlap it with an
        in-flight capture.
        """
        if not self.offload:
            return False
        if self._shared_storage is not None:
            # A DP-shared table is one physical copy per node-local group; a
            # local per-rank demotion would copy only this rank's alias onto the
            # device and silently diverge from its peers (which keep reading the
            # shared segment). The shared load is fail-fast instead, so no
            # per-layer fallback is ever needed here.
            raise RuntimeError(
                "engram fallback_to_device() is not supported for a DP-shared host table: "
                f"layer={self.layer_id} rank={self.tp_rank} shares its table with its "
                "node-local DP group, and a per-rank HBM demotion would desynchronize the "
                "group. engram_config.dp_shared_memory requires the shared host load to succeed"
            )
        if self._graph_capacity > 0 or self._graph_staged_device is not None:
            raise RuntimeError(
                "engram fallback_to_device() must run before ACL-graph capture: "
                f"layer={self.layer_id} rank={self.tp_rank} already allocated a "
                f"graph-visible staging block (capacity={self._graph_capacity}). A "
                "captured graph has baked that address, so demoting the table to HBM "
                "here would make every replay read freed memory. Reconcile the "
                "offload decision before the first capture instead."
            )
        # Two-phase commit: run both accelerator copies into locals and mutate
        # nothing until both have succeeded. A per-parameter commit
        # (``self.weight.data = ...`` then ``self.weight_scale.data = ...``)
        # would leave ``weight`` on the device and ``weight_scale`` still on the
        # host if the second ``.to()`` raised (e.g. device OOM), handing the
        # caller a half-demoted embedding. With the locals, a raise on either
        # copy drops the fresh tensor and leaves both parameter storages,
        # ``offload`` and the host scratch/staging state exactly as they were.
        # Scope: atomic for these two parameters and this embedding's flags
        # only; the surrounding reload / row-copy sequence is not covered (see
        # ``DeepseekV41ForCausalLM._load_engram_ascend_private``).
        device_weight = self.weight.data.to(self.setup_device)
        device_weight_scale = self.weight_scale.data.to(self.setup_device)
        self.weight.data = device_weight
        self.weight_scale.data = device_weight_scale
        self.offload = False
        self._ids_host = None
        self._owned_host = None
        self._rows_host = None
        self._host_capacity = 0
        self._ids_host_pinned = False
        self._rows_host_pinned = False
        self._graph_staged_device = None
        self._graph_capacity = 0
        self._eager_staged_device = None
        self._eager_capacity = 0
        return True

    def close_host_offload(self) -> None:
        """Release a DP-shared host table on shutdown / reload.

        Order matters: the ``weight`` / ``weight_scale`` parameters alias the
        shared mappings, so they are dropped FIRST (no reader follows a freed
        segment), then the storage closes its handles. A failed close keeps
        ``_shared_storage`` set so the call is retryable and never leaves a live
        parameter pointing at unmapped memory. No-op for the private path.
        """
        storage = self._shared_storage
        if storage is None:
            return
        self.weight = nn.Parameter(torch.empty(0, dtype=torch.int8, device="cpu"), requires_grad=False)
        self.weight_scale = nn.Parameter(torch.empty(0, dtype=torch.float32, device="cpu"), requires_grad=False)
        # A raising ``close`` skips the assignment below, so ``_shared_storage``
        # stays set and the call is retryable; the parameters are already
        # detached, so nothing dangles.
        storage.close()
        self._shared_storage = None

    def local_block(self, indices: torch.Tensor, *, graph_visible: bool = False) -> torch.Tensor:
        """This rank's ``[T, part_n_hash_cols, dim]`` rows for ``indices``.

        Device lookup or (offload) pageable-host gather through the pinned
        staging buffers, per :attr:`offload`.
        Kept separate from :meth:`gather_heads` so an offloaded graph only
        stages this rank's block on the host while the TP all-gather stays
        inside the captured graph (a per-step host collective would desync
        ranks and change the graph's fixed collective order).

        ``graph_visible`` routes an offload gather into the persistent
        graph-visible staging block instead of the growable eager one.
        """
        if self.offload:
            return self._local_block_offload(indices, graph_visible=graph_visible)
        return self._local_block_device(indices)

    def gather_heads(self, local: torch.Tensor) -> torch.Tensor:
        """Reassemble the full ``[T, n_hash_cols, dim]`` head set from a local block.

        Only this rank's head columns are read; foreign rows contribute zeros
        and the TP all-gather reassembles the full head set. Each rank owns
        exactly ``n_hash_cols // tp_size`` complete heads (enforced at
        construction), so the gathered width is exactly ``n_hash_cols`` -- no
        padding, no post-gather trim.
        """
        rows = local
        if self.tp_size > 1:
            from xllm.python import distributed as _distributed

            rows = _distributed.tp_all_gather(rows, dim=1, world_size=self.tp_size)
        return rows

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """``[T, n_hash_cols]`` global row ids -> ``[T, n_hash_cols, dim]`` bf16."""
        return self.gather_heads(self.local_block(indices))


# ---------------------------------------------------------------------------
# Per-layer Engram module
# ---------------------------------------------------------------------------


class Engram(nn.Module):
    """Writes an n-gram lookup into the residual stream, gated by match quality.

    The hash ids fetch ``n_hash_cols`` rows; ``wkv`` turns them into one key
    per hc copy plus a shared value. The gate is a normalized dot product of
    the stream against the key, signed-sqrt'ed before the sigmoid (matching
    the reference kernel bit-for-bit, including the ``dot == 0`` edge that
    keeps the positive clamp value).
    """

    def __init__(
        self,
        layout: EngramLayout,
        layer_hash_index: int,
        hidden_size: int,
        hc_mult: int,
        rms_norm_eps: float,
        tp_size: int,
        tp_rank: int,
        dtype: torch.dtype,
        device: torch.device,
        storage: str = ENGRAM_STORAGE_FP8,
        gate_unrotate: torch.Tensor | None = None,
        offload: bool = False,
        dp_shared_memory: bool = False,
        dp_group: Any = None,
    ) -> None:
        super().__init__()
        self.layer_hash_index = layer_hash_index
        self.dim = hidden_size
        self.hc_mult = hc_mult
        self.eps = rms_norm_eps
        self.embed_tokens = EngramEmbedding(
            layout.num_embeddings[layer_hash_index],
            layout.head_dim,
            layout.head_sizes(layer_hash_index),
            tp_size,
            tp_rank,
            storage=storage,
            device=device,
            offload=offload,
            layer_id=layout.layer_ids[layer_hash_index],
            dp_shared_memory=dp_shared_memory,
            dp_group=dp_group,
        )
        n_hash_cols = layout.n_hash_cols
        # Replicated (not TP-sharded), matching the reference ReplicatedLinear.
        self.wkv = nn.Linear(
            n_hash_cols * layout.head_dim,
            hidden_size * (hc_mult + 1),
            bias=False,
            dtype=dtype,
            device=device,
        )
        self.q_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        self.k_weight = nn.Parameter(
            torch.empty(hc_mult, hidden_size, dtype=dtype, device=device),
            requires_grad=False,
        )
        # QuaRot gate un-rotation (Eco-Tech W8A8 export): the stream arrives
        # rotated (h_rot) while q/k/wkv-K are in the original basis, so the
        # gate dot uses h_rot @ Q.T. None on unrotated checkpoints.
        self.register_buffer("gate_unrotate", gate_unrotate, persistent=False)
        self._staged_rows: torch.Tensor | None = None
        # Offload: this rank's host-gathered block, staged by the runner outside
        # the graph. The TP all-gather stays in-graph (gather_heads below).
        self._staged_local: torch.Tensor | None = None
        # Whether ``_staged_local`` currently points into the graph-visible
        # staging block (as opposed to the growable eager one).
        self._staged_local_graph = False
        # Row count of the pass ``begin_stage_offload`` last queued, read back by
        # ``finish_stage_offload`` (the two halves of the batched host staging).
        self._staging_num_tokens = 0

    @property
    def offload(self) -> bool:
        """Whether the table lives in host memory (mirrors the embedding's path)."""
        return self.embed_tokens.offload

    def close_host_offload(self) -> None:
        """Release this layer's DP-shared host table (no-op otherwise)."""
        self.embed_tokens.close_host_offload()

    def prepare_embeddings(self, hash_ids: torch.Tensor, *, graph_visible: bool = False) -> None:
        """Gather this layer's rows before the decoder layer loop starts.

        Offload mode stages only this rank's block into a persistent device
        block (the run happens outside the ACL graph); the default path
        resolves the fully gathered head set eagerly. ``graph_visible`` must be
        True for any pass whose result a captured graph reads (the runner's
        pre-graph staging pass and the graph warmup/capture forwards), so those
        rows land in the address-stable graph staging block.
        """
        if self.offload:
            self._staged_local = self.embed_tokens.local_block(hash_ids, graph_visible=graph_visible)
            self._staged_local_graph = graph_visible
        else:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)

    def begin_stage_offload(self, hash_ids: torch.Tensor, *, graph_visible: bool = False) -> None:
        """Queue this layer's host-staging D2H copies (offload mode only).

        First phase of the batched host staging (:func:`stage_offload_rows`):
        every layer queues its D2H copies before the single device drain, so an
        N-layer pass costs one synchronize instead of one per layer. Records the
        staged row count for :meth:`finish_stage_offload`. A no-op for the device
        path (nothing is staged off the device).
        """
        if not self.offload:
            return
        self.embed_tokens.stage_ids_offload(hash_ids, graph_visible=graph_visible)
        self._staging_num_tokens = hash_ids.shape[0]

    def finish_stage_offload(self, *, graph_visible: bool = False) -> None:
        """Run the CPU gather + H2D upload and publish the staged block (offload only).

        Second phase of :func:`stage_offload_rows`; must run after the single
        device drain that followed :meth:`begin_stage_offload`. A no-op for the
        device path.
        """
        if not self.offload:
            return
        self._staged_local = self.embed_tokens.finish_stage_offload(
            self._staging_num_tokens,
            graph_visible=graph_visible,
        )
        self._staged_local_graph = graph_visible

    def _reject_stale_row_count(self, staged: torch.Tensor, num_tokens: int | None) -> None:
        """Reject a staging block whose row count does not match this step.

        The graph-replay path (``hash_ids=None``) does not re-run Python, so a
        staging block sized for a different batch would make the captured graph
        read the wrong rows.
        """
        if num_tokens is not None and staged.shape[0] != num_tokens:
            raise RuntimeError(
                "engram mixed-mode staging row mismatch: staged "
                f"{staged.shape[0]} rows but this step has "
                f"{num_tokens} tokens; the graph would read the wrong rows"
            )

    def embed(self, hash_ids: torch.Tensor | None, num_tokens: int | None = None) -> torch.Tensor:
        """Return the gathered ``[T, n_hash_cols, dim]`` rows for ``hash_ids``.

        ``None`` is the offload graph-replay contract: the local block was
        staged by the pre-graph host pass, so only the in-graph head all-gather
        runs (a replay does not re-run Python, and a host collective here would
        desync the TP ranks). ``num_tokens`` is the current step's token-row
        count; :meth:`forward` passes it so the mixed-mode branch below can
        refuse a staging block sized for a different batch. It is optional only
        so direct/test callers need not supply it.
        """
        if hash_ids is None:
            if self._staged_local is not None:
                # Same mixed-mode defense as the ``_staged_rows`` branch below:
                # the graph-replay path (``hash_ids=None``) does not re-run
                # Python, so a staging block sized for a different batch would
                # make the graph read the wrong rows. Dead today under the
                # all-or-nothing reconcile, but required if that is ever relaxed.
                self._reject_stale_row_count(self._staged_local, num_tokens)
                return self.embed_tokens.gather_heads(self._staged_local)
            if self._staged_rows is not None:
                # Mixed-mode defense (production-unreachable today, kept as
                # defence in depth). A layer whose host table fell back to HBM
                # stages its device rows in ``_staged_rows``, so the graph-replay
                # contract (``hash_ids=None``) could resolve instead of raising
                # inside the captured region. It is dead today because two gates
                # exclude it: ``DeepseekV41Model._reconcile_engram_offload_active``
                # demotes the *whole* model to the device path when any single
                # layer falls back (all-or-nothing), and the capture branch only
                # passes ``hash_ids=None`` while ``engram_offload_active``. If a
                # future change relaxes that all-or-nothing reconcile (e.g. a
                # per-layer offload decision, or an all-offload-off capture path),
                # this branch becomes live and must return rows matching the
                # current batch -- hence the exact shape check below.
                self._reject_stale_row_count(self._staged_rows, num_tokens)
                return self._staged_rows
            raise RuntimeError(
                "engram embed called without hash ids and no staged rows; "
                "offload mode requires the pre-graph staging pass"
            )
        if self.offload:
            # Keep ``_staged_local`` on the block that matches this forward's
            # graph visibility. A graph-visible forward -- the runner's
            # pre-graph pass and, critically, the graph warmup resolves --
            # MUST leave ``_staged_local`` on the address-stable graph block:
            # the subsequent capture reads whatever this pass leaves behind,
            # so re-mounting to the (growable) eager block here would bake the
            # eager address into the captured graph; an eager pass that later
            # grows (or frees) that block would then leave the graph replay
            # reading a moved or freed address. Only a pass outside the graph
            # may use the eager block.
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
            # Exact shape equality (``!=``), not a capacity ``<`` test: a
            # smaller ``hash_ids`` must re-gather, otherwise the staged prefix
            # rows from a larger earlier gather would be silently returned for
            # a different, shorter batch (with ``<`` a reconstructed 8-row
            # staging block would be reused verbatim for a 3-row call).
            if (
                self._staged_local is None
                or self._staged_local.shape[0] != hash_ids.shape[0]
                or self._staged_local_graph != graph_visible
            ):
                self._staged_local = self.embed_tokens.local_block(hash_ids, graph_visible=graph_visible)
                self._staged_local_graph = graph_visible
            return self.embed_tokens.gather_heads(self._staged_local[: hash_ids.shape[0]])
        if self._staged_rows is None or self._staged_rows.shape[0] < hash_ids.shape[0]:
            self._staged_rows = self.embed_tokens.lookup(hash_ids)
        return self._staged_rows[: hash_ids.shape[0]]

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor | None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Gated residual injection into the full mHC stream.

        Args:
            hidden_states: ``[T, hc_mult, dim]`` residual stream entering this
                block (between the previous sublayer's post and this block's
                pre, so the mix coefficients see the injected stream).
            hash_ids: ``[T, n_hash_cols]`` row ids of this layer, or None when
                the rows were staged by the pre-graph offload pass.
            token_mask: ``[T]``; False shuts the gate so those positions pass
                through untouched (image spans).
        """
        rows = self.embed(hash_ids, num_tokens=hidden_states.shape[0]).to(self.wkv.weight.dtype)
        kv = self.wkv(rows.flatten(-2))
        num_tokens, hc_mult, dim = hidden_states.shape
        keys = kv[:, : hc_mult * dim].view(num_tokens, hc_mult, dim).float()
        value = kv[:, hc_mult * dim :].float()
        hidden = hidden_states.float()
        # QuaRot: q/k/keys are original-basis, the stream is rotated; the
        # gate dot must run on the un-rotated stream (h_rot @ Q.T). The value
        # half of wkv is pre-folded to output the rotated basis, so the
        # residual update below must NOT rotate the increment again.
        gate_hidden = hidden
        if self.gate_unrotate is not None:
            gate_hidden = hidden @ self.gate_unrotate
        hidden_rms = torch.rsqrt(gate_hidden.square().mean(-1) + self.eps)
        key_rms = torch.rsqrt(keys.square().mean(-1) + self.eps)
        dot = (gate_hidden * self.q_weight.float() * self.k_weight.float() * keys).sum(-1)
        dot = dot * hidden_rms * key_rms * float(dim) ** -0.5
        gate_input = dot.abs().clamp_min(_ENGRAM_GATE_CLAMP).sqrt()
        gate_input = torch.where(dot < 0.0, -gate_input, gate_input)
        gate = torch.sigmoid(gate_input)
        if token_mask is not None:
            gate = torch.where(token_mask.reshape(-1, 1), gate, torch.zeros_like(gate))
        # The value is shared across hc copies ([T, dim]); the gate is
        # per-copy ([T, hc]).
        output = hidden + gate.unsqueeze(-1) * value.unsqueeze(1)
        return output.to(hidden_states.dtype)


def stage_offload_rows(
    engrams: Sequence[Engram],
    hash_ids: Sequence[torch.Tensor],
    *,
    graph_visible: bool,
) -> None:
    """Two-phase batched host staging for a set of Engram layers.

    Phase 1 (:meth:`Engram.begin_stage_offload`, each layer) queues every
    layer's async device->host id/owned copies (``non_blocking`` on pinned
    staging); a single :func:`_sync_accelerator` then drains the device; phase 2
    (:meth:`Engram.finish_stage_offload`, each layer) runs the CPU gather +
    pinned->device upload. Per-layer staging would
    drain the whole device once per layer (see
    :meth:`EngramEmbedding._local_block_offload`), so an N-layer pass paid N
    blocking drains where N-1 waited on copies already drained; this pass pays
    exactly one. The protected drain inside
    :meth:`EngramEmbedding._ensure_host_scratch` is unchanged: it still runs on
    the pass that first grows the pinned scratch, guarding that realloc against
    an in-flight H2D, not the D2H visibility this function batches.

    ``engrams`` and ``hash_ids`` are parallel sequences (each layer with its own
    ids); ``graph_visible`` selects the persistent graph staging block (True on
    the runner's pre-graph pass). A no-op for the non-offload path: each
    ``begin``/``finish`` is a no-op when the layer keeps its table on the device.
    """
    if not engrams:
        return
    for engram, ids in zip(engrams, hash_ids, strict=True):
        engram.begin_stage_offload(ids, graph_visible=graph_visible)
    _sync_accelerator(engrams[0].embed_tokens.setup_device)
    for engram in engrams:
        engram.finish_stage_offload(graph_visible=graph_visible)
