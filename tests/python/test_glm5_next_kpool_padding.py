# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

import os

import pytest
import torch

from xllm.python.models import glm5_next


def _make_indexer(device: torch.device) -> glm5_next.Glm5NextIndexer:
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=8,
        index_kpool_compress=True,
    )
    indexer = glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device)
    indexer.index_kpool_always_select_tail = True
    return indexer


@pytest.mark.parametrize("query_len", [1, 4])
@torch.inference_mode()
def test_padding_keeps_row_identity_and_causal_lengths(query_len: int, monkeypatch: pytest.MonkeyPatch) -> None:
    indexer = _make_indexer(torch.device("cpu"))
    rows = 2 * query_len
    query = torch.arange(rows, dtype=torch.bfloat16).view(2, query_len, 1, 1).expand(-1, -1, 2, 128).contiguous()
    weights = torch.ones(2, query_len, 2, dtype=torch.bfloat16)
    mask = torch.ones(2, query_len, dtype=torch.bool)
    mask[1, -1] = False
    positions = torch.arange(rows).reshape(2, query_len) + 9
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    lengths = positions[:, -1].to(torch.int32) + 1
    calls = []
    expected_rows = torch.arange(rows)[mask.reshape(-1)]
    identity = torch.arange(rows).reshape(2, query_len, 1, 1)
    expanded = (identity * 4 + torch.arange(4)).expand(-1, -1, 2, -1).flatten(2)
    counts = lengths[:, None] if query_len == 1 else positions + 1
    tail = counts[..., None] // 4 * 4 + torch.arange(3)
    tail = tail.masked_fill(torch.arange(3) >= counts[..., None] % 4, -1)
    expected = torch.cat((expanded, tail), dim=-1)
    expected[~mask] = -1

    def _kernel(
        q: torch.Tensor,
        cache: torch.Tensor,
        w: torch.Tensor,
        tail_k: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        identity_tokens = q[:, 0, 0, 0].long()
        calls.append((identity_tokens, kwargs["actual_seq_k"] * 4, kwargs["block_table"]))
        # The kernel returns the expanded compact row; the mock mirrors the rows
        # the test expects.
        return expected.reshape(rows, -1)[expected_rows][:, None, :].to(torch.int32), None

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        torch.zeros(8, 4, 1, 128, dtype=torch.bfloat16),
        tables,
        positions,
    )
    assert result is not None, "padding must remain on the fused path"
    assert result.shape == (2, query_len, 11)
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], expected_rows)
    expected_lengths = lengths if query_len == 1 else positions.reshape(-1) + 1
    torch.testing.assert_close(calls[0][1], (expected_lengths[mask.reshape(-1)].to(torch.int32) // 4) * 4)
    torch.testing.assert_close(calls[0][2], tables[expected_rows // query_len])
    torch.testing.assert_close(result, expected)


@torch.inference_mode()
def test_all_padding_returns_sentinels_without_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    indexer = _make_indexer(torch.device("cpu"))

    def _unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("an all-padding batch must not call the NPU kernel")

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _unexpected, raising=False)
    result = indexer._select_topk_fused_pa(
        torch.ones(2, 4, 2, 128, dtype=torch.bfloat16),
        torch.ones(2, 4, 2, dtype=torch.bfloat16),
        torch.tensor([14, 17], dtype=torch.int32),
        torch.zeros(2, 4, dtype=torch.bool),
        torch.zeros(2, 16, 1, 128, dtype=torch.bfloat16),
        torch.tensor([[0], [1]], dtype=torch.int32),
        torch.tensor([[10, 11, 12, 13], [13, 14, 15, 16]]),
    )
    assert result is not None
    torch.testing.assert_close(result, torch.full((2, 4, 11), -1, dtype=torch.long))


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@torch.inference_mode()
def test_real_npu_padding_matches_independent_causal_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    indexer = _make_indexer(device)
    query = torch.ones(2, 4, 2, 128, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 4, 2, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([14, 17], device=device, dtype=torch.int32)
    positions = lengths[:, None] - 4 + torch.arange(4, device=device)
    mask = torch.ones(2, 4, device=device, dtype=torch.bool)
    mask[0, 1] = False
    mask[1, 0] = False
    mask[1, 3] = False
    cache = torch.zeros(2, 16, 1, 128, device=device, dtype=torch.bfloat16)
    cache[:, :4].copy_(torch.arange(1, 5, device=device, dtype=torch.bfloat16).view(1, 4, 1, 1).expand(2, -1, -1, 128))
    tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)
    result = indexer._select_topk_fused_pa(query, weights, lengths, mask, cache, tables, positions)
    assert result is not None
    expected_positions = (
        (range(11), range(4, 12), range(4, 13), range(4, 14)),
        (range(4, 14), range(4, 15), range(8, 16), range(8, 17)),
    )
    for batch in range(2):
        for row in range(4):
            if not bool(mask[batch, row].item()):
                assert bool((result[batch, row] == -1).all().item())
                continue
            valid = result[batch, row]
            valid = valid[valid >= 0].sort().values.cpu()
            torch.testing.assert_close(valid, torch.tensor(list(expected_positions[batch][row]), dtype=torch.long))
    torch.npu.synchronize()


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("query_len", [1, 4])
@torch.inference_mode()
def test_real_npu_production_shape_matches_independent_rows(monkeypatch: pytest.MonkeyPatch, query_len: int) -> None:
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=32,
        index_head_dim=128,
        index_kpool=4,
        index_topk=2048,
        index_kpool_compress=True,
    )
    indexer = glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, device).to(device)
    indexer.index_kpool_always_select_tail = True
    torch.manual_seed(20261102)
    query = torch.zeros(2, query_len, 32, 128, device=device, dtype=torch.bfloat16)
    query[..., 0] = 1
    query[..., 1] = 1 / 128
    weights = torch.full((2, query_len, 32), 1 / 32, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([65536, 65024], device=device, dtype=torch.int32)
    positions = lengths[:, None] - query_len + torch.arange(query_len, device=device)
    # The production concern is a padded batch: keep one (or, for multi-query
    # rows, three) sentinel rows while exercising 64K causal lengths.
    mask = torch.ones(2, query_len, dtype=torch.bool, device=device)
    mask[0, -1] = False
    if query_len > 1:
        mask[1, 0] = False
        mask[1, -1] = False
    cache = torch.zeros(2048, 16, 1, 128, device=device, dtype=torch.bfloat16)
    pool_ids = torch.arange(32768, device=device).remainder(16384).reshape(2048, 16, 1)
    cache[..., 0] = pool_ids.div(128, rounding_mode="floor").to(torch.bfloat16)
    cache[..., 1] = pool_ids.remainder(128).to(torch.bfloat16)
    tables = torch.arange(2048, device=device, dtype=torch.int32).reshape(2, 1024)
    # Exercise the per-query page table branch as well as 64K causal lengths.
    query_tables = tables[:, None, :].expand(2, query_len, 1024).reshape(2 * query_len, 1024).contiguous()

    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        cache,
        tables,
        positions,
        query_tables,
    )
    assert result is not None
    assert result.shape == (2, query_len, 2051)
    for batch, row in mask.cpu().nonzero().tolist():
        visible = int(positions[batch, row].item()) + 1
        end_pool = visible // 4
        expected = torch.arange((end_pool - 512) * 4, visible, dtype=torch.long)
        actual = result[batch, row][result[batch, row] >= 0].sort().values.cpu()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        valid = result[batch, row][result[batch, row] >= 0]
        assert bool((valid <= positions[batch, row]).all().item())
    assert bool((result[~mask] == -1).all().item())
    torch.npu.synchronize()


@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_unpadded_and_graph_preserve_original_kernel_contract(
    monkeypatch: pytest.MonkeyPatch, graph_mode: bool
) -> None:
    indexer = _make_indexer(torch.device("cpu"))
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: graph_mode)
    query = torch.zeros(2, 4, 2, 128, dtype=torch.bfloat16)
    weights = torch.ones(2, 4, 2, dtype=torch.bfloat16)
    lengths = torch.tensor([14, 17], dtype=torch.int32)
    positions = lengths[:, None] - 4 + torch.arange(4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    if graph_mode:
        mask[0, 1] = False
    calls = []

    def _kernel(
        q: torch.Tensor,
        cache: torch.Tensor,
        w: torch.Tensor,
        tail: torch.Tensor,
        topk: int,
        pool_size: int,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append((tail.clone(), topk, pool_size))
        return torch.arange(11, dtype=torch.int32).expand(q.shape[0], 1, 11).clone(), None

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        torch.zeros(2, 16, 1, 128, dtype=torch.bfloat16),
        torch.tensor([[0], [1]], dtype=torch.int32),
        positions,
    )
    expected = torch.arange(11).expand(2, 4, 11).clone()
    expected[~mask] = -1
    torch.testing.assert_close(result, expected)
    assert len(calls) == 1
    tail, topk, pool_size = calls[0]
    assert (topk, pool_size) == (8, 4)
    torch.testing.assert_close(tail, torch.where(mask, positions + 1, 0).reshape(-1).remainder(4).int())


@torch.inference_mode()
def test_padding_without_tail_selection_stays_fused_with_zero_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default flag (``index_kpool_always_select_tail=False``) stays fused.

    The dense path marks the partial pool invalid (``read_pools``) and does not
    append the tail, so the fused call passes a zero tail and keeps the padded
    batch on the fast path instead of falling back to dense.
    """
    indexer = _make_indexer(torch.device("cpu"))
    indexer.index_kpool_always_select_tail = False
    query = torch.zeros(2, 1, 1, 128, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, 2, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, dtype=torch.bool)
    mask[1, 0] = False
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    lengths = torch.tensor([14, 14], dtype=torch.int32)
    calls = []

    def _kernel(
        q: torch.Tensor,
        cache: torch.Tensor,
        w: torch.Tensor,
        tail: torch.Tensor,
        topk: int,
        pool_size: int,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append((tail.clone(), topk, pool_size))
        return torch.arange(8, dtype=torch.int32).reshape(1, 1, 8), None

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        torch.zeros(8, 4, 1, 128, dtype=torch.bfloat16),
        tables,
    )
    assert result is not None, "padded batch without tail selection must stay fused"
    assert len(calls) == 1
    tail, topk, pool_size = calls[0]
    assert (topk, pool_size) == (8, 4)
    torch.testing.assert_close(tail, torch.zeros_like(tail))
    assert result.shape == (2, 1, 8)
    assert result[0, 0].tolist() == list(range(8))
    assert bool((result[1, 0] == -1).all().item())


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@torch.inference_mode()
def test_real_npu_padding_without_tail_selection_stays_fused(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real NPU: a padded row without tail selection keeps the complete pools."""
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    indexer = _make_indexer(device)
    indexer.index_kpool_always_select_tail = False

    query = torch.ones(2, 1, indexer.n_heads, indexer.head_dim, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, indexer.n_heads, device=device, dtype=torch.bfloat16)
    mask = torch.tensor([[True], [False]], device=device)
    lengths = torch.tensor([14, 14], device=device, dtype=torch.int32)
    cache = torch.zeros(2, 16, 1, indexer.head_dim, device=device, dtype=torch.bfloat16)
    pool_values = torch.arange(1, 5, device=device, dtype=torch.bfloat16).view(1, 4, 1, 1)
    cache[:, :4].copy_(pool_values.expand(2, -1, 1, indexer.head_dim))
    tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    result = indexer._select_topk_fused_pa(query, weights, lengths, mask, cache, tables)
    assert result is not None
    assert result.shape == (2, 1, indexer.topk)
    row = result[0, 0]
    valid = row[row >= 0]
    assert valid.numel() == indexer.topk
    assert not (row[: indexer.topk] < 0).any()
    # Three complete pools (tokens 0..11) are candidates; the two highest-scoring
    # ones are pools 2 and 1, returned in score order as tokens 8..11 and 4..7.
    # The partial tail (12, 13) is not selected because tail selection is disabled.
    torch.testing.assert_close(valid.sort().values.cpu(), torch.arange(4, 12, dtype=torch.int64))
    assert bool((result[1, 0] == -1).all().item())


def test_non_divisible_topk_falls_back_to_dense(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-divisible topk must not truncate ``topk // index_kpool``.

    The fused kernel cannot express it, so the indexer logs one warning and keeps
    every batch (padded or not) on the dense indexer instead of failing.
    """
    config = glm5_next.Glm5NextConfig(
        hidden_size=16,
        q_lora_rank=16,
        index_n_heads=2,
        index_head_dim=128,
        index_kpool=4,
        index_topk=6,
        index_kpool_compress=True,
    )
    monkeypatch.setattr(glm5_next, "_fused_kpool_warned", False)
    indexer = glm5_next.Glm5NextIndexer(config, 1, torch.bfloat16, torch.device("cpu"))
    # The shared logger emits the warning once; the module flag records that.
    assert glm5_next._fused_kpool_warned is True
    calls: list[bool] = []

    def _kernel(*args: object, **kwargs: object) -> None:
        calls.append(True)
        raise AssertionError("fused kernel must not run with a non-divisible topk")

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        torch.zeros(2, 1, 1, 128, dtype=torch.bfloat16),
        torch.ones(2, 1, 2, dtype=torch.bfloat16),
        torch.tensor([16, 16], dtype=torch.int32),
        torch.ones(2, 1, dtype=torch.bool),
        torch.zeros(8, 4, 1, 128, dtype=torch.bfloat16),
        torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )
    assert result is None
    assert not calls


@torch.inference_mode()
def test_short_rows_pack_tail_ahead_of_sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    """A short row must keep its tail tokens ahead of the ``-1`` holes.

    ``topk=8 / index_kpool=4`` requests two pools; a row with only one complete
    pool leaves the second slot ``-1``. The kernel's compact layout packs the
    visible tail before that hole, which is what SFA needs because it stops
    consuming a row at its first ``-1``.
    """
    indexer = _make_indexer(torch.device("cpu"))
    query = torch.zeros(2, 1, 1, 128, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, 2, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, dtype=torch.bool)
    mask[1, 0] = False  # padded batch -> compacted path
    lengths = torch.tensor([6, 8], dtype=torch.int32)
    tables = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)

    def _kernel(q: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        # One valid row after compaction: complete pool 0 (tokens 0..3), the
        # visible tail tokens 4 and 5, then the -1 sentinels.
        return torch.tensor([[[0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]]], dtype=torch.int32), None

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        torch.zeros(8, 4, 1, 128, dtype=torch.bfloat16),
        tables,
    )
    assert result is not None
    assert result.shape == (2, 1, 11)
    row = result[0, 0].tolist()
    # tokens 0..3 from the filled pool, then the visible tail tokens 4 and 5,
    # then the -1 sentinels.
    assert row == [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]
    first_negative = next(i for i, value in enumerate(row) if value < 0)
    assert all(value < 0 for value in row[first_negative:]), "no valid slot may follow the first -1"
    assert bool((result[1, 0] == -1).all().item())


def _short_row_cache(device: torch.device) -> torch.Tensor:
    """Pool cache where pool ``p`` token ``k`` carries ``p * kpool + k``.

    The pool buffer block size must be a multiple of 16 (kernel tiling rule), so
    only the first ``kpool`` slots of each pool carry tokens.
    """
    cache = torch.zeros(8, 16, 1, 128, device=device, dtype=torch.bfloat16)
    token_ids = torch.arange(8 * 4, device=device, dtype=torch.bfloat16).reshape(8, 4)
    cache[:, :4, 0, 0] = token_ids
    return cache


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@torch.inference_mode()
def test_real_npu_short_rows_pack_tail_before_sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Real NPU: a padded short row keeps its tail ahead of the ``-1`` holes.

    ``topk=8 / index_kpool=4`` asks for two pool ids.  A row of 6 visible tokens
    only has one complete pool, so the kernel leaves the second slot ``-1``; the
    partial tail (tokens 4, 5) must still be packed before that hole, otherwise
    SFA - which stops at the first ``-1`` - never attends the newest tokens.
    """
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    indexer = _make_indexer(device)
    query = torch.ones(3, 1, 2, 128, device=device, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, 2, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([6, 7, 9], device=device, dtype=torch.int32)
    mask = torch.ones(3, 1, device=device, dtype=torch.bool)
    mask[2, 0] = False  # padded batch -> compacted path
    cache = _short_row_cache(device)
    tables = torch.tensor([[0], [1], [2]], device=device, dtype=torch.int32)

    result = indexer._select_topk_fused_pa(query, weights, lengths, mask, cache, tables)
    assert result is not None
    assert result.shape == (3, 1, 11)
    torch.testing.assert_close(
        result[0, 0],
        torch.tensor([0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1], device=device, dtype=torch.long),
    )
    torch.testing.assert_close(
        result[1, 0],
        torch.tensor([0, 1, 2, 3, 4, 5, 6, -1, -1, -1, -1], device=device, dtype=torch.long),
    )
    assert bool((result[2, 0] == -1).all().item())
    # SFA contract: nothing valid may follow the first -1 of a row.
    for row in result[:, 0]:
        values = row.tolist()
        first_negative = next((i for i, value in enumerate(values) if value < 0), len(values))
        assert all(value < 0 for value in values[first_negative:])
    torch.npu.synchronize()


@pytest.mark.skipif(not os.getenv("XLLM_KDA_TEST_NPU_DEVICE"), reason="NPU device not configured")
@pytest.mark.parametrize("graph_mode", [False, True])
@torch.inference_mode()
def test_real_npu_graph_matches_eager_result(monkeypatch: pytest.MonkeyPatch, graph_mode: bool) -> None:
    """Real NPU: the ACL-graph path (``in_acl_graph``) matches eager results."""
    pytest.importorskip("torch_npu")
    from xllm.python.kernels_npu.sparse_attention import pool_key_indexer

    device = torch.device(os.environ["XLLM_KDA_TEST_NPU_DEVICE"])
    torch.npu.set_device(device)
    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", pool_key_indexer, raising=False)
    indexer = _make_indexer(device)
    query = torch.ones(2, 1, 2, 128, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 1, 2, device=device, dtype=torch.bfloat16)
    lengths = torch.tensor([6, 11], device=device, dtype=torch.int32)
    mask = torch.ones(2, 1, device=device, dtype=torch.bool)  # unpadded -> non-compacted path
    cache = _short_row_cache(device)
    tables = torch.tensor([[0], [1]], device=device, dtype=torch.int32)

    def _run() -> torch.Tensor:
        out = indexer._select_topk_fused_pa(query, weights, lengths, mask, cache, tables)
        assert out is not None
        return out

    eager = _run().clone()
    if not graph_mode:
        torch.testing.assert_close(_run(), eager)
        return

    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: True)
    for _ in range(3):
        _run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = _run()
    torch.npu.synchronize()
    lengths.sub_(4)  # 6 -> 2, 11 -> 7 visible tokens, replayed inside the graph
    graph.replay()
    torch.npu.synchronize()
    monkeypatch.setattr(glm5_next, "in_acl_graph", lambda: False)
    torch.testing.assert_close(captured, _run())


@torch.inference_mode()
def test_padded_batch_packs_only_the_rows_with_empty_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    """A mixed batch must keep the compact layout for every valid row.

    ``topk=8 / index_kpool=4`` asks for two pools. The 6-token row only has one
    complete pool, so its visible tail is packed before the ``-1`` hole; the
    40-token row fills both pools and has no tail.
    """
    indexer = _make_indexer(torch.device("cpu"))
    query = torch.zeros(3, 1, 1, 128, dtype=torch.bfloat16)
    weights = torch.ones(3, 1, 2, dtype=torch.bfloat16)
    mask = torch.ones(3, 1, dtype=torch.bool)
    mask[2, 0] = False  # padded batch -> compacted path
    lengths = torch.tensor([6, 40, 12], dtype=torch.int32)
    tables = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)

    def _kernel(q: torch.Tensor, *args: object, **kwargs: object) -> tuple[torch.Tensor, None]:
        # row 0 (6 tokens): one complete pool + tail tokens 4 and 5.
        # row 1 (40 tokens): two filled pools, no tail.
        return torch.tensor(
            [
                [[0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]],
                [[0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1]],
            ],
            dtype=torch.int32,
        ), None

    monkeypatch.setattr(glm5_next.kernels, "pool_key_indexer", _kernel, raising=False)
    result = indexer._select_topk_fused_pa(
        query,
        weights,
        lengths,
        mask,
        torch.zeros(8, 16, 1, 128, dtype=torch.bfloat16),
        tables,
    )
    assert result is not None
    assert result.shape == (3, 1, 11)
    # Hole row: tail (tokens 4, 5) packed ahead of the -1 run.
    assert result[0, 0].tolist() == [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1]
    # Full row: pools 0 and 1 expanded, no hole, no partial tail (40 % 4 == 0).
    assert result[1, 0].tolist() == [0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1]
    assert bool((result[2, 0] == -1).all().item())
