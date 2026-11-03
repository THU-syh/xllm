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

"""Tests for Python process-group rendezvous ownership."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist

from xllm.python.models import glm5_2

from .conftest import load_source_module

# The conftest stubs ``xllm.python.distributed``, so the real module is loaded
# from its file. The shared loader keeps this instance across every suite that
# needs it: re-executing the file would re-register the global ``xllm_ops::*``
# operators and void the monkeypatches made here.
_MODULE_PATH = Path(__file__).parents[2] / "xllm" / "python" / "distributed" / "collectives.py"
collectives = load_source_module(_MODULE_PATH, "_xllm_collectives_under_test")


def test_tp_all_to_all_uses_native_group_without_python_rendezvous(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def _exchange(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
        assert world_size == 2
        calls.append(tensor.clone())
        return tensor.flip(0)

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(tp_all_to_all=_exchange))
    value = torch.arange(12).reshape(4, 3)
    result = collectives.all_to_all(value, 2, "tp")
    torch.testing.assert_close(result, value.flip(0))
    assert len(calls) == 1
    torch.testing.assert_close(calls[0], value)


@pytest.mark.parametrize("native_present", (False, True))
def test_all_to_all_fallback_accepts_noncontiguous_input(monkeypatch: pytest.MonkeyPatch, native_present: bool) -> None:
    group = SimpleNamespace(size=lambda: 2)
    calls = []

    def _no_native_runtime(tensor: torch.Tensor, world_size: int) -> None:
        calls.append(world_size)

    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: _no_native_runtime if native_present else None)
    monkeypatch.setattr(collectives, "_require_group", lambda *_: group)
    value = torch.arange(24).reshape(3, 8).t()
    assert not value.is_contiguous()

    def _exchange(output: torch.Tensor, input_tensor: torch.Tensor, group: object) -> None:
        assert input_tensor.is_contiguous() and output.is_contiguous()
        output.copy_(input_tensor.flip(0))

    monkeypatch.setattr(dist, "all_to_all_single", _exchange)
    result = collectives.all_to_all(value, 2, "tp")
    torch.testing.assert_close(result, value.flip(0))
    assert calls == ([2] if native_present else [])


def test_all_to_all_fallback_rejects_world_size_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: None)
    monkeypatch.setattr(collectives, "_require_group", lambda *_: SimpleNamespace(size=lambda: 4))
    with pytest.raises(RuntimeError, match="world-size mismatch"):
        collectives.all_to_all(torch.zeros(8, 3), 2, "tp")


@pytest.mark.parametrize("world_size", (1, 2))
def test_all_to_all_fake_matches_real_noncontiguous_strides(monkeypatch: pytest.MonkeyPatch, world_size: int) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    monkeypatch.setattr(collectives, "_native_runtime_op", lambda *_: None)
    monkeypatch.setattr(collectives, "_require_group", lambda *_: SimpleNamespace(size=lambda: world_size))
    monkeypatch.setattr(dist, "all_to_all_single", lambda output, input_tensor, group: output.copy_(input_tensor))
    value = torch.arange(24).reshape(3, 8).t()
    actual = collectives.all_to_all(value, world_size, "tp")
    with FakeTensorMode() as mode:
        fake = collectives.all_to_all(mode.from_tensor(value), world_size, "tp")
    assert fake.shape == actual.shape
    assert fake.dtype == actual.dtype
    assert fake.device == actual.device
    assert fake.stride() == actual.stride()
    assert actual.is_contiguous() == (world_size > 1)


class _FakeGroup:
    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_moe_ep_hccl_info_uses_existing_group() -> None:
    queried = []
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: queried.append(rank) or "ep_test_group")
    group = SimpleNamespace(rank=lambda: 1, size=lambda: 2, _get_backend=lambda device: backend)
    collectives._groups[("moe_ep", "cpu")] = group
    assert collectives.moe_ep_hccl_info("cpu") == ("ep_test_group", 1, 2)
    assert queried == [1]


def test_moe_ep_hccl_info_rejects_missing_group() -> None:
    with pytest.raises(RuntimeError, match="initialized process group"):
        collectives.moe_ep_hccl_info("cpu")


@pytest.mark.parametrize("comm_name", ["", None, 1])
def test_moe_ep_hccl_info_rejects_invalid_name(comm_name: object) -> None:
    backend = SimpleNamespace(get_hccl_comm_name=lambda rank: comm_name)
    collectives._groups[("moe_ep", "cpu")] = SimpleNamespace(
        rank=lambda: 0, size=lambda: 2, _get_backend=lambda device: backend
    )
    with pytest.raises(RuntimeError, match="nonempty HCCL"):
        collectives.moe_ep_hccl_info("cpu")


def test_moe_ep_hccl_info_rejects_non_hccl_group() -> None:
    collectives._groups[("moe_ep", "cpu")] = SimpleNamespace(
        rank=lambda: 0, size=lambda: 2, _get_backend=lambda device: object()
    )
    with pytest.raises(RuntimeError, match="HCCL process group"):
        collectives.moe_ep_hccl_info("cpu")


@pytest.fixture(autouse=True)
def _clear_collective_state():
    def reset():
        collectives._groups.clear()
        collectives._group_ranks.clear()
        collectives._stores.clear()
        collectives._dp_host_steps.clear()
        collectives._symm_eligible.clear()
        collectives._symm_buffers.clear()
        collectives._moe_ep_comm_name_cache.clear()
        collectives._world_topology = None
        collectives._world_initialized = False

    reset()
    yield
    reset()


class _FakeStore:
    def __init__(self, topology: list[dict[str, object]] | None = None) -> None:
        self.values: dict[str, bytes] = {}
        if topology is not None:
            for rank, entry in enumerate(topology):
                self.values[f"xllm/python_collectives/topology/v1/{rank}"] = json.dumps(entry).encode("utf-8")

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.encode("utf-8")

    def get(self, key: str) -> bytes:
        return self.values[key]


def _mock_process_groups(
    monkeypatch: pytest.MonkeyPatch,
    global_rank: int,
    topology: list[dict[str, object]] | None = None,
):
    """Stand in for c10d so the rendezvous can be inspected without a world.

    ``new_group`` reports this rank's position inside the membership it is
    handed, which is what the module checks its caller's rank against.
    """
    if topology is None:
        topology = [{"hostname": "node-0", "device_index": rank} for rank in range(16)]
    base_store = _FakeStore(topology)
    tcp_store = MagicMock(return_value=base_store)
    init_world = MagicMock()
    new_group = MagicMock(
        side_effect=lambda ranks, timeout, backend: _FakeGroup(
            ranks.index(global_rank) if global_rank in ranks else -1, len(ranks)
        )
    )
    monkeypatch.setattr(dist, "TCPStore", tcp_store)
    monkeypatch.setattr(dist, "init_process_group", init_world)
    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(collectives.socket, "gethostname", lambda: "node-0")
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", lambda _a, _b: True)
    return base_store, tcp_store, init_world, new_group


def test_tp_group_reuses_initialized_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=2)

    first = collectives.init_process_group("tp", "host", 1234, 0, 2, "cuda:2", 2, 4, 1)
    second = collectives.init_process_group("tp", "host", 1234, 0, 2, "cuda:2", 2, 4, 1)

    assert first is second
    assert collectives.tp_group("cuda:2") is first
    assert collectives._group_ranks[("tp", "cuda:2")] == (2, 3)
    init_world.assert_called_once()
    assert new_group.call_count == 2


def test_tp_group_requires_initialized_group() -> None:
    with pytest.raises(RuntimeError, match="TP group is not initialized"):
        collectives.tp_group("cpu")


@pytest.mark.parametrize("global_rank,root_rank", [(0, 0), (1, 0), (2, 2), (3, 2)])
def test_tp_broadcast_stays_within_each_dp_replica(
    monkeypatch: pytest.MonkeyPatch,
    global_rank: int,
    root_rank: int,
) -> None:
    _mock_process_groups(monkeypatch, global_rank=global_rank)
    monkeypatch.setattr(collectives, "_backend_for", lambda device: "gloo")
    group = collectives.init_tp_group(
        "host",
        1234,
        global_rank % 2,
        2,
        "cpu",
        global_rank,
        4,
        global_rank // 2,
    )
    native_op = MagicMock(side_effect=AssertionError("Broadcast must use the Python group"))
    monkeypatch.setattr(collectives, "_native_runtime_op", native_op)
    broadcast = MagicMock()
    monkeypatch.setattr(dist, "broadcast", broadcast)
    tokens = torch.tensor([2, 3])

    collectives.broadcast_(tokens, src=0, group_name="tp")

    assert group.size() == 2
    assert collectives._group_ranks[("tp", "cpu")] == (root_rank, root_rank + 1)
    broadcast.assert_called_once_with(tokens, src=root_rank, group=group)
    native_op.assert_not_called()


@pytest.mark.parametrize("allowed", (False, True))
def test_dp_all_true_reuses_host_store_and_retires_step(
    monkeypatch: pytest.MonkeyPatch,
    allowed: bool,
) -> None:
    store = MagicMock()
    store.multi_get.return_value = [b"1", b"1" if allowed else b"0"]
    store.add.return_value = 2
    collectives._group_ranks[("dp", "npu:0")] = (1, 3)
    collectives._stores[("host", 1234)] = store
    monkeypatch.setattr(collectives, "_world_endpoint", ("host", 1234))
    monkeypatch.setattr(collectives, "_world_rank", 1)
    create_group = MagicMock(side_effect=AssertionError("unexpected process group creation"))
    monkeypatch.setattr(dist, "new_group", create_group)

    for step in range(2):
        assert collectives.dp_all_true(True, "npu:0", 2) == allowed
        prefix = f"xllm/python_collectives/dp_admission/1,3/{step}"
        store.set.assert_called_with(f"{prefix}/1", b"1")
        store.multi_get.assert_called_with([f"{prefix}/1", f"{prefix}/3"])
        store.add.assert_called_with(f"{prefix}/readers", 1)
        assert store.delete_key.call_count == 3 * (step + 1)
    create_group.assert_not_called()


def test_dp_all_true_requires_complete_dp_membership() -> None:
    with pytest.raises(RuntimeError, match="initialized 2-rank DP group"):
        collectives.dp_all_true(True, "cpu", 2)


def _run_dp_host_agreement(global_rank: int, port: int) -> None:
    endpoint = ("127.0.0.1", port)
    store = dist.TCPStore(*endpoint, None, False, timedelta(seconds=20))
    collectives._stores[endpoint] = store
    collectives._world_endpoint = endpoint
    collectives._world_rank = global_rank
    collectives._group_ranks[("dp", "cpu")] = (global_rank % 2, global_rank % 2 + 2)
    for step in range(20):
        local_allowed = global_rank != 2 or step % 2 == 0
        expected = global_rank % 2 == 1 or step % 2 == 0
        assert collectives.dp_all_true(local_allowed, "cpu", 2) == expected
        if global_rank == 0:
            time.sleep(0.001)


def test_dp_host_agreement_across_processes() -> None:
    store = dist.TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=20))
    initial_keys = store.num_keys()
    torch.multiprocessing.start_processes(_run_dp_host_agreement, args=(store.port,), nprocs=4, start_method="fork")
    assert store.num_keys() == initial_keys


def test_tp_group_does_not_reuse_dp_group() -> None:
    collectives._groups[("dp", "cpu")] = _FakeGroup(rank=0, size=2)
    collectives._group_ranks[("dp", "cpu")] = (0, 2)
    with pytest.raises(RuntimeError, match="TP group is not initialized"):
        collectives.tp_group("cpu")


def _run_glm_ep1_tp_collective(global_rank: int, rendezvous_path: str) -> None:
    world_size = 4
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        tp_groups = [
            dist.new_group(
                ranks=[0, 1],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
            dist.new_group(
                ranks=[2, 3],
                backend="gloo",
                timeout=timedelta(seconds=20),
            ),
        ]
        collectives._groups[("tp", "cpu")] = tp_groups[global_rank // 2]
        # This is the topology that exposed the bug: with EP1, moe_tp spans
        # both CP cohorts and must not be used to combine expert partials.
        collectives._groups[("moe_tp", "cpu")] = dist.group.WORLD

        cp_rank = global_rank // 2
        tp_rank = global_rank % 2
        local_value = float(cp_rank * 10 + tp_rank + 1)
        routed = torch.tensor([[local_value]])
        shared = torch.tensor([[local_value * 10]])
        moe = SimpleNamespace(
            ep_size=1,
            moe_tp_size=world_size,
            cfg=SimpleNamespace(tp_size=2),
        )

        with patch.object(
            glm5_2.distributed,
            "all_reduce_",
            collectives.all_reduce_,
            create=True,
        ):
            output = glm5_2.Glm52MoE._combine_expert_outputs(moe, routed, shared, False)

        expected = torch.tensor([[33.0 if cp_rank == 0 else 253.0]])
        torch.testing.assert_close(output, expected)
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_glm53_dp_cp_ep_collectives(global_rank: int, rendezvous_path: str) -> None:
    world_size = 8
    dp_rank, local_rank = divmod(global_rank, 4)
    cp_rank, tp_rank = divmod(local_rank, 2)
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous_path}",
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(seconds=20),
        )
        for name, size, stride, expected_rank in (
            ("tp", 2, None, tp_rank),
            ("dp", 2, None, dp_rank),
            ("moe_ep", world_size, None, global_rank),
            ("cp", 2, 2, cp_rank),
        ):
            memberships = collectives._group_memberships(name, size, world_size, stride)
            for index, ranks in enumerate(memberships):
                group = dist.new_group(ranks=ranks, backend="gloo", timeout=timedelta(seconds=20))
                if global_rank in ranks:
                    assert group.rank() == expected_rank
                    if name == "cp":
                        assert index == dp_rank * 2 + tp_rank
                    collectives._groups[(name, "cpu")] = group

        cp_rows = collectives.all_gather(torch.tensor([[float(global_rank)]]), 0, 2, "cp")
        expected_cp = torch.tensor([[float(dp_rank * 4 + tp_rank)], [float(dp_rank * 4 + 2 + tp_rank)]])
        torch.testing.assert_close(cp_rows, expected_cp)

        local_rows = torch.full((1 if dp_rank == 0 else 3, 1), float(global_rank))
        gathered, offset = collectives.gather_dp_execution_tokens(local_rows, (1, 3), dp_rank)
        expected_dp = torch.tensor([float(local_rank)] + [float(local_rank + 4)] * 3).unsqueeze(-1)
        assert offset == dp_rank
        torch.testing.assert_close(gathered, expected_dp)

        expert_partial = torch.tensor([[float(global_rank + 1)]])
        collectives.moe_ep_all_reduce(expert_partial)
        torch.testing.assert_close(expert_partial, torch.tensor([[36.0]]))
    finally:
        collectives._groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()


def test_parallel_groups_share_one_multitenant_tcp_store(monkeypatch):
    base_store, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=0)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)
    collectives.init_process_group("moe_tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 2, 0)

    tcp_store.assert_called_once()
    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 2, True)
    assert tcp_store.call_args.kwargs["wait_for_workers"] is False
    assert tcp_store.call_args.kwargs["multi_tenant"] is True

    # Every parallel group is a subgroup of one world, so the world rendezvous
    # happens once no matter how many groups the caller asks for.
    init_world.assert_called_once()
    assert init_world.call_args.kwargs["store"] is base_store
    assert init_world.call_args.kwargs["rank"] == 0
    assert init_world.call_args.kwargs["world_size"] == 2
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 1],
        [0, 1],
    ]


def test_native_runtime_bridge_bypasses_python_process_groups(monkeypatch):
    calls: list[str] = []

    runtime = SimpleNamespace(
        tp_all_reduce=lambda tensor: (calls.append("tp_reduce"), tensor.add_(1)),
        tp_all_gather=lambda tensor, dim: (
            calls.append(f"tp_gather:{dim}"),
            torch.cat((tensor, tensor), dim=dim),
        )[1],
        dp_all_gather=lambda tensor, counts: (
            calls.append(f"dp_gather:{counts}"),
            torch.cat((tensor, tensor), dim=0),
        )[1],
        moe_tp_all_reduce=lambda tensor: (
            calls.append("moe_tp_reduce"),
            tensor.add_(2),
        ),
        moe_ep_all_reduce=lambda tensor: (
            calls.append("moe_ep_reduce"),
            tensor.add_(4),
        ),
    )
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    python_reduce = MagicMock(side_effect=AssertionError("c10d fallback used"))
    python_gather = MagicMock(side_effect=AssertionError("c10d fallback used"))
    monkeypatch.setattr(collectives, "all_reduce_", python_reduce)
    monkeypatch.setattr(collectives, "all_gather", python_gather)

    value = torch.tensor([[1.0]])
    collectives.tp_all_reduce(value)
    gathered = collectives.tp_all_gather(value, 1, 2)
    dp_gathered, dp_offset = collectives.gather_dp_execution_tokens(
        value,
        (1, 1),
        rank=1,
    )
    collectives.moe_tp_all_reduce(value)
    collectives.moe_ep_all_reduce(value)

    assert calls == [
        "tp_reduce",
        "tp_gather:1",
        "dp_gather:[1, 1]",
        "moe_tp_reduce",
        "moe_ep_reduce",
    ]
    assert gathered.tolist() == [[2.0, 2.0]]
    assert dp_gathered.tolist() == [[2.0], [2.0]]
    assert dp_offset == 1
    assert value.tolist() == [[8.0]]
    python_reduce.assert_not_called()
    python_gather.assert_not_called()


def _run_gloo_workers(worker: Callable[[int, str], None], nprocs: int, rendezvous_path: Path) -> None:
    process_context = torch.multiprocessing.start_processes(
        worker,
        args=(str(rendezvous_path),),
        nprocs=nprocs,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 30.0
    try:
        while not process_context.join(
            timeout=max(0.0, deadline - time.monotonic()),
            grace_period=5.0,
        ):
            if time.monotonic() >= deadline:
                pytest.fail(f"{worker.__name__} Gloo collective test timed out")
    finally:
        for process in process_context.processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.monotonic() + 5.0
        for process in process_context.processes:
            process.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        for process in process_context.processes:
            if process.is_alive():
                process.kill()
                process.join(timeout=5.0)


def test_moe_ep_comm_name_prefers_native_runtime(monkeypatch):
    """The C++-held EP group wins over any Python-created one."""
    native_calls: list[str] = []

    def native_comm_name() -> str:
        native_calls.append("native")
        return "hccl-native"

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_comm_name=native_comm_name))
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(0, 2)

    assert collectives.moe_ep_comm_name("cpu") == "hccl-native"
    assert collectives.moe_ep_comm_name("cpu") == "hccl-native"
    # Cached after the first probe.
    assert native_calls == ["native"]


def test_moe_ep_comm_name_falls_back_to_python_group(monkeypatch):
    """Without the native binding, the Python-created HCCL group is used."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", None)
    monkeypatch.setattr(collectives, "_moe_ep_backend_device", lambda: torch.device("cpu"))
    backend_calls: list[int] = []
    group = _FakeGroup(1, 2)
    group._get_backend = lambda device: SimpleNamespace(
        get_hccl_comm_name=lambda rank: (backend_calls.append(rank), f"hccl-{rank}")[1]
    )
    collectives._groups[("moe_ep", "cpu")] = group

    assert collectives.moe_ep_comm_name("cpu") == "hccl-1"
    assert collectives.moe_ep_comm_name("cpu") == "hccl-1"
    # Cached: the communicator is built once, not once per layer/step.
    assert backend_calls == [1]


def test_moe_ep_comm_name_returns_none_without_any_group(monkeypatch):
    monkeypatch.setitem(sys.modules, "xllm_runtime", None)
    monkeypatch.setattr(collectives, "_moe_ep_backend_device", lambda: torch.device("cpu"))

    assert collectives.moe_ep_comm_name("cpu") is None


@pytest.mark.parametrize(
    "runtime",
    [None, SimpleNamespace(moe_ep_comm_name=lambda: "")],
    ids=["absent", "empty"],
)
def test_moe_ep_comm_name_ignores_unusable_native_binding(monkeypatch, runtime):
    """A missing binding, or one that reports no name, falls through."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    monkeypatch.setattr(collectives, "_moe_ep_backend_device", lambda: torch.device("cpu"))
    group = _FakeGroup(0, 2)
    group._get_backend = lambda device: SimpleNamespace(get_hccl_comm_name=lambda rank: "hccl-python")
    collectives._groups[("moe_ep", "cpu")] = group

    assert collectives.moe_ep_comm_name("cpu") == "hccl-python"


def test_moe_ep_comm_name_falls_back_when_native_raises(monkeypatch):
    def explode() -> str:
        raise RuntimeError("stale binding")

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_comm_name=explode))
    monkeypatch.setattr(collectives, "_moe_ep_backend_device", lambda: torch.device("cpu"))

    assert collectives.moe_ep_comm_name("cpu") is None


def test_moe_ep_rank_prefers_native_runtime(monkeypatch):
    """The C++-held EP group's rank wins over any Python-created one."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_rank=lambda: 3))
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(1, 4)

    assert collectives.moe_ep_rank("cpu") == 3


def test_moe_ep_rank_falls_back_to_python_group(monkeypatch):
    """Without the native binding, the Python-created group's rank is used."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", None)
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(2, 4)

    assert collectives.moe_ep_rank("cpu") == 2


def test_moe_ep_rank_returns_zero_without_any_group(monkeypatch):
    monkeypatch.setitem(sys.modules, "xllm_runtime", None)

    assert collectives.moe_ep_rank("cpu") == 0


@pytest.mark.parametrize(
    "runtime",
    [None, SimpleNamespace(), SimpleNamespace(moe_ep_rank=lambda: None)],
    ids=["absent", "missing-attr", "null"],
)
def test_moe_ep_rank_ignores_unusable_native_binding(monkeypatch, runtime):
    """A missing binding, or one that reports no rank, falls through safely."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", runtime)
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(1, 4)

    assert collectives.moe_ep_rank("cpu") == 1


def test_moe_ep_rank_falls_back_when_native_raises(monkeypatch):
    """A present-but-broken native binding must not raise; fall back to the group."""

    def explode() -> int:
        raise RuntimeError("stale binding")

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_rank=explode))
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(2, 4)

    assert collectives.moe_ep_rank("cpu") == 2


def test_moe_ep_rank_falls_back_when_native_raises_without_group(monkeypatch):
    """A broken native binding with no Python group resolves to the 0 fallback."""

    def explode() -> int:
        raise RuntimeError("stale binding")

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_rank=explode))

    assert collectives.moe_ep_rank("cpu") == 0


def test_moe_ep_rank_from_native_returns_native_value(monkeypatch):
    """The probe reports the native rank, including a legitimate rank 0."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_rank=lambda: 0))

    assert collectives.moe_ep_rank_from_native("cpu") == 0


def test_moe_ep_rank_from_native_rejects_python_group_fallback(monkeypatch):
    """Without the native binding the probe is ``None`` -- it must not leak the
    ``_groups`` rank that ``moe_ep_rank`` would silently return."""
    monkeypatch.setitem(sys.modules, "xllm_runtime", None)
    collectives._groups[("moe_ep", "cpu")] = _FakeGroup(2, 4)

    assert collectives.moe_ep_rank("cpu") == 2
    assert collectives.moe_ep_rank_from_native("cpu") is None


def test_moe_ep_rank_from_native_none_when_native_raises(monkeypatch):
    def explode() -> int:
        raise RuntimeError("stale binding")

    monkeypatch.setitem(sys.modules, "xllm_runtime", SimpleNamespace(moe_ep_rank=explode))

    assert collectives.moe_ep_rank_from_native("cpu") is None


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm_ep1_tp_reduce_does_not_mix_cp_cohorts(tmp_path: Path) -> None:
    _run_gloo_workers(_run_glm_ep1_tp_collective, 4, tmp_path / "glm-ep1-tp-reduce")


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo backend is unavailable")
def test_glm53_dp2_cp2_tp2_ep8_collectives_keep_dp_cohorts(tmp_path: Path) -> None:
    _run_gloo_workers(_run_glm53_dp_cp_ep_collectives, 8, tmp_path / "glm53-dp2-cp2-tp2-ep8")


def test_dcp_group_is_strided_like_kv_split_rank(monkeypatch):
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)

    # world=8, dcp=2 → group_count=4; membership matches rank/(world/dcp).
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ]


@pytest.mark.parametrize(
    "world, cp, tp, expected",
    [
        (4, 2, 1, [[0, 1], [2, 3]]),
        (8, 2, 2, [[0, 2], [1, 3], [4, 6], [5, 7]]),
        (8, 4, 1, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        (4, 2, 2, [[0, 2], [1, 3]]),
    ],
)
def test_cp_memberships_preserve_dp_cohorts(world: int, cp: int, tp: int, expected: list[list[int]]) -> None:
    assert collectives._group_memberships("cp", cp, world, tp) == expected


@pytest.mark.parametrize("stride", [0, -1, 3])
def test_cp_memberships_reject_invalid_cohorts(stride: int) -> None:
    with pytest.raises(ValueError, match="complete DP cohorts"):
        collectives._group_memberships("cp", 2, 8, stride)


# ``_engram_dp_memberships`` slices each (strided) DP membership into consecutive
# ``shard_size`` blocks, so a sharing group is always a whole node-local block
# that maps one TP shard. These pin the exact partition, the invariants it must
# keep, and the invalid shard sizes it must reject -- the function itself is only
# otherwise exercised indirectly through ``init_engram_dp_group``.
@pytest.mark.parametrize(
    "dp_size, shard_size, world, expected",
    [
        # world=8, dp=4 -> DP memberships [[0, 2, 4, 6], [1, 3, 5, 7]]; shards of 2.
        (4, 2, 8, [[0, 2], [4, 6], [1, 3], [5, 7]]),
        # shard_size == dp_size -> each DP membership is one sharing group.
        (4, 4, 8, [[0, 2, 4, 6], [1, 3, 5, 7]]),
        # world=8, dp=8 -> one DP membership [0..7]; shards of 4.
        (8, 4, 8, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        # world=8, dp=2 -> DP memberships [[0, 4], [1, 5], [2, 6], [3, 7]].
        (2, 2, 8, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        # world=16, dp=4 -> count=4, then shards of 2 across each cohort.
        (4, 2, 16, [[0, 4], [8, 12], [1, 5], [9, 13], [2, 6], [10, 14], [3, 7], [11, 15]]),
    ],
)
def test_engram_dp_memberships_partition_dp_cohorts(
    dp_size: int, shard_size: int, world: int, expected: list[list[int]]
) -> None:
    assert collectives._engram_dp_memberships(dp_size, shard_size, world) == expected


@pytest.mark.parametrize(
    "dp_size, shard_size, world",
    [
        (4, 2, 8),
        (8, 4, 8),
        (2, 1, 8),
        (4, 4, 16),
        (8, 2, 16),
        (6, 3, 12),
    ],
)
def test_engram_dp_memberships_are_increasing_shards_of_one_dp_cohort(
    dp_size: int, shard_size: int, world: int
) -> None:
    """Every group is a full, increasing, node-local shard of exactly one DP cohort."""
    memberships = collectives._engram_dp_memberships(dp_size, shard_size, world)
    dp_cohorts = collectives._group_memberships("dp", dp_size, world)

    # The groups partition all global ranks exactly once, each a full shard.
    assert len(memberships) * shard_size == world
    assert sorted(rank for group in memberships for rank in group) == list(range(world))
    assert all(len(group) == shard_size for group in memberships)
    # Increasing (new_group requires a sorted rank list) ...
    assert all(group == sorted(group) for group in memberships)
    # ... and each group lies inside exactly one DP membership (node-local + one
    # TP shard); it never straddles two cohorts.
    for group in memberships:
        assert sum(set(group) <= set(cohort) for cohort in dp_cohorts) == 1


@pytest.mark.parametrize("dp_size, shard_size, world", [(4, 3, 8), (4, 0, 8), (4, -1, 8), (6, 4, 12)])
def test_engram_dp_memberships_reject_bad_shard_size(dp_size: int, shard_size: int, world: int) -> None:
    with pytest.raises(ValueError, match="must divide DP size"):
        collectives._engram_dp_memberships(dp_size, shard_size, world)


@pytest.mark.parametrize("dp_size, world", [(3, 8), (5, 12)])
def test_engram_dp_memberships_reject_dp_that_does_not_tile_the_world(dp_size: int, world: int) -> None:
    with pytest.raises(ValueError, match="does not divide the world size"):
        collectives._engram_dp_memberships(dp_size, 1, world)


@pytest.mark.parametrize(
    "world_size, dp_size, replica_size, expected_shard_size",
    [
        (8, 4, 2, 2),
        (16, 8, 2, 4),
        (8, 8, 1, 4),
        (16, 4, 4, 2),
    ],
)
def test_engram_dp_memberships_agree_with_the_topology_shard_size(
    world_size: int, dp_size: int, replica_size: int, expected_shard_size: int
) -> None:
    """The shard size the topology gate returns must yield node-local groups.

    ``init_engram_dp_group`` feeds ``get_engram_dp_shard_size``'s result straight
    into ``_engram_dp_memberships`` and then re-checks that no group spans hosts,
    so the two must line up: for a topology the gate accepted, every group the
    shard size produces is on a single host. Ties the otherwise-separately-tested
    size function to the membership slicer it drives, and pins the exact shard
    size instead of accepting the 1 (no sharing) fallback as a pass.
    """
    hosts = ["node-0"] * (world_size // 2) + ["node-1"] * (world_size // 2)
    topology = [{"hostname": host} for host in hosts]
    shard_size = collectives.get_engram_dp_shard_size(world_size, dp_size, replica_size, topology)
    assert shard_size == expected_shard_size

    memberships = collectives._engram_dp_memberships(dp_size, shard_size, world_size)
    for group in memberships:
        assert len({topology[rank]["hostname"] for rank in group}) == 1


def _engram_topology(hosts: str | None) -> list[dict[str, object]] | None:
    """Per-global-rank topology, one hostname character per rank (``None`` passes through)."""
    if hosts is None:
        return None
    return [{"hostname": host} for host in hosts]


# ``get_engram_dp_shard_size`` is a pure function with many "decline to share"
# fallbacks (return 1). These pin every one of them plus the accepted layouts so
# a regression cannot silently turn a real shard size into 1 -- or worse, 1 into
# a bogus sharing group -- on the host that matters.
@pytest.mark.parametrize(
    "world_size, dp_size, replica_size, hosts, expected",
    [
        # No topology at all: nothing to prove node locality with.
        (16, 4, 4, None, 1),
        # Non-positive world / DP / replica sizes cannot describe a split.
        (0, 4, 4, "", 1),
        (16, 0, 4, "a" * 16, 1),
        (16, -4, 4, "a" * 16, 1),
        (16, 4, 0, "a" * 16, 1),
        (16, 4, -2, "a" * 16, 1),
        # Topology shorter than the world it is supposed to describe.
        (16, 4, 4, "a" * 8, 1),
        # 6 ranks over 2 nodes of 2 replicas x 2: does not tile whole-node blocks.
        (6, 3, 2, "aaabbb", 1),
        # gcd(dp, replicas_per_node) == 1 -> no sharing is possible.
        (8, 4, 2, "aabbccdd", 1),
        # One node: every DP rank already shares host memory.
        (16, 4, 4, "a" * 16, 4),
        # Two nodes of 8: shard size is gcd(4, 2) == 2.
        (16, 4, 4, "a" * 8 + "b" * 8, 2),
        # Non-power-of-two: world 12 over 2 nodes of 6, gcd(6, 3) == 3.
        (12, 6, 2, "a" * 6 + "b" * 6, 3),
        # Interleaved hosts: each node's ranks are not a contiguous block.
        (8, 4, 2, "abababab", 1),
        # Uneven node sizes: the second node is short, so ranks do not tile.
        (8, 4, 2, "aaaaaabb", 1),
    ],
)
def test_get_engram_dp_shard_size_accepts_and_declines(
    world_size: int, dp_size: int, replica_size: int, hosts: str | None, expected: int
) -> None:
    topology = _engram_topology(hosts)
    assert collectives.get_engram_dp_shard_size(world_size, dp_size, replica_size, topology) == expected


@pytest.mark.parametrize("hostname", [None, "", 7, 1.5, ["a"]])
def test_get_engram_dp_shard_size_rejects_nonstring_hostname(hostname: object) -> None:
    """A bad hostname must not be trusted to prove node locality."""
    topology: list[dict[str, object]] = [{"hostname": "a"} for _ in range(8)]
    topology[3] = {"hostname": hostname}
    assert collectives.get_engram_dp_shard_size(8, 4, 2, topology) == 1


def test_get_engram_dp_shard_size_rejects_missing_hostname() -> None:
    topology: list[dict[str, object]] = [{"hostname": "a"} for _ in range(8)]
    topology[3] = {}  # no "hostname" key at all
    assert collectives.get_engram_dp_shard_size(8, 4, 2, topology) == 1


def test_cp_group_initialization_uses_tp_stride(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    group = collectives.init_process_group("cp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, 1)
    assert group.rank() == 1
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 1], [2, 3]]


def test_dcp_group_initialization_stays_within_dp(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=5)
    group = collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 5, 8, 3, 2)
    assert group.rank() == 0
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_materializes_strided_activations(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1)
    assert not value.is_contiguous()
    group = _FakeGroup(0, 2)
    monkeypatch.setattr(collectives, "_require_group", lambda value, name: group)

    def gather(chunks: list[torch.Tensor], tensor: torch.Tensor, group: object) -> None:
        assert tensor.is_contiguous()
        assert all(chunk.is_contiguous() for chunk in chunks)
        torch.testing.assert_close(tensor, value)
        chunks[0].copy_(tensor)
        chunks[1].copy_(tensor + 100)

    def gather_rows(output: torch.Tensor, tensor: torch.Tensor, group: object) -> None:
        assert tensor.is_contiguous()
        assert output.is_contiguous()
        torch.testing.assert_close(tensor, value)
        output.copy_(torch.cat((tensor, tensor + 100), dim=0))

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_gather_into_tensor", gather_rows)
    actual = collectives.all_gather(value, dim, 2, "cp")
    torch.testing.assert_close(actual, torch.cat((value, value + 100), dim=dim))
    assert not value.is_contiguous()


@pytest.mark.parametrize("dim", [0, 1])
def test_all_gather_keeps_contiguous_input_buffer(monkeypatch: pytest.MonkeyPatch, dim: int) -> None:
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    group = _FakeGroup(0, 2)
    monkeypatch.setattr(collectives, "_require_group", lambda tensor, name: group)

    def gather(chunks: list[torch.Tensor], tensor: torch.Tensor, group: object) -> None:
        assert tensor.data_ptr() == value.data_ptr()
        for index, chunk in enumerate(chunks):
            chunk.copy_(tensor + 100 * index)

    def gather_rows(output: torch.Tensor, tensor: torch.Tensor, group: object) -> None:
        assert tensor.data_ptr() == value.data_ptr()
        output.copy_(torch.cat((tensor, tensor + 100)))

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_gather_into_tensor", gather_rows)
    result = collectives.all_gather(value, dim, 2, "cp")
    torch.testing.assert_close(result, torch.cat((value, value + 100), dim=dim))


@pytest.mark.parametrize("global_rank", range(8))
def test_dcp_group_consumes_upstream_qwen_memberships(monkeypatch: pytest.MonkeyPatch, global_rank: int) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=global_rank)
    memberships = [[0, 1], [2, 3], [4, 5], [6, 7]]
    rank, group_index = global_rank % 2, global_rank // 2

    # The C++ bridge passes the new optional argument positionally.
    group = collectives.init_process_group(
        "dcp",
        "127.0.0.1",
        46001,
        rank,
        2,
        "cuda:0",
        global_rank,
        8,
        group_index,
        group_ranks=memberships,
    )

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    own_ranks = collectives._group_ranks[("dcp", "cuda:0")]
    assert own_ranks[group.rank()] == global_rank
    assert group.rank() == global_rank % 2
    assert collectives.dcp_group("cuda:0") is group


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
@pytest.mark.parametrize("pass_none", [False, True])
def test_default_topologies_remain_compatible(
    monkeypatch: pytest.MonkeyPatch, group_name: str, pass_none: bool
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    contiguous = group_name in ("tp", "moe_tp", "layerwise")
    rank, group_index = (1, 1) if contiguous else (0, 3)
    kwargs = {"group_ranks": None} if pass_none else {}
    groups = [
        collectives.init_process_group(group_name, "127.0.0.1", 46001, rank, 2, "cuda:0", 3, 8, group_index, **kwargs)
        for _ in range(2)
    ]

    assert groups[0] is groups[1]
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == (
        [[0, 1], [2, 3], [4, 5], [6, 7]] if contiguous else [[0, 4], [1, 5], [2, 6], [3, 7]]
    )


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
def test_explicit_topology_overrides_defaults_and_preserves_group_order(
    monkeypatch: pytest.MonkeyPatch, group_name: str
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    # Neither built-in layout has this group order. Consume it without deriving.
    memberships = [[2, 3], [0, 1], [6, 7], [4, 5]]
    derive_memberships = MagicMock(side_effect=AssertionError("topology was re-derived"))
    monkeypatch.setattr(collectives, "_group_memberships", derive_memberships)

    groups = [
        collectives.init_process_group(group_name, "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 8, 0, group_ranks=memberships)
        for _ in range(2)
    ]

    assert groups[0] is groups[1]
    assert groups[0].rank() == 1
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert collectives._group_ranks[(group_name, "cuda:0")] == (2, 3)
    derive_memberships.assert_not_called()


def test_tp_accepts_explicit_strided_memberships(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=5)
    memberships = [[0, 4], [1, 5], [2, 6], [3, 7]]

    group = collectives.init_process_group("tp", "127.0.0.1", 46001, 1, 2, "cuda:0", 5, 8, 1, group_ranks=memberships)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert group.rank() == 1
    assert collectives._group_ranks[("tp", "cuda:0")] == (1, 5)


def test_dcp_groups_preserve_upstream_data_parallel_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=10)
    memberships = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]

    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:0", 10, 16, 5, group_ranks=memberships)

    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == memberships
    assert collectives._group_ranks[("dcp", "cuda:0")] == (10, 11)


def test_explicit_topology_does_not_affect_other_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=2)
    collectives.init_process_group(
        "dcp",
        "127.0.0.1",
        46001,
        0,
        2,
        "cuda:0",
        2,
        8,
        1,
        group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
    )
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 1], [2, 3], [4, 5], [6, 7]]
    new_group.reset_mock()

    # Another group and another device still use the original default rules.
    collectives.init_process_group("dp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 8, 2)
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 4], [1, 5], [2, 6], [3, 7]]
    new_group.reset_mock()
    collectives.init_process_group("dcp", "127.0.0.1", 46001, 0, 2, "cuda:1", 2, 8, 2)
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 4], [1, 5], [2, 6], [3, 7]]


@pytest.mark.parametrize("group_name", ["tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp"])
@pytest.mark.parametrize("explicit_topology_first", [False, True])
def test_rejects_different_members_with_same_rank_and_size(
    monkeypatch: pytest.MonkeyPatch, group_name: str, explicit_topology_first: bool
) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=0)
    # Choose an explicit layout different from this group type's default.
    explicit_memberships = (
        [[0, 4], [1, 5], [2, 6], [3, 7]]
        if group_name in ("tp", "moe_tp", "layerwise")
        else [[0, 1], [2, 3], [4, 5], [6, 7]]
    )
    first_memberships = explicit_memberships if explicit_topology_first else None
    second_memberships = None if explicit_topology_first else explicit_memberships
    collectives.init_process_group(
        group_name, "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 8, 0, group_ranks=first_memberships
    )
    new_group.reset_mock()

    # Both layouts give rank 0 local rank 0 and size 2; only the peers differ.
    with pytest.raises(RuntimeError, match="different members"):
        collectives.init_process_group(
            group_name, "127.0.0.1", 46001, 0, 2, "cuda:0", 0, 8, 0, group_ranks=second_memberships
        )
    new_group.assert_not_called()


@pytest.mark.parametrize(
    ("rank", "group_index", "world_size", "global_world_size"),
    [(0, 1, 2, 8), (1, 0, 2, 8), (-1, 1, 2, 8), (1, -1, 2, 8), (1, 1, 4, 8), (1, 1, 2, 16)],
)
def test_rejects_inconsistent_explicit_topology_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch,
    rank: int,
    group_index: int,
    world_size: int,
    global_world_size: int,
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)

    with pytest.raises((ValueError, RuntimeError), match="dcp topology"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            rank,
            world_size,
            "cuda:0",
            3,
            global_world_size,
            group_index,
            group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
        )
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


def test_explicit_topology_creation_failure_is_not_hidden(monkeypatch: pytest.MonkeyPatch) -> None:
    _, _, _, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    new_group.side_effect = RuntimeError("collective creation failed")
    derive_memberships = MagicMock(side_effect=AssertionError("default topology used"))
    monkeypatch.setattr(collectives, "_group_memberships", derive_memberships)

    with pytest.raises(RuntimeError, match="collective creation failed"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            1,
            2,
            "cuda:0",
            3,
            8,
            1,
            group_ranks=[[0, 1], [2, 3], [4, 5], [6, 7]],
        )
    assert collectives.dcp_group("cuda:0") is None
    assert new_group.call_count == 1
    derive_memberships.assert_not_called()


@pytest.mark.parametrize(
    "memberships",
    [[], [[]], [[0, 1], [2]], [[0, 1], [1, 2]], [[0, 1], [2, 4]], [[-1, 0], [1, 2]], [[1, 0], [2, 3]]],
)
def test_rejects_invalid_memberships_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch, memberships: list[list[int]]
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    with pytest.raises(ValueError, match="dcp topology"):
        collectives.init_process_group("dcp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, group_ranks=memberships)
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


@pytest.mark.parametrize("invalid_rank", [True, "0", 0.0])
def test_rejects_noninteger_topology_ranks_before_rendezvous(
    monkeypatch: pytest.MonkeyPatch, invalid_rank: object
) -> None:
    _, tcp_store, init_world, new_group = _mock_process_groups(monkeypatch, global_rank=3)
    with pytest.raises(TypeError, match="ranks must be integers"):
        collectives.init_process_group(
            "dcp",
            "127.0.0.1",
            46001,
            1,
            2,
            "cuda:0",
            3,
            4,
            1,
            group_ranks=[[invalid_rank, 1], [2, 3]],
        )
    tcp_store.assert_not_called()
    init_world.assert_not_called()
    new_group.assert_not_called()


def test_explicit_topology_keeps_cached_members_independent_of_caller(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_process_groups(monkeypatch, global_rank=3)
    memberships = [[0, 1], [2, 3]]
    group = collectives.init_process_group("dcp", "127.0.0.1", 46001, 1, 2, "cuda:0", 3, 4, 1, group_ranks=memberships)
    assert memberships == [[0, 1], [2, 3]]
    memberships[1][:] = [0, 1]
    memberships.clear()

    assert group.rank() == 1
    assert collectives._group_ranks[("dcp", "cuda:0")] == (2, 3)


def test_dp_execution_gather_uses_fixed_collective_for_equal_counts(monkeypatch):
    value = torch.zeros(4, 8)
    gathered = torch.zeros(8, 8)
    fixed_gather = MagicMock(return_value=gathered)
    variable_gather = MagicMock()
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (4, 4),
        rank=1,
    )

    assert output is gathered
    assert offset == 4
    fixed_gather.assert_called_once_with(
        value,
        dim=0,
        world_size=2,
        group_name="dp",
    )
    variable_gather.assert_not_called()


def test_dp_execution_gather_uses_variable_collective_for_uneven_counts(monkeypatch):
    value = torch.zeros(1, 8)
    gathered = torch.zeros(4, 8)
    fixed_gather = MagicMock()
    variable_gather = MagicMock(return_value=gathered)
    monkeypatch.setattr(collectives, "all_gather", fixed_gather)
    monkeypatch.setattr(collectives, "all_gather_variable", variable_gather)

    output, offset = collectives.gather_dp_execution_tokens(
        value,
        (3, 1),
        rank=1,
    )

    assert output is gathered
    assert offset == 3
    variable_gather.assert_called_once_with(value, [3, 1], 1, "dp")
    fixed_gather.assert_not_called()


def test_dp_execution_gather_rejects_local_shape_mismatch() -> None:
    with pytest.raises(RuntimeError, match="does not match the local tensor"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 2),
            rank=1,
        )


def test_dp_execution_gather_rejects_zero_execution_count() -> None:
    with pytest.raises(RuntimeError, match="must be positive"):
        collectives.gather_dp_execution_tokens(
            torch.zeros(1, 8),
            (3, 0),
            rank=1,
        )


def test_tcp_store_master_is_global_rank_zero_not_group_rank_zero(monkeypatch):
    _, tcp_store, _, _ = _mock_process_groups(monkeypatch, global_rank=2)

    collectives.init_process_group("tp", "127.0.0.1", 46001, 0, 2, "cuda:0", 2, 4, 1)

    assert tcp_store.call_args.args[:4] == ("127.0.0.1", 46001, 4, False)


def test_symmetric_memory_rejects_cross_host_group(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-1", "device_index": 0},
    ]
    can_access_peer = MagicMock(return_value=True)
    monkeypatch.setattr(torch.cuda, "can_device_access_peer", can_access_peer)

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])
    can_access_peer.assert_not_called()


def test_symmetric_memory_rejects_incomplete_peer_domain(monkeypatch):
    collectives._world_topology = [
        {"hostname": "node-0", "device_index": 0},
        {"hostname": "node-0", "device_index": 1},
    ]
    monkeypatch.setattr(
        torch.cuda,
        "can_device_access_peer",
        lambda source, destination: (source, destination) != (1, 0),
    )

    assert not collectives._supports_symmetric_memory(torch.device("cuda:0"), [0, 1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64, torch.int32])
def test_symmetric_buffer_rejects_unsupported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    empty = MagicMock()
    rendezvous = MagicMock()
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(_FakeGroup(0, 2), group_name, tensor) is None
    empty.assert_not_called()
    rendezvous.assert_not_called()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_symmetric_buffer_accepts_supported_dtype(monkeypatch, dtype):
    group_name = "tp"
    device = torch.device("cuda:0")
    collectives._symm_eligible[(group_name, str(device))] = True
    tensor = MagicMock()
    tensor.device = device
    tensor.dtype = dtype
    tensor.is_contiguous.return_value = True
    tensor.numel.return_value = 8
    tensor.element_size.return_value = torch.empty((), dtype=dtype).element_size()
    buffer = object()
    group = _FakeGroup(0, 2)
    group.group_name = "tp-group"
    empty = MagicMock(return_value=buffer)
    rendezvous = MagicMock()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(collectives.symm_mem, "empty", empty)
    monkeypatch.setattr(collectives.symm_mem, "rendezvous", rendezvous)

    assert collectives._symm_buffer(group, group_name, tensor) is buffer
    empty.assert_called_once_with(8, dtype=dtype, device=device)
    rendezvous.assert_called_once_with(buffer, "tp-group")


def test_row_all_gather_uses_one_contiguous_output(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(0, 2)
    collectives._groups[("tp", "cpu")] = group
    value = torch.arange(24, dtype=torch.float64).reshape(4, 6)[:, ::2]
    calls = []

    def gather(output: torch.Tensor, source: torch.Tensor, *, group: object) -> None:
        assert source.is_contiguous()
        calls.append(output.data_ptr())
        output.copy_(torch.cat((source, source + 1)))

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather)
    old = MagicMock(side_effect=AssertionError("row gather must not allocate a tensor list"))
    monkeypatch.setattr(dist, "all_gather", old)
    result = collectives.all_gather(value, 0, 2)
    torch.testing.assert_close(result, torch.cat((value, value + 1)))
    assert len(calls) == 1 and result.is_contiguous()
    old.assert_not_called()


def test_reduce_scatter_uses_sum_and_local_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    group = _FakeGroup(1, 2)
    collectives._groups[("moe_ep", "cpu")] = group
    value = torch.arange(48, dtype=torch.float32).reshape(6, 8)[:, ::2]
    observed = []

    def reduce(output: torch.Tensor, source: torch.Tensor, *, group: object) -> None:
        assert source.is_contiguous()
        observed.append(source.shape)
        output.copy_((source * 3).chunk(2)[1])

    monkeypatch.setattr(dist, "reduce_scatter_tensor", reduce)
    result = collectives.reduce_scatter(value, 2, "moe_ep")
    torch.testing.assert_close(result, value[3:] * 3)
    assert observed == [value.shape]
    assert result.shape == collectives._reduce_scatter_fake(value, 2).shape


@pytest.mark.parametrize("rows,world", [(3, 2), (0, 2), (4, 0), (4, -1)])
def test_reduce_scatter_rejects_invalid_row_contract(rows: int, world: int) -> None:
    with pytest.raises(ValueError, match="nonempty input rows"):
        collectives.reduce_scatter(torch.zeros(rows, 4), world)


def test_reduce_scatter_one_rank_does_not_alias_input() -> None:
    value = torch.arange(12).reshape(3, 4)
    actual = collectives.reduce_scatter(value, 1)
    torch.testing.assert_close(actual, value)
    assert actual.data_ptr() != value.data_ptr()
