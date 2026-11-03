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

"""Tensor-parallel process groups and collectives.

The collectives themselves are hardware-neutral: only the ProcessGroup backend
differs (NCCL on CUDA, HCCL on NPU), and torch picks it from the device. They
are graph operators so that a compiled graph keeps the communication node in
place instead of splitting around it.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Sequence
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed import ProcessGroup

_GROUP_NAMES = frozenset(("tp", "dp", "moe_tp", "moe_ep", "cp", "layerwise", "dcp", "engram_dp"))
# ``tp`` and ``moe_tp`` own a contiguous block of global ranks, while ``dp``,
# ``moe_ep``, ``cp`` and ``dcp`` stride across those blocks. Context groups also
# need the within-DP stride when their world contains multiple DP cohorts.
# ``cp`` strides by the attention TP size: ranks sharing a (dp, tp) slot but
# holding different sequence shards form one CP group, matching the C++
# compute_cp_group_ranks layout (rank = dp*cp*tp + cp_rank*tp + tp_rank).
# CP and legacy DCP callers with DP pass an explicit within-cohort stride.
# Explicit memberships can override these defaults when the upstream runtime
# has already finalized a model-specific topology.
_CONTIGUOUS_GROUPS = frozenset(("tp", "moe_tp", "layerwise"))

_groups = {}
_group_ranks = {}
_stores = {}
_world_topology = None
_world_rank = None
_world_endpoint = None
_symm_eligible = {}
_symm_buffers = {}
_world_initialized = False
_dp_host_steps: dict[tuple[int, ...], int] = {}
# Resolved MoE EP communicator names, keyed by device string. The group
# topology is fixed at rendezvous and ``get_hccl_comm_name`` lazily builds the
# communicator, so a per-layer/per-step probe must resolve at most once.
_moe_ep_comm_name_cache: dict[str, str] = {}


def _backend_for(device: torch.device) -> str:
    if device.type == "cuda":
        return "nccl"
    import torch_npu  # noqa: F401

    return "hccl"


def _shared_store(host: str, port: int, global_rank: int, global_world_size: int) -> dist.Store:
    store_key = (host, port)
    store = _stores.get(store_key)
    if store is None:
        store = dist.TCPStore(
            host,
            port,
            global_world_size,
            global_rank == 0,
            timedelta(minutes=5),
            wait_for_workers=False,
            multi_tenant=True,
        )
        _stores[store_key] = store
    return store


def _exchange_world_topology(
    store: dist.Store,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> list[dict[str, object]]:
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    local = {"hostname": socket.gethostname(), "device_index": device_index}
    key_prefix = "xllm/python_collectives/topology/v1"
    store.set(f"{key_prefix}/{global_rank}", json.dumps(local))
    return [json.loads(store.get(f"{key_prefix}/{rank}").decode("utf-8")) for rank in range(global_world_size)]


def _ensure_world(
    host: str,
    port: int,
    device: torch.device,
    global_rank: int,
    global_world_size: int,
) -> None:
    """Initialize the default process group covering every rank.

    The parallel groups are subgroups of it. Building them with ``new_group``
    instead of constructing communicators directly registers them with c10d,
    which is what lets a group back a symmetric-memory allocation.
    """
    global _world_initialized, _world_topology, _world_rank, _world_endpoint
    if _world_initialized:
        return
    store = _shared_store(host, port, global_rank, global_world_size)
    dist.init_process_group(
        backend=_backend_for(device),
        store=store,
        rank=global_rank,
        world_size=global_world_size,
        timeout=timedelta(minutes=5),
    )
    _world_topology = _exchange_world_topology(store, device, global_rank, global_world_size)
    _world_rank = global_rank
    _world_endpoint = (host, port)
    _world_initialized = True


def _group_memberships(
    group_name: str,
    world_size: int,
    global_world_size: int,
    group_stride: int | None = None,
) -> list[list[int]]:
    """Every group of this kind, in an order all ranks agree on.

    ``new_group`` is collective over the whole world, so each rank has to create
    all groups of a kind in the same order, not only the one it belongs to.
    """
    if world_size <= 0 or global_world_size % world_size:
        raise ValueError(f"{group_name} size {world_size} does not divide the world size {global_world_size}")
    if group_stride is not None:
        if group_name not in ("cp", "dcp") or group_stride <= 0 or global_world_size % (world_size * group_stride):
            raise ValueError("CP/DCP group stride must describe complete DP cohorts")
        # Context parallelism varies within one DP cohort; a global stride instead mixes
        # different requests whenever DP > 1.
        cohort_size = world_size * group_stride
        return [
            [base + tp_rank + cp_rank * group_stride for cp_rank in range(world_size)]
            for base in range(0, global_world_size, cohort_size)
            for tp_rank in range(group_stride)
        ]
    count = global_world_size // world_size
    if group_name in _CONTIGUOUS_GROUPS:
        return [[index * world_size + offset for offset in range(world_size)] for index in range(count)]
    return [[index + offset * count for offset in range(world_size)] for index in range(count)]


def _validate_group_topology(
    group_name: str,
    memberships: list[list[int]],
    rank: int,
    world_size: int,
    global_rank: int,
    global_world_size: int,
    group_index: int,
    *,
    explicit_memberships: bool,
) -> None:
    """Validate topology and caller coordinates before creating or reusing a group."""
    if (
        global_world_size <= 0
        or world_size <= 0
        or len(memberships) * world_size != global_world_size
        or any(len(ranks) != world_size for ranks in memberships)
    ):
        raise ValueError(f"{group_name} topology sizes do not match the process-group initialization")
    if explicit_memberships:
        all_ranks = [member for ranks in memberships for member in ranks]
        if any(not isinstance(member, int) or isinstance(member, bool) for member in all_ranks):
            raise TypeError(f"{group_name} topology ranks must be integers")
        if sorted(all_ranks) != list(range(global_world_size)):
            raise ValueError(f"{group_name} topology must partition all global ranks exactly once")
        # c10d new_group sorts its rank list; reject a conflicting local-rank
        # order instead of silently changing the upstream topology.
        if any(ranks != sorted(ranks) for ranks in memberships):
            raise ValueError(f"{group_name} topology members within each group must be sorted")
    if (
        not 0 <= group_index < len(memberships)
        or not 0 <= rank < world_size
        or memberships[group_index][rank] != global_rank
    ):
        raise RuntimeError(f"{group_name} topology rank or group index does not match the caller")


def _supports_symmetric_memory(device: torch.device, ranks: list[int]) -> bool:
    if device.type != "cuda" or _world_topology is None:
        return False
    topology = [_world_topology[rank] for rank in ranks]
    hostnames = {entry["hostname"] for entry in topology}
    if len(hostnames) != 1:
        return False
    device_indices = [entry["device_index"] for entry in topology]
    if any(not isinstance(index, int) or index < 0 for index in device_indices):
        return False
    if len(set(device_indices)) != len(device_indices):
        return False
    return all(
        torch.cuda.can_device_access_peer(source, destination)
        for source in device_indices
        for destination in device_indices
        if source != destination
    )


def init_process_group(
    group_name: str,
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
    group_stride: int | None = None,
    group_ranks: Sequence[Sequence[int]] | None = None,
    backend: str | None = None,
) -> ProcessGroup:
    """Create a parallel group from explicit memberships or the legacy layout.

    ``group_ranks`` lists every group of this kind in the same order on all
    processes, not just the current process's peers. Each sorted inner list
    contains global ranks; ``group_index`` selects our list and ``rank`` our
    position in it. None preserves the default layout for ``group_name``.

    ``backend`` overrides the device-derived backend for this group only
    (default unchanged). The node-local Engram sharing group passes ``"gloo"``
    because its object collectives are not available on HCCL; a cached group
    created with a different backend is rejected rather than silently reused.
    """
    if group_name not in _GROUP_NAMES:
        raise ValueError(f"unsupported parallel group: {group_name}")
    device_obj = torch.device(device)
    group_key = (group_name, str(device_obj))
    if group_stride is not None and group_ranks is not None:
        raise ValueError("group_stride and group_ranks are mutually exclusive")
    if group_ranks is not None:
        memberships = [list(ranks) for ranks in group_ranks]
    else:
        memberships = _group_memberships(group_name, world_size, global_world_size, group_stride)
    _validate_group_topology(
        group_name,
        memberships,
        rank,
        world_size,
        global_rank,
        global_world_size,
        group_index,
        explicit_memberships=group_ranks is not None,
    )
    group = _groups.get(group_key)
    if group is not None:
        if group.rank() != rank or group.size() != world_size:
            raise RuntimeError(
                f"{group_name} group for {group_key[1]} is already initialized as "
                f"rank {group.rank()}/{group.size()}, requested "
                f"rank {rank}/{world_size}"
            )
        if _group_ranks.get(group_key) != tuple(memberships[group_index]):
            raise RuntimeError(f"{group_name} group for {group_key[1]} is already initialized with different members")
        if backend is not None and dist.get_backend(group) != backend:
            raise RuntimeError(
                f"{group_name} group for {group_key[1]} is already initialized with backend "
                f"{dist.get_backend(group)}, requested {backend}; refusing to reuse it"
            )
        return group

    _ensure_world(host, port, device_obj, global_rank, global_world_size)
    effective_backend = backend if backend is not None else _backend_for(device_obj)

    own = None
    own_ranks = None
    for index, ranks in enumerate(memberships):
        candidate = dist.new_group(ranks=ranks, timeout=timedelta(minutes=5), backend=effective_backend)
        if global_rank in ranks:
            own = candidate
            own_index = index
            own_ranks = ranks
    if own is None or own.rank() != rank:
        raise RuntimeError(
            f"derived {group_name} membership disagrees with the caller: global "
            f"rank {global_rank} of {global_world_size} expected rank {rank} of "
            f"{world_size}, got {None if own is None else own.rank()}"
        )
    if own_index != group_index:
        raise RuntimeError(f"derived {group_name} group index {own_index} does not match the caller's {group_index}")

    assert own_ranks is not None
    _groups[group_key] = own
    _group_ranks[group_key] = tuple(own_ranks)
    _symm_eligible[group_key] = _supports_symmetric_memory(device_obj, own_ranks)
    return own


def init_tp_group(
    host: str,
    port: int,
    rank: int,
    world_size: int,
    device: str,
    global_rank: int,
    global_world_size: int,
    group_index: int,
) -> ProcessGroup:
    return init_process_group(
        "tp",
        host,
        port,
        rank,
        world_size,
        device,
        global_rank,
        global_world_size,
        group_index,
    )


def _require_group(x: torch.Tensor, group_name: str) -> ProcessGroup:
    group = _groups.get((group_name, str(x.device)))
    if group is None:
        raise RuntimeError(f"{group_name} collective called before its process group was initialized for {x.device}")
    return group


def tp_group(device: torch.device | str) -> ProcessGroup:
    group = _groups.get(("tp", str(torch.device(device))))
    if group is None:
        raise RuntimeError(f"TP group is not initialized for {device}")
    return group


def tp_rank(device: torch.device | str) -> int:
    """Rank in the TP group for ``device`` (0 when no TP group exists)."""
    group = _groups.get(("tp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_rank(device: torch.device | str) -> int:
    """Rank in the CP group for ``device`` (0 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.rank() if group is not None else 0


def cp_world_size(device: torch.device | str) -> int:
    """Size of the CP group for ``device`` (1 when no CP group exists)."""
    group = _groups.get(("cp", str(torch.device(device))))
    return group.size() if group is not None else 1


def cp_process_group(device: torch.device | str) -> ProcessGroup:
    """The CP group for direct ``torch.distributed`` calls.

    The graph-operator collectives above (``all_gather`` / ``broadcast_``)
    are synchronous by construction; callers that need an async work handle
    or point-to-point ops resolve the raw ProcessGroup here. Raises when
    the CP group was never initialized for ``device`` -- the same contract
    ``_require_group`` gives the wrapped collectives.
    """
    group = _groups.get(("cp", str(torch.device(device))))
    if group is None:
        raise RuntimeError(f"cp process group was not initialized for {device}")
    return group


def layerwise_rank(device: torch.device | str) -> int:
    """Rank in the layerwise split group, or zero when disabled."""
    group = _groups.get(("layerwise", str(torch.device(device))))
    return group.rank() if group is not None else 0


def _native_runtime_op(name: str) -> object | None:
    """Return an embedded C++ collective when running under PyExecutorImpl."""
    try:
        import xllm_runtime
    except ImportError:
        return None
    return getattr(xllm_runtime, name, None)


def tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "tp")


def tp_all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    op = _native_runtime_op("tp_all_gather")
    if op is not None:
        return op(x, dim)
    return all_gather(x, dim, world_size, "tp")


def dp_all_gather(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
) -> torch.Tensor:
    op = _native_runtime_op("dp_all_gather")
    if op is not None:
        return op(x, token_counts)
    if all(count == token_counts[0] for count in token_counts):
        return all_gather(
            x,
            dim=0,
            world_size=len(token_counts),
            group_name="dp",
        )
    return all_gather_variable(x, token_counts, rank, "dp")


def dp_all_true(value: bool, device: torch.device | str, world_size: int) -> bool:
    """Agree on a CPU condition through the existing rendezvous store."""
    ranks = _group_ranks.get(("dp", str(torch.device(device))))
    store = _stores.get(_world_endpoint)
    if ranks is None or len(ranks) != world_size or _world_rank not in ranks or store is None:
        raise RuntimeError(f"DP host agreement requires an initialized {world_size}-rank DP group for {device}")
    step = _dp_host_steps.get(ranks, 0)
    _dp_host_steps[ranks] = step + 1
    prefix = f"xllm/python_collectives/dp_admission/{','.join(map(str, ranks))}/{step}"
    keys = [f"{prefix}/{rank}" for rank in ranks]
    store.set(f"{prefix}/{_world_rank}", b"1" if value else b"0")
    decisions = store.multi_get(keys)
    # The final reader retires this step only after every peer has copied all
    # decisions. Step-specific keys allow faster peers to start the next one.
    readers_key = f"{prefix}/readers"
    if store.add(readers_key, 1) == world_size:
        for key in keys:
            store.delete_key(key)
        store.delete_key(readers_key)
    return all(decision == b"1" for decision in decisions)


def moe_tp_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_tp_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_tp")


def moe_ep_all_reduce(x: torch.Tensor) -> None:
    op = _native_runtime_op("moe_ep_all_reduce")
    if op is not None:
        op(x)
        return
    all_reduce_(x, "moe_ep")


def moe_ep_hccl_info(device: torch.device | str) -> tuple[str, int, int]:
    """Return the initialized expert group's HCCL name, rank and size."""
    device_obj = torch.device(device)
    group = _groups.get(("moe_ep", str(device_obj)))
    if group is None:
        raise RuntimeError(f"MoE EP dispatch requires an initialized process group for {device_obj}")
    rank, world_size = group.rank(), group.size()
    if world_size <= 1 or not 0 <= rank < world_size:
        raise RuntimeError(f"MoE EP dispatch requires an initialized multi-rank group: rank {rank}/{world_size}")
    get_comm_name = getattr(group._get_backend(device_obj), "get_hccl_comm_name", None)
    if get_comm_name is None:
        raise RuntimeError("MoE EP dispatch requires an HCCL process group")
    comm_name = get_comm_name(rank)
    if not isinstance(comm_name, str) or not comm_name:
        raise RuntimeError("MoE EP dispatch requires a nonempty HCCL communication name")
    return comm_name, rank, world_size


def dcp_group(device: torch.device | str) -> ProcessGroup | None:
    return _groups.get(("dcp", str(torch.device(device))))


def moe_ep_group(device: torch.device | str) -> ProcessGroup | None:
    """The MoE EP process group for ``device`` (``None`` when EP is not initialized)."""
    return _groups.get(("moe_ep", str(torch.device(device))))


def moe_ep_rank(device: torch.device | str) -> int:
    """Rank in the MoE EP group for ``device`` (0 when no EP group exists).

    The V4-family Python models own their EP split in C++, so the native
    runtime's rank is preferred: no Python ``moe_ep`` group exists there and
    the ``_groups`` fallback would silently report 0. The Python-created group
    is the fallback for the models whose groups this module builds; ``0`` is
    the final fallback.
    """
    native_rank = _native_moe_ep_rank()
    if native_rank is not None:
        return native_rank
    group = moe_ep_group(device)
    return group.rank() if group is not None else 0


def moe_ep_rank_from_native(device: torch.device | str) -> int | None:
    """MoE EP rank held by the embedded C++ runtime, or ``None`` when only a fallback could answer.

    ``moe_ep_rank`` deliberately collapses three sources (native runtime,
    Python ``_groups``, hard ``0``) into one int. The V4-family models own their
    EP split in C++, so a caller that routes on the value -- the MC2 guard -- must
    NOT accept the ``_groups``/``0`` fallback as authoritative: it would silently
    pin every rank to ``0`` and dispatch to the wrong rank. This probe keeps the
    source distinction without changing ``moe_ep_rank``'s contract.

    ``None`` means no usable native rank (binding absent, present-but-failing,
    or reporting ``None``); ``0`` is a legitimate native answer (rank 0).
    """
    return _native_moe_ep_rank()


def _native_moe_ep_rank() -> int | None:
    """MoE EP rank held by the embedded C++ runtime, or ``None``.

    The binding lives in ``xllm_runtime``; a build that predates it -- or a
    host without the embedded runtime -- simply has no attribute, which
    ``_native_runtime_op`` reports as ``None``. A present binding that fails is
    treated the same way: this is a capability probe on a path that has a
    supported fallback, so no exception may escape the guard.
    """
    op = _native_runtime_op("moe_ep_rank")
    if op is None:
        return None
    try:
        rank = op()
    except Exception:  # noqa: BLE001 - capability probe, fallback is defined.
        return None
    if rank is None:
        return None
    return int(rank)


def _native_moe_ep_comm_name() -> str | None:
    """HCCL communicator name held by the embedded C++ runtime, or ``None``.

    The binding lives in ``xllm_runtime``; a build that predates it -- or a
    host without the embedded runtime -- simply has no attribute, which
    ``_native_runtime_op`` reports as ``None``. A present binding that fails is
    treated the same way: this is a capability probe on a path that has a
    supported fallback, so no exception may escape the guard.
    """
    op = _native_runtime_op("moe_ep_comm_name")
    if op is None:
        return None
    try:
        name = op()
    except Exception:  # noqa: BLE001 - capability probe, fallback is defined.
        return None
    if not isinstance(name, str) or not name:
        return None
    return name


def _moe_ep_backend_device() -> torch.device:
    """Device whose registered backend the MoE EP group's HCCL wrapper lives in.

    Broken out so the Python fallback can be exercised on a host without
    torch_npu (which is what makes ``torch.device("npu")`` constructible).
    """
    return torch.device("npu")


def _python_moe_ep_comm_name(device: torch.device | str) -> str | None:
    """HCCL communicator name of the Python-created MoE EP process group."""
    group = moe_ep_group(device)
    if group is None:
        return None
    # ``_get_backend`` is the only handle torch gives on the HCCL wrapper; the
    # same idiom is used by vllm-ascend's token dispatcher. A backend without
    # the accessor simply cannot serve the MC2 kernels, so report that as
    # "unavailable" rather than guessing a name.
    backend = group._get_backend(_moe_ep_backend_device())  # noqa: SLF001
    get_comm_name = getattr(backend, "get_hccl_comm_name", None)
    if get_comm_name is None:
        return None
    return get_comm_name(group.rank())


def moe_ep_comm_name(device: torch.device | str) -> str | None:
    """HCCL communicator name of the MoE EP group, or ``None`` when unavailable.

    The ``npu_moe_distribute_{dispatch,combine}_v2`` kernels take the raw HCCL
    communicator name (``group_ep``), not a ``ProcessGroup``. The V4-family
    Python models own their EP split in C++, so the native runtime's group is
    preferred; the Python-created group is the fallback for the models whose
    groups this module builds. ``None`` means the caller must fall back to the
    portable expert path -- no EP group on either side, a non-HCCL backend, or
    a torch build without ``get_hccl_comm_name``.

    A resolved name is cached per device: unlike a ``None`` result, the name is
    only meaningful once the group exists, and ``get_hccl_comm_name`` lazily
    builds the communicator on its first call.
    """
    cache_key = str(torch.device(device))
    cached = _moe_ep_comm_name_cache.get(cache_key)
    if cached is not None:
        return cached
    name = _native_moe_ep_comm_name()
    if name is None:
        name = _python_moe_ep_comm_name(device)
    if name is not None:
        _moe_ep_comm_name_cache[cache_key] = name
    return name


def world_topology() -> list[dict[str, object]] | None:
    """Per-global-rank node topology, or None before the world is initialized."""
    return _world_topology


def get_engram_dp_shard_size(
    world_size: int,
    dp_size: int,
    replica_size: int,
    topology: Sequence[dict[str, object]] | None,
) -> int:
    """Size of the node-local DP sharing shards for a DP-shared Engram table.

    :func:`engram_dp_shard_size` calls this with the live world so it stays
    testable on its own.

    Mirrors vLLM's ``_engram_dp_shard_size``: the DP ranks of one node are
    grouped in shards of ``gcd(dp_size, replicas_per_node)`` so every sharing
    group is node-local and (because a DP group already fixes TP) maps the same
    TP shard. Falls back to 1 -- no sharing -- when the ranks do not tile into
    whole-node replica blocks, when a node's ranks are not a contiguous
    global-rank block, or when the topology is unavailable. A return of 1 is
    the caller's signal that ``dp_shared_memory`` cannot be honored.

    ``topology`` is the per-global-rank ``{"hostname": ...}`` list from
    ``world_topology``; the hostname is only a precheck -- the authoritative
    evidence that the sharing group really shares host memory is a successful
    ``multiprocessing.shared_memory`` attach with a matching token.
    """
    if topology is None or world_size <= 0 or dp_size <= 0 or replica_size <= 0 or len(topology) != world_size:
        return 1
    hostnames = [entry.get("hostname") for entry in topology]
    # A missing/empty hostname cannot prove node locality: ``set()`` would treat
    # None (or "") as a distinct node, inflating ``node_count`` and making the
    # contiguous-rank check below compare against a bogus host. Fall back to 1
    # -- no sharing -- instead of deriving a shard size from a bad topology.
    if any(not isinstance(host, str) or not host for host in hostnames):
        return 1
    node_count = len(set(hostnames))
    replicas_per_node, remainder = divmod(world_size, node_count * replica_size)
    if remainder:
        return 1
    shard_size = math.gcd(dp_size, replicas_per_node)
    if shard_size == 1 or node_count == 1:
        return shard_size
    ranks_per_node = replicas_per_node * replica_size
    for start in range(0, world_size, ranks_per_node):
        node_ranks = [rank for rank, host in enumerate(hostnames) if host == hostnames[start]]
        if node_ranks != list(range(start, start + ranks_per_node)):
            return 1
    return shard_size


def _engram_dp_memberships(dp_size: int, shard_size: int, global_world_size: int) -> list[list[int]]:
    """Every node-local Engram sharing group, in a world-agreed order.

    The DP group already fixes the TP shard, so slicing each DP membership into
    consecutive ``shard_size`` blocks yields groups that share both the node and
    the TP shard. The slices are increasing (a DP membership strides upward by
    global rank), which ``new_group`` requires.
    """
    # Explicit guard, not ``assert``: an invariant that must hold in production
    # cannot be dropped by ``python -O``.
    if not (shard_size > 0 and dp_size % shard_size == 0):
        raise ValueError(f"engram DP shard size {shard_size} must divide DP size {dp_size}")
    memberships: list[list[int]] = []
    for dp_members in _group_memberships("dp", dp_size, global_world_size):
        for start in range(0, len(dp_members), shard_size):
            memberships.append(dp_members[start : start + shard_size])
    return memberships


def engram_dp_shard_size(dp_size: int, replica_size: int) -> int:
    """Effective node-local Engram sharing shard size (1 when unavailable)."""
    if not _world_initialized or _world_topology is None:
        return 1
    return get_engram_dp_shard_size(len(_world_topology), dp_size, replica_size, _world_topology)


def init_engram_dp_group(device: str, dp_size: int, shard_size: int) -> ProcessGroup:
    """Create the node-local gloo DP sharing group for a DP-shared Engram table.

    Object collectives (broadcast/gather of segment descriptors) have no HCCL
    equivalent, so the group is created with ``backend="gloo"`` (matching
    vllm-ascend). ``shard_size`` is the caller's :func:`engram_dp_shard_size`
    result -- this module does not recompute it (a distributed helper must not
    depend on the caller's world or carry its ``engram_config`` error text).
    Returns this rank's group; the generic guard below rejects a degenerate
    single-rank shard so the caller can surface the required configuration
    error instead of silently running unshared.
    """
    if not _world_initialized or _world_topology is None or _world_rank is None or _world_endpoint is None:
        raise RuntimeError("engram DP-shared host table requires an initialized process-group world")
    global_world_size = len(_world_topology)
    if shard_size <= 1:
        raise RuntimeError(
            f"engram DP sharing requires more than one rank per shard; got DP={dp_size}, shard_size={shard_size}"
        )
    memberships = _engram_dp_memberships(dp_size, shard_size, global_world_size)
    # Node-locality precheck only; the authoritative proof is a successful
    # attach with a matching token (SharedHostEngramStorage).
    for ranks in memberships:
        hosts = {_world_topology[rank].get("hostname") for rank in ranks}
        if len(hosts) != 1:
            raise RuntimeError(
                "engram DP sharing group spans multiple nodes: "
                f"ranks {ranks} on hosts {sorted(str(host) for host in hosts)}"
            )
    own_index, own_rank = None, None
    for index, ranks in enumerate(memberships):
        if _world_rank in ranks:
            own_index, own_rank = index, ranks.index(_world_rank)
            break
    if own_index is None:
        raise RuntimeError(f"global rank {_world_rank} is missing from the engram DP memberships")
    assert own_rank is not None
    host, port = _world_endpoint
    return init_process_group(
        "engram_dp",
        host,
        port,
        own_rank,
        shard_size,
        device,
        _world_rank,
        global_world_size,
        own_index,
        group_ranks=memberships,
        backend="gloo",
    )


def _require_named_group(group_name: str) -> ProcessGroup:
    for (name, _device), group in _groups.items():
        if name == group_name:
            return group
    raise RuntimeError(f"{group_name} collective called before its process group was initialized")


def engram_broadcast_object_list(object_list: list, src: int = 0, group_name: str = "engram_dp") -> None:
    """Broadcast picklable objects from group rank ``src`` (default leader)."""
    group = _require_named_group(group_name)
    global_src = dist.get_global_rank(group, src)
    dist.broadcast_object_list(object_list, src=global_src, group=group)


def engram_all_gather_object(object_list: list, obj: object, group_name: str = "engram_dp") -> None:
    """Gather one picklable object per group rank into ``object_list``."""
    group = _require_named_group(group_name)
    dist.all_gather_object(object_list, obj, group=group)


def engram_barrier(group_name: str = "engram_dp", timeout: timedelta | None = None) -> None:
    """Block until every rank of the named group reaches this point.

    ``timeout`` is ``None`` by default, which is exactly the old behavior: a
    plain ``dist.barrier`` that keeps the group's own timeout (the 5-minute
    value ``new_group`` gave it). Passing a ``timeout`` switches to
    ``dist.monitored_barrier`` -- the c10d barrier with a per-call deadline --
    with ``wait_all_ranks=True``. ``wait_all_ranks`` only changes how the
    monitor (group rank 0) aggregates failures: ``True`` lets it report every
    failed rank in one raise, ``False`` makes it fail-fast on the first miss.
    It does **not** change *which* ranks raise. The per-call ``timeout`` is
    likewise enforced on the monitor (group rank 0) only: a non-zero survivor
    rank whose peer never reaches the fence still blocks until the group's own
    timeout (5 minutes) before it raises. It is Gloo-only, which is the backend
    the engram DP sharing group is created with. The deadline bounds this one
    rendezvous on the monitor; it does not shorten the group's own timeout.
    """
    group = _require_named_group(group_name)
    if timeout is None:
        dist.barrier(group=group)
        return
    dist.monitored_barrier(group=group, timeout=timeout, wait_all_ranks=True)


# A one-shot symmetric-memory reduction is an ordinary kernel on the current
# stream, so a captured graph runs it inline. NCCL runs collectives on its own
# stream, which costs a fork/join per call -- measured at ~32us of device idle
# before every all-reduce in the decode graph, and there are dozens per step.
# Past this size the staging copy stops paying for itself and NCCL's ring wins,
# and the bound also keeps the buffer cache to decode-sized shapes rather than
# one entry per distinct prefill length.
_SYMM_MEM_MAX_BYTES = 512 * 1024
# Distinct shapes still accumulate one buffer each; stop allocating rather than
# grow without limit when a workload sweeps many small shapes.
_SYMM_MEM_MAX_BUFFERS = 64
_SYMM_MEM_DTYPES = frozenset((torch.float32, torch.bfloat16))


def _symm_buffer(group: ProcessGroup, group_name: str, x: torch.Tensor) -> torch.Tensor | None:
    """Return a symmetric-memory staging buffer for ``x``, or None.

    Allocation and rendezvous are collective and cannot run inside a graph
    capture, so a shape first seen during capture falls back to NCCL. The
    capture path is warmed up eagerly beforehand, which is where the decode
    shapes get their buffers.
    """
    group_key = (group_name, str(x.device))
    if not _symm_eligible.get(group_key, False):
        return None
    if not x.is_contiguous() or x.dtype not in _SYMM_MEM_DTYPES:
        return None
    if x.numel() * x.element_size() > _SYMM_MEM_MAX_BYTES:
        return None

    key = (group_name, str(x.device), x.dtype, x.numel())
    buffer = _symm_buffers.get(key)
    if buffer is not None:
        return buffer
    if torch.cuda.is_current_stream_capturing():
        return None
    if len(_symm_buffers) >= _SYMM_MEM_MAX_BUFFERS:
        return None

    buffer = symm_mem.empty(x.numel(), dtype=x.dtype, device=x.device)
    symm_mem.rendezvous(buffer, group.group_name)
    _symm_buffers[key] = buffer
    return buffer


@torch.library.custom_op("xllm_ops::all_reduce_", mutates_args={"x"})
def all_reduce_(x: torch.Tensor, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    buffer = _symm_buffer(group, group_name, x)
    if buffer is None:
        dist.all_reduce(x, group=group)
        return
    flat = x.view(-1)
    buffer.copy_(flat)
    torch.ops.symm_mem.one_shot_all_reduce_out(buffer, "sum", group.group_name, flat)


@torch.library.custom_op("xllm_ops::broadcast_", mutates_args={"x"})
def broadcast_(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    group = _require_group(x, group_name)
    if not 0 <= src < group.size():
        raise RuntimeError(f"invalid {group_name} source rank {src}")
    ranks = _group_ranks.get((group_name, str(x.device)))
    if ranks is None:
        raise RuntimeError(f"{group_name} group rank map is unavailable")
    dist.broadcast(x, src=ranks[src], group=group)


@broadcast_.register_fake
def _(x: torch.Tensor, src: int, group_name: str = "tp") -> None:
    del src, group_name
    return None


@all_reduce_.register_fake
def _(x: torch.Tensor, group_name: str = "tp") -> None:
    return None


@torch.library.custom_op("xllm_ops::all_gather", mutates_args=())
def all_gather(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    if dim == 0:
        output = x.new_empty((x.shape[0] * world_size, *x.shape[1:]))
        dist.all_gather_into_tensor(output, x.contiguous(), group=group)
        return output
    # Non-row gathers can receive strided inputs. HCCL requires contiguous
    # buffers; contiguous() reuses storage when the input is already contiguous.
    x = x.contiguous()
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    dist.all_gather(chunks, x, group=group)
    return torch.cat(chunks, dim=dim)


@all_gather.register_fake
def _(x: torch.Tensor, dim: int, world_size: int, group_name: str = "tp") -> torch.Tensor:
    shape = list(x.shape)
    shape[dim] *= world_size
    return x.new_empty(shape)


@torch.library.custom_op("xllm_ops::all_to_all", mutates_args=())
def all_to_all(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    """Exchange equally sized rank-major row blocks without reducing values."""
    if world_size <= 0 or x.ndim == 0 or x.shape[0] % world_size:
        raise ValueError("all_to_all requires row blocks divisible by world size")
    if world_size == 1:
        return x.clone()
    if group_name == "tp":
        op = _native_runtime_op("tp_all_to_all")
        if op is not None:
            output = op(x, world_size)
            if output is not None:
                return output
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    input_tensor = x.contiguous()
    output = torch.empty_like(input_tensor)
    dist.all_to_all_single(output, input_tensor, group=group)
    return output


@all_to_all.register_fake
def _all_to_all_fake(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    # TP=1 clones with preserved strides; a multi-rank exchange returns contiguous rows.
    return torch.empty_like(x) if world_size == 1 else x.new_empty(x.shape)


def _reduce_scatter_shape(x: torch.Tensor, world_size: int) -> tuple[int, ...]:
    if world_size <= 0 or x.ndim == 0 or x.shape[0] <= 0 or x.shape[0] % world_size:
        raise ValueError("reduce_scatter requires a positive world size dividing nonempty input rows")
    return (x.shape[0] // world_size, *x.shape[1:])


@torch.library.custom_op("xllm_ops::reduce_scatter", mutates_args=())
def reduce_scatter(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    """Reduce full-hidden partials directly into disjoint contiguous token rows."""
    shape = _reduce_scatter_shape(x, world_size)
    if world_size == 1:
        return x.clone()
    group = _require_group(x, group_name)
    if group.size() != world_size:
        raise RuntimeError(f"{group_name} world-size mismatch: expected {world_size}, got {group.size()}")
    output = x.new_empty(shape)
    dist.reduce_scatter_tensor(output, x.contiguous(), group=group)
    return output


@reduce_scatter.register_fake
def _reduce_scatter_fake(x: torch.Tensor, world_size: int, group_name: str = "tp") -> torch.Tensor:
    return x.new_empty(_reduce_scatter_shape(x, world_size))


@torch.library.custom_op("xllm_ops::all_gather_variable", mutates_args=())
def all_gather_variable(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    group = _require_group(x, group_name)
    if group.size() != len(token_counts):
        raise RuntimeError(
            f"{group_name} size mismatch: group has {group.size()} ranks, "
            f"but token_counts has {len(token_counts)} entries"
        )
    if not 0 <= rank < len(token_counts):
        raise RuntimeError(f"invalid {group_name} rank {rank}")

    local_tokens = token_counts[rank]
    if local_tokens < 0 or local_tokens > x.shape[0]:
        raise RuntimeError(f"invalid local token count {local_tokens} for input with {x.shape[0]} rows")
    padded_tokens = max(max(token_counts, default=0), 1)
    padded_shape = list(x.shape)
    padded_shape[0] = padded_tokens
    padded = x.new_zeros(padded_shape)
    if local_tokens:
        padded[:local_tokens].copy_(x[:local_tokens])

    chunks = [torch.empty_like(padded) for _ in token_counts]
    dist.all_gather(chunks, padded, group=group)
    valid_chunks = [chunk[:count] for chunk, count in zip(chunks, token_counts) if count]
    if not valid_chunks:
        empty_shape = list(x.shape)
        empty_shape[0] = 0
        return x.new_empty(empty_shape)
    return torch.cat(valid_chunks, dim=0)


@all_gather_variable.register_fake
def _(
    x: torch.Tensor,
    token_counts: list[int],
    rank: int,
    group_name: str,
) -> torch.Tensor:
    del rank, group_name
    shape = list(x.shape)
    shape[0] = sum(token_counts)
    return x.new_empty(shape)


def gather_dp_execution_tokens(
    x: torch.Tensor,
    execution_token_counts: Sequence[int],
    rank: int,
) -> tuple[torch.Tensor, int]:
    """Gather the execution rows materialized by every DP rank."""
    counts = [int(count) for count in execution_token_counts]
    if not counts:
        return x, 0
    if any(count <= 0 for count in counts):
        raise RuntimeError(f"DP execution token counts must be positive, got {counts}")
    if not 0 <= rank < len(counts):
        raise RuntimeError(f"invalid DP rank {rank} for token counts {counts}")
    if counts[rank] != x.shape[0]:
        raise RuntimeError(
            "DP execution token count does not match the local tensor: "
            f"rank={rank}, rows={x.shape[0]}, token_counts={counts}"
        )

    return dp_all_gather(x, counts, rank), sum(counts[:rank])


__all__ = [
    "init_process_group",
    "init_tp_group",
    "init_engram_dp_group",
    "engram_dp_shard_size",
    "get_engram_dp_shard_size",
    "world_topology",
    "engram_broadcast_object_list",
    "engram_all_gather_object",
    "engram_barrier",
    "tp_group",
    "tp_rank",
    "cp_rank",
    "cp_world_size",
    "layerwise_rank",
    "dcp_group",
    "all_reduce_",
    "broadcast_",
    "all_gather",
    "reduce_scatter",
    "all_gather_variable",
    "dp_all_gather",
    "dp_all_true",
    "gather_dp_execution_tokens",
]
