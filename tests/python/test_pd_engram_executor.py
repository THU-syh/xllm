# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PD history must be restored after warmup and before each execution path."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.model_executor.executor import ModelExecutor


@pytest.mark.parametrize("runner_kind", ["eager", "inductor", "graph"])
def test_pd_history_is_restored_before_execution(runner_kind: str) -> None:
    backend = object()
    metadata = SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        engram_history=[26, 9],
    )
    events: list[str] = []

    def restore(actual_backend: object, actual_metadata: object) -> None:
        assert actual_backend is backend
        assert actual_metadata is metadata
        events.append("restore")

    executor = ModelExecutor.__new__(ModelExecutor)
    executor._kv_bound = True
    executor.layerwise_split_size = 1
    executor._execution_metadata_builders = []
    executor._is_spec_draft = False
    executor.model = SimpleNamespace(model=SimpleNamespace(restore_engram_history=restore))
    executor._restore_engram_history = restore
    executor.attention_backend = backend
    executor.decode_graph_runner = None
    executor.inductor_runner = None
    runner = MagicMock()
    runner.supports_draft_to_target_overlap = False

    def execute(*_args: object, **_kwargs: object) -> torch.Tensor:
        assert events[-1] == "restore"
        events.append("execute")
        return torch.ones(1)

    runner.execute.side_effect = execute
    if runner_kind == "graph":
        runner.can_execute.return_value = True
        runner.warmup.side_effect = lambda *_args: events.append("warmup")
        executor.decode_graph_runner = runner
    elif runner_kind == "inductor":
        executor.inductor_runner = runner
    else:
        executor.eager_runner = runner

    result = executor.execute(torch.zeros(1), torch.zeros(1), metadata)
    assert torch.equal(result, torch.ones(1))
    assert events == (["warmup"] if runner_kind == "graph" else []) + ["restore", "execute"]
