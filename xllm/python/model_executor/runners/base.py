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

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.model_executor.execution_context import ExecutionMetadataBuilder
from xllm.python.model_executor.forward_context import EplbRuntimeState, LayerSynchronizer
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.model_output import DraftProposal, ModelOutput

ModelExecutionOutput = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | ModelOutput


@dataclass(frozen=True)
class GraphInputPreparation:
    """Device readiness and host failure propagation for deferred graph inputs."""

    # None when the model captures its readiness wait inside the graph.
    ready_event: Any
    completion: Future[None]


class BaseRunner(ABC):
    execution_metadata_builders: tuple[ExecutionMetadataBuilder, ...] = ()

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.device = device
        self.layer_caches: list[LayerCache] = []
        self.execution_metadata_builders = ()

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        self.layer_caches = layer_caches

    def bind_execution_metadata_builders(
        self,
        builders: Sequence[ExecutionMetadataBuilder],
    ) -> None:
        self.execution_metadata_builders = tuple(builders)

    @staticmethod
    def _slice_output(
        output: ModelExecutionOutput,
        num_tokens: int,
        *,
        num_reqs: int | None = None,
        clone: bool = False,
    ) -> ModelExecutionOutput:
        """Trim graph padding, optionally copying outputs out of graph storage."""

        def _trim(tensor: torch.Tensor, count: int) -> torch.Tensor:
            tensor = tensor[:count]
            return tensor.clone() if clone else tensor

        if isinstance(output, tuple):
            return _trim(output[0], num_tokens), _trim(output[1], num_tokens)
        if isinstance(output, torch.Tensor):
            return _trim(output, num_tokens)
        if not isinstance(output, ModelOutput):
            raise TypeError("Graph output must be a Tensor, a tuple of Tensors, or ModelOutput")

        def _trim_optional(tensor: torch.Tensor | None, count: int) -> torch.Tensor | None:
            return _trim(tensor, count) if tensor is not None else None

        proposal = None
        confidence = None
        if output.draft_proposal is not None or output.draft_confidence_probs is not None:
            if num_reqs is None:
                raise ValueError("Pipeline output with request-scoped fields requires num_reqs")
            if output.draft_proposal is not None:
                proposal = DraftProposal(
                    token_ids=_trim(output.draft_proposal.token_ids, num_reqs),
                    draft_probs=_trim_optional(output.draft_proposal.draft_probs, num_reqs),
                )
            confidence = _trim_optional(output.draft_confidence_probs, num_reqs)

        return ModelOutput(
            draft_proposal=proposal,
            draft_confidence_probs=confidence,
            hidden_states=_trim_optional(output.hidden_states, num_tokens),
            residual=_trim_optional(output.residual, num_tokens),
            aux_hidden_states=_trim_optional(output.aux_hidden_states, num_tokens),
        )

    @abstractmethod
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        pass
