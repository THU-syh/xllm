/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <optional>
#include <vector>

#include "core/framework/speculative/dflash_async_input_builder.h"
#include "runtime/dflash/dflash_worker_impl.h"

namespace xllm {

class ProcessGroup;

// DSpark = DFlash block-diffusion draft + a low-rank Markov head that adds a
// prefix-dependent bias to each block position's draft logits. It reuses the
// entire DFlash pipeline (prefill, context-K/V injection, validate, cache
// writes).
class DSparkWorkerImpl final : public DFlashWorkerImpl {
 public:
  DSparkWorkerImpl(const ParallelArgs& parallel_args,
                   const torch::Device& device,
                   const runtime::Options& options);

  ~DSparkWorkerImpl() override;

  // Load the draft weights, then (for reduced-vocabulary drafts) precompute the
  // draft-to-target id map from the checkpoint's ``d2t`` table.
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

 protected:
  // N-wide query block: anchor (slot 0) carries the last real token and
  // predicts the first draft token; every position is a prediction. DFlash's
  // prepare_query_inputs reads this hook to pick the block layout.
  bool sample_from_anchor() const override { return true; }

  // Build the DSpark query and produce the complete proposal block.
  DraftBlock run_decode_draft(const ForwardInput& input,
                              ForwardInput& validate_input) override;

  std::optional<ForwardOutput> step_decode(const ForwardInput& input) override;
  void prepare_validate_output_async(const ForwardInput& input) override;
  void finalize_validate_output_async(const ForwardInput& input,
                                      const SampleOutput& output,
                                      ForwardOutput& target_output) override;
  std::optional<ForwardOutput> step_prefill(const ForwardInput& input) override;
  std::optional<ForwardOutput> step_empty(const ForwardInput& input) override;
  void finish_empty_decode(const ForwardInput& input,
                           ForwardOutput& output,
                           std::optional<ForwardOutput>& draft_output) override;
  bool can_prepare_without_compute_stream_wait(
      const ForwardInput& input) const override;

 private:
  friend class DSparkWorkerImplTestPeer;

  bool can_prelaunch_draft_for_init(bool supports_device_inputs) const;
  bool can_prelaunch_draft_for_this_step(const ForwardInput& input) const;
  std::vector<int64_t> build_batch_to_pending_row_mapping(
      const ForwardInput& input) const;
  DraftBlock run_decode_draft_from_pending(const ForwardInput& input,
                                           const std::vector<int64_t>& rows);
  void commit_pending_context();
  DraftBlock execute_draft_query(const ForwardInput& input,
                                 ForwardInput query,
                                 bool prepared,
                                 ForwardInput* validate_input = nullptr);

  struct PendingContext {
    ForwardInput identity;
    torch::Tensor accepted_device;
    torch::Tensor accepted_host;
    torch::Tensor embeddings;
    torch::Tensor base_positions;
    StreamEventPtr d2h_ready;
    StreamEventPtr context_ready;
  };

  bool async_output_enabled_ = false;
  bool draft_prelaunch_enabled_for_init_ = false;
  std::optional<PendingContext> pending_context_;
  // One idle step's inputs; retired only after the following step is queued.
  ForwardOutput idle_decode_output_;
  torch::Tensor prepared_base_positions_;
  dflash_async::DraftCacheBlockTable prepared_context_block_table_;
  StreamEventPtr prepared_context_ready_;
  std::vector<torch::Tensor> consumed_context_tensors_;

  struct BlockSample {
    DraftProposal proposal;
    torch::Tensor confidence_probs;
  };

  BlockSample sample_block(const torch::Tensor& base_logits,
                           const torch::Tensor& last_hidden,
                           const torch::Tensor& anchor_token_ids,
                           const SamplingParameters& sampling_params) const;

  void synchronize_sampled_token_ids(
      torch::Tensor& sampled_token_ids,
      const SamplingParameters& sampling_params) const;

  ProcessGroup* sampling_process_group_ = nullptr;

  // draft-id -> target-id map (hot_token_id[i] = d2t[i] + i), precomputed from
  // the checkpoint's ``d2t`` table. Undefined for full-vocabulary drafts, in
  // which case no remapping is applied.
  torch::Tensor hot_token_id_;
};

}  // namespace xllm
