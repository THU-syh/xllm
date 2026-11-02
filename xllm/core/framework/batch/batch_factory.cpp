/* Copyright 2025-2026 The xLLM Authors.

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

#include "batch_factory.h"

#include <absl/container/flat_hash_map.h>

#include <numeric>

#include "common/metrics.h"

namespace xllm {

namespace {

bool is_beam_search(const std::vector<std::shared_ptr<Request>>& requests) {
  for (const auto& request : requests) {
    if (request->check_beam_search()) {
      return true;
    }
  }
  return false;
}
// Non-owning Request identities are needed only while classifying this batch.
class BatchWorkloadCollector final {
 public:
  BatchWorkloadCollector(const std::vector<std::shared_ptr<Request>>& requests,
                         int32_t dp_size,
                         SchedulingSample* sample) {
    if (sample == nullptr) {
      return;
    }
    sample->workloads.resize(dp_size);
    workloads_ = &sample->workloads;
    num_requests_ = requests.size();
    request_stages_.resize(static_cast<size_t>(dp_size) * num_requests_);
    const size_t num_sequences =
        std::accumulate(requests.begin(),
                        requests.end(),
                        size_t{0},
                        [](size_t count, const auto& request) {
                          return count + request->sequences().size();
                        });
    sequence_owners_.reserve(num_sequences);
    for (size_t index = 0; index < requests.size(); ++index) {
      for (const auto& sequence : requests[index]->sequences()) {
        sequence_owners_.emplace(sequence.get(), index);
      }
    }
  }

  void record(const Sequence* sequence,
              size_t prompt_tokens,
              size_t decode_tokens) {
    if (workloads_ == nullptr) {
      return;
    }
    const int32_t dp = sequence->dp_rank();
    CHECK_GE(dp, 0);
    CHECK_LT(static_cast<size_t>(dp), workloads_->size());
    const auto owner = sequence_owners_.find(sequence);
    CHECK(owner != sequence_owners_.end());
    const size_t stage_index =
        static_cast<size_t>(dp) * num_requests_ + owner->second;
    SchedulingWorkload& workload = (*workloads_)[dp];
    workload.prompt_tokens += static_cast<int64_t>(prompt_tokens);
    workload.decode_tokens += static_cast<int64_t>(decode_tokens);
    uint8_t& stages = request_stages_[stage_index];
    if (prompt_tokens > 0 && (stages & kPrefill) == 0) {
      stages |= kPrefill;
      ++workload.prefill_requests;
    }
    if (decode_tokens > 0 && (stages & kDecode) == 0) {
      stages |= kDecode;
      ++workload.decode_requests;
    }
  }

 private:
  static constexpr uint8_t kPrefill = 1;
  static constexpr uint8_t kDecode = 2;
  std::vector<SchedulingWorkload>* workloads_ = nullptr;
  absl::flat_hash_map<const Sequence*, size_t> sequence_owners_;
  // One byte per (DP, Request). Count cross-DP requests independently.
  std::vector<uint8_t> request_stages_;
  size_t num_requests_ = 0;
};

}  // namespace

std::vector<Batch> BatchFactory::create_batches(
    const std::vector<std::shared_ptr<Request>>& running_requests,
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos,
    std::shared_ptr<SchedulingSample> sample) {
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  std::vector<Batch> batches(dp_size_);
  BatchWorkloadCollector workload_collector(
      running_requests, dp_size_, sample.get());
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    Sequence* sequence = running_sequences[i];
    const size_t token_budget = running_sequences_budgets[i];

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() >
                sequence->kv_state().kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() -
                  sequence->kv_state().kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;
    workload_collector.record(sequence, prompt_tokens, generated_tokens);

    // if dp enabled, each sequence is required to
    // dispatch to the same rank in the whole lifetime
    batches[sequence->dp_rank()].add(sequence, token_budget);
  }

  if (is_beam_search(running_requests)) {
    for (const auto& request : running_requests) {
      auto seq_group = request->sequence_group();
      int32_t dp_rank = seq_group->dp_rank();
      batches[dp_rank].add(seq_group);
    }
  }

  for (int32_t i = 0; i < dp_size_; i++) {
    batches[i].set_scheduling_sample(sample);
    if (batches[i].empty()) {
      continue;
    }
    if (swap_block_transfer_infos != nullptr &&
        swap_block_transfer_infos->size() == dp_size_) {
      batches[i].set_swap_block_transfer_infos(
          std::move(swap_block_transfer_infos->at(i)));
      swap_block_transfer_infos->at(i).clear();
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);

  if (running_sequences.size() > 0) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens / running_sequences.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens / running_sequences.size()));
  }

  return batches;
}

std::vector<Batch> BatchFactory::create_rec_batches(
    const std::vector<std::shared_ptr<Request>>& running_requests,
    const std::vector<Sequence*>& running_sequences,
    const std::vector<size_t>& running_sequences_budgets,
    std::vector<std::vector<BlockTransferInfo>>* swap_block_transfer_infos,
    std::shared_ptr<SchedulingSample> sample) {
  size_t num_prompt_tokens = 0;
  size_t num_generated_tokens = 0;
  std::vector<Batch> batches(dp_size_);
  BatchWorkloadCollector workload_collector(
      running_requests, dp_size_, sample.get());
  for (size_t i = 0; i < running_sequences.size(); ++i) {
    Sequence* sequence = running_sequences[i];
    const size_t token_budget = running_sequences_budgets[i];

    const size_t remaining_prompt_tokens =
        sequence->num_prompt_tokens() >
                sequence->kv_state().kv_cache_tokens_num()
            ? sequence->num_prompt_tokens() -
                  sequence->kv_state().kv_cache_tokens_num()
            : 0;
    const size_t prompt_tokens =
        std::min(remaining_prompt_tokens, token_budget);
    const size_t generated_tokens = token_budget - prompt_tokens;
    num_prompt_tokens += prompt_tokens;
    num_generated_tokens += generated_tokens;
    workload_collector.record(sequence, prompt_tokens, generated_tokens);

    batches[sequence->dp_rank()].set_batch_id();
  }

  for (const auto& request : running_requests) {
    auto seq_group = request->sequence_group();
    int32_t dp_rank = seq_group->dp_rank();
    batches[dp_rank].add(seq_group);
  }

  for (int32_t i = 0; i < dp_size_; i++) {
    batches[i].set_scheduling_sample(sample);
    if (batches[i].empty()) {
      continue;
    }
    if (swap_block_transfer_infos != nullptr &&
        swap_block_transfer_infos->size() == dp_size_) {
      batches[i].set_swap_block_transfer_infos(
          std::move(swap_block_transfer_infos->at(i)));
      swap_block_transfer_infos->at(i).clear();
    }
  }

  COUNTER_ADD(num_processing_tokens_total_prompt, num_prompt_tokens);
  COUNTER_ADD(num_processing_tokens_total_generated, num_generated_tokens);

  if (running_sequences.size() > 0) {
    HISTOGRAM_OBSERVE(
        num_prompt_tokens_per_request,
        static_cast<int64_t>(num_prompt_tokens / running_sequences.size()));
    HISTOGRAM_OBSERVE(
        num_generated_tokens_per_request,
        static_cast<int64_t>(num_generated_tokens / running_sequences.size()));
  }

  return batches;
}

}  // namespace xllm
