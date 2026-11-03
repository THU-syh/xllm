/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "platform/device.h"
#include "runtime/dflash/dflash2_worker_impl.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/executor.h"
#include "runtime/llm_worker_impl.h"

namespace xllm {

class DFlashWorkerImplTestPeer final {
 public:
  static ForwardInput prepare_query(DFlashWorkerImpl& worker,
                                    const ForwardInput& input) {
    worker.mask_token_id_ = 4;
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return query;
  }

  static ForwardInput prepare_json_states(DFlashWorkerImpl& worker,
                                          const ForwardInput& input,
                                          const torch::Tensor& tokens_host,
                                          const StreamEventPtr& ready) {
    auto guard = worker.compute_stream_->set_stream_guard();
    ForwardInput validation;
    worker.prepare_json_validation_states(
        input, tokens_host, ready, validation);
    CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    return validation;
  }
};

class DSparkWorkerImplTestPeer final {
 public:
  static SamplingParameters prepare_sampling_indices(DSparkWorkerImpl& worker,
                                                     int32_t num_tokens) {
    SamplingParameters parameters;
    worker.update_sampling_params(parameters, 1, num_tokens);
    return parameters;
  }

  static void initialize_decode_modes(DSparkWorkerImpl& worker,
                                      bool supports_device_inputs) {
    worker.async_output_enabled_ =
        worker.enable_schedule_overlap() &&
        !worker.options_.enable_adaptive_speculative_decode();
    worker.draft_prelaunch_enabled_for_init_ =
        worker.can_prelaunch_draft_for_init(supports_device_inputs);
  }

  static void set_parallel_sizes(DSparkWorkerImpl& worker,
                                 int32_t cp_size,
                                 int32_t kv_split_size) {
    worker.parallel_args_.cp_size(cp_size);
    worker.parallel_args_.kv_split_size(kv_split_size);
  }

  static bool can_prelaunch_draft_for_init(const DSparkWorkerImpl& worker,
                                           bool supports_device_inputs) {
    return worker.can_prelaunch_draft_for_init(supports_device_inputs);
  }

  static void configure_execution(DSparkWorkerImpl& worker,
                                  std::unique_ptr<LLMWorkerImpl> draft,
                                  std::unique_ptr<WorkerImpl> target,
                                  std::shared_ptr<EmbeddingCache> cache) {
    worker.draft_impl_ = std::move(draft);
    worker.impl_ = std::move(target);
    worker.context_ = worker.impl_->context_;
    worker.dtype_ = worker.impl_->dtype();
    worker.embedding_cache_ = std::move(cache);
    worker.mask_token_id_ = 31;
    worker.draft_sas_mode_ = dflash_detail::DSparkSasMode::NATIVE;
    worker.enable_fused_kernel_ = false;
    initialize_decode_modes(worker, true);
  }

  static std::optional<ForwardOutput> run_decode_step(
      DSparkWorkerImpl& worker,
      const ForwardInput& input) {
    return worker.step_decode(input);
  }

  static void install_output(DSparkWorkerImpl& worker, ForwardOutput output) {
    worker.last_step_output_ = std::move(output);
    worker.last_step_output_valid_ = true;
    worker.last_step_sample_sequence_ids_ = {"request"};
  }

  static dflash_async::DraftCacheBlockTable prepare_context_block_table(
      DSparkWorkerImpl& worker,
      const ForwardInput& input) {
    worker.prepare_validate_output_async(input);
    CHECK(worker.compute_stream_->wait_event(worker.prepared_context_ready_));
    return worker.prepared_context_block_table_;
  }

  static ForwardOutput finalize_output(DSparkWorkerImpl& worker,
                                       const ForwardInput& input,
                                       const SampleOutput& sample,
                                       std::unique_ptr<LLMWorkerImpl> draft) {
    worker.draft_impl_ = std::move(draft);
    worker.prepare_validate_output_async(input);
    ForwardOutput output;
    worker.finalize_validate_output_async(input, sample, output);
    return output;
  }

  static bool async_output_enabled(const DSparkWorkerImpl& worker) {
    return worker.async_output_enabled_;
  }

  static bool draft_prelaunch_enabled_for_init(const DSparkWorkerImpl& worker) {
    return worker.draft_prelaunch_enabled_for_init_;
  }

  static void finish_empty_decode(DSparkWorkerImpl& worker,
                                  ForwardOutput& output,
                                  std::optional<ForwardOutput>& draft_output,
                                  bool warmup = false) {
    ForwardInput input;
    input.input_params.meta.is_graph_warmup = warmup;
    worker.finish_empty_decode(input, output, draft_output);
  }

  static size_t idle_input_count(const DSparkWorkerImpl& worker) {
    return worker.idle_decode_output_.retained_inputs.size();
  }

  static bool can_prepare_without_wait(const DSparkWorkerImpl& worker,
                                       const ForwardInput& input) {
    return worker.can_prepare_without_compute_stream_wait(input);
  }

  static bool can_prelaunch_draft_for_this_step(const DSparkWorkerImpl& worker,
                                                const ForwardInput& input) {
    return worker.can_prelaunch_draft_for_this_step(input);
  }

  static std::vector<int64_t> pending_rows(const DSparkWorkerImpl& worker,
                                           const ForwardInput& input) {
    return worker.build_batch_to_pending_row_mapping(input);
  }

  static void commit_pending_context(DSparkWorkerImpl& worker) {
    worker.commit_pending_context();
  }

  static void install_pending_context(DSparkWorkerImpl& worker,
                                      ForwardInput identity,
                                      const torch::Tensor& accepted_device,
                                      const torch::Tensor& accepted_host,
                                      const torch::Tensor& embeddings,
                                      const torch::Tensor& positions,
                                      StreamEventPtr ready,
                                      std::shared_ptr<EmbeddingCache> cache) {
    initialize_decode_modes(worker, true);
    worker.embedding_cache_ = std::move(cache);
    worker.pending_context_.emplace();
    worker.pending_context_->identity = std::move(identity);
    worker.pending_context_->accepted_device = accepted_device;
    worker.pending_context_->accepted_host = accepted_host;
    worker.pending_context_->embeddings = embeddings;
    worker.pending_context_->base_positions = positions;
    worker.pending_context_->d2h_ready = ready;
    worker.pending_context_->context_ready = std::move(ready);
  }

  static bool has_pending(const DSparkWorkerImpl& worker) {
    return worker.pending_context_.has_value();
  }

  static void configure_draft_query(DSparkWorkerImpl& worker,
                                    int32_t mask_token_id) {
    ModelArgs draft_args;
    draft_args.draft_query_uses_decode(true);
    worker.draft_impl_->context_ =
        ModelContext(worker.parallel_args_,
                     draft_args,
                     QuantArgs{},
                     torch::TensorOptions().device(worker.device_));
    worker.mask_token_id_ = mask_token_id;
    worker.draft_sas_mode_ = dflash_detail::DSparkSasMode::NATIVE;
  }

  static dflash_async::NextDraftInputs prepare_pending_draft_inputs(
      DSparkWorkerImpl& worker,
      const ForwardInput& input) {
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    auto guard = worker.compute_stream_->set_stream_guard();
    CHECK(worker.compute_stream_->wait_event(
        worker.pending_context_->context_ready));
    const std::vector<int64_t> rows =
        worker.build_batch_to_pending_row_mapping(input);
    torch::Tensor row_indices =
        torch::tensor(rows, torch::kLong).to(worker.device_);
    dflash_async::DraftCacheBlockTable table =
        dflash_async::prepare_draft_cache_block_table(
            query.input_params.attention.device.block_tables,
            query.input_params.multi_block_tables,
            query.input_params.meta.num_sequences,
            worker.device_);
    dflash_async::NextDraftInputs result = dflash_async::prepare_next_draft(
        worker.pending_context_->accepted_device.index_select(0, row_indices),
        worker.pending_context_->base_positions.index_select(0, row_indices),
        table,
        worker.mask_token_id_,
        worker.options_.block_size(),
        worker.options_.num_speculative_tokens());
    CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    return result;
  }

  static ForwardInput prepare_host_corrected_draft_inputs(
      DSparkWorkerImpl& worker,
      ForwardInput input) {
    worker.prepare_decode_input(input);
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return query;
  }

  static size_t consumed_tensor_count(const DSparkWorkerImpl& worker) {
    return worker.consumed_context_tensors_.size();
  }
};

namespace {

class DSparkDraftInputParityTest
    : public ::testing::TestWithParam<std::tuple<int32_t, bool>> {};

TEST_P(DSparkDraftInputParityTest,
       PendingHistoryMatchesHostCorrectedQueriesAcrossSteps) {
  const int32_t block_size = std::get<0>(GetParam());
  const bool use_multi_block_tables = std::get<1>(GetParam());
  constexpr int32_t kQueryWidth = 15;
  constexpr int32_t kAcceptedWidth = kQueryWidth + 1;
  constexpr int32_t kInitialBatchSize = 3;
  constexpr int32_t kTableWidth = 256;
  constexpr int32_t kMaskTokenId = 32000;
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .speculative_algorithm("dspark")
      .num_speculative_tokens(kQueryWidth)
      .block_size(block_size);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::configure_draft_query(worker, kMaskTokenId);
  auto cache = std::make_shared<EmbeddingCache>(kInitialBatchSize, true);
  ForwardInput previous;
  previous.input_params.meta.num_sequences = kInitialBatchSize;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {0, 1, 2};
  previous.input_params.embedding.request_ids = {"first", "second", "third"};
  previous.sample_sequence_ids = {"sequence-0", "sequence-1", "sequence-2"};
  previous.sampling_params.all_greedy_sample = true;
  previous.positions_host =
      torch::tensor({0, block_size - 1, 2 * block_size - 2}, torch::kInt32);
  previous.input_params.attention.host.block_tables =
      torch::arange(kInitialBatchSize * kTableWidth, torch::kInt32)
          .reshape({kInitialBatchSize, kTableWidth})
          .flip({1});
  if (use_multi_block_tables) {
    previous.input_params.multi_block_tables = {
        previous.input_params.attention.host.block_tables.narrow(1, 0, 2)
            .contiguous(),
        previous.input_params.attention.host.block_tables};
  }

  for (int32_t step = 0; step < kAcceptedWidth; ++step) {
    SCOPED_TRACE(::testing::Message()
                 << "step=" << step << ", block_size=" << block_size
                 << ", multi_block_tables=" << use_multi_block_tables);
    const int32_t previous_batch_size =
        previous.input_params.meta.num_sequences;
    torch::Tensor accepted_host =
        torch::full({previous_batch_size, kAcceptedWidth}, -1, torch::kLong);
    auto accepted_values = accepted_host.accessor<int64_t, 2>();
    for (int32_t row = 0; row < previous_batch_size; ++row) {
      const int32_t accepted_count = (step + row * 5) % kAcceptedWidth + 1;
      for (int32_t offset = 0; offset < accepted_count; ++offset) {
        accepted_values[row][offset] =
            step * 1000 +
            previous.input_params.embedding.embedding_ids[row] * 100 + offset;
      }
    }
    torch::Tensor accepted_device = accepted_host.to(device.unwrap());
    torch::Tensor embeddings =
        torch::arange(previous_batch_size * kAcceptedWidth * 8, torch::kFloat)
            .reshape({previous_batch_size, kAcceptedWidth, 8})
            .to(device.unwrap());
    torch::Tensor base_positions =
        previous.positions_host.to(device.unwrap(), torch::kLong);
    auto compute = device.current_stream();
    StreamEventPtr ready = compute->record_event();
    ASSERT_NE(ready, nullptr);
    DSparkWorkerImplTestPeer::install_pending_context(worker,
                                                      previous,
                                                      accepted_device,
                                                      accepted_host,
                                                      embeddings,
                                                      base_positions,
                                                      ready,
                                                      cache);

    std::vector<int64_t> rows = step % 2 == 0 ? std::vector<int64_t>{0, 1, 2}
                                              : std::vector<int64_t>{2, 0, 1};
    if (step == kAcceptedWidth - 1) {
      rows.pop_back();
    }
    const int32_t batch_size = static_cast<int32_t>(rows.size());
    torch::Tensor row_indices = torch::tensor(rows, torch::kLong);
    ForwardInput input = previous;
    input.input_params.meta.num_sequences = batch_size;
    input.input_params.embedding.embedding_ids.clear();
    input.input_params.embedding.request_ids.clear();
    input.sample_sequence_ids.clear();
    for (int64_t row : rows) {
      input.input_params.embedding.embedding_ids.emplace_back(
          previous.input_params.embedding.embedding_ids[row]);
      input.input_params.embedding.request_ids.emplace_back(
          previous.input_params.embedding.request_ids[row]);
      input.sample_sequence_ids.emplace_back(previous.sample_sequence_ids[row]);
    }
    input.token_ids_host = torch::full({batch_size}, -1, torch::kInt32);
    input.positions_host =
        previous.positions_host.index_select(0, row_indices) + 1;
    input.token_ids = input.token_ids_host.to(device.unwrap());
    input.positions = input.positions_host.to(device.unwrap());
    input.input_params.attention.host.q_seq_lens.assign(batch_size, 1);
    input.input_params.attention.host.kv_seq_lens.clear();
    ExecutionBatchMetadata& execution_batch =
        input.input_params.execution_batch;
    execution_batch.num_reqs = batch_size;
    execution_batch.num_tokens = batch_size;
    execution_batch.num_scheduled_tokens.assign(batch_size, 1);
    execution_batch.num_computed_tokens.clear();
    execution_batch.is_prefilling.assign(batch_size, 0);
    execution_batch.query_start_loc = {0};
    const auto host_positions = input.positions_host.accessor<int32_t, 1>();
    for (int32_t row = 0; row < batch_size; ++row) {
      input.input_params.attention.host.kv_seq_lens.emplace_back(
          host_positions[row] + 1);
      execution_batch.num_computed_tokens.emplace_back(host_positions[row]);
      execution_batch.query_start_loc.emplace_back(row + 1);
    }
    input.input_params.attention.host.block_tables =
        previous.input_params.attention.host.block_tables.index_select(
            0, row_indices);
    input.input_params.multi_block_tables.clear();
    for (const torch::Tensor& table :
         previous.input_params.multi_block_tables) {
      input.input_params.multi_block_tables.emplace_back(
          table.index_select(0, row_indices));
    }

    ASSERT_EQ(DSparkWorkerImplTestPeer::pending_rows(worker, input), rows);
    dflash_async::NextDraftInputs pending =
        DSparkWorkerImplTestPeer::prepare_pending_draft_inputs(worker, input);
    DSparkWorkerImplTestPeer::commit_pending_context(worker);
    ASSERT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
    ForwardInput ordinary =
        DSparkWorkerImplTestPeer::prepare_host_corrected_draft_inputs(worker,
                                                                      input);
    EXPECT_TRUE(ordinary.input_params.meta.batch_forward_type.is_decode());
    EXPECT_TRUE(torch::equal(pending.query_tokens.cpu(),
                             ordinary.token_ids.cpu().to(torch::kLong)));
    EXPECT_TRUE(torch::equal(pending.query_positions.cpu(),
                             ordinary.positions.cpu().to(torch::kLong)));
    EXPECT_TRUE(torch::equal(
        pending.query_slots.cpu().to(torch::kLong),
        ordinary.input_params.attention.device.new_cache_slots.cpu().to(
            torch::kLong)));
    EXPECT_TRUE(torch::equal(
        pending.kv_lengths.cpu().to(torch::kLong),
        ordinary.input_params.attention.device.kv_seq_lens.cpu().to(
            torch::kLong)));

    previous = std::move(input);
    previous.positions_host =
        ordinary.positions_host.reshape({batch_size, kQueryWidth})
            .select(1, 0)
            .contiguous();
  }
}

INSTANTIATE_TEST_SUITE_P(CacheLayouts,
                         DSparkDraftInputParityTest,
                         ::testing::Values(std::make_tuple(4, false),
                                           std::make_tuple(4, true),
                                           std::make_tuple(128, false),
                                           std::make_tuple(128, true)));

TEST(DSparkDecodeStepTest, IdleDecodeRetainsInputsUntilNextSubmission) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(5);
  std::weak_ptr<ForwardInput> final_input;
  {
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
    auto target_input = std::make_shared<ForwardInput>();
    auto draft_input = std::make_shared<ForwardInput>();
    std::weak_ptr<ForwardInput> first_target = target_input;
    std::weak_ptr<ForwardInput> first_draft = draft_input;
    ForwardOutput output;
    output.retained_inputs.emplace_back(std::move(target_input));
    std::optional<ForwardOutput> draft_output = ForwardOutput();
    draft_output->retained_inputs.emplace_back(std::move(draft_input));
    DSparkWorkerImplTestPeer::finish_empty_decode(worker, output, draft_output);
    EXPECT_EQ(output.ready_event, nullptr);
    EXPECT_TRUE(output.retained_inputs.empty());
    EXPECT_TRUE(draft_output->retained_inputs.empty());
    EXPECT_FALSE(first_target.expired());
    EXPECT_FALSE(first_draft.expired());
    EXPECT_EQ(DSparkWorkerImplTestPeer::idle_input_count(worker), 2);

    auto next_input = std::make_shared<ForwardInput>();
    final_input = next_input;
    ForwardOutput next_output;
    next_output.retained_inputs.emplace_back(std::move(next_input));
    draft_output.reset();
    DSparkWorkerImplTestPeer::finish_empty_decode(
        worker, next_output, draft_output);
    EXPECT_TRUE(first_target.expired());
    EXPECT_TRUE(first_draft.expired());
    EXPECT_FALSE(final_input.expired());
    EXPECT_EQ(DSparkWorkerImplTestPeer::idle_input_count(worker), 1);
  }
  EXPECT_TRUE(final_input.expired());
}

TEST(DSparkDecodeStepTest, IdleDecodeKeepsSynchronousFallback) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(5);
  for (const bool warmup : {false, true}) {
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    DSparkWorkerImplTestPeer::initialize_decode_modes(worker, warmup);
    ForwardOutput output;
    output.retained_inputs.emplace_back(std::make_shared<ForwardInput>());
    std::optional<ForwardOutput> draft_output = ForwardOutput();
    DSparkWorkerImplTestPeer::finish_empty_decode(
        worker, output, draft_output, warmup);
    EXPECT_EQ(output.ready_event, nullptr);
    EXPECT_EQ(output.retained_inputs.size(), 1);
    EXPECT_EQ(DSparkWorkerImplTestPeer::idle_input_count(worker), 0);
  }
}

TEST(DSparkDecodeStepTest, AsyncOutputDoesNotRequireDraftPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = true;

  EXPECT_FALSE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, false);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  input.input_params.meta.is_graph_warmup = true;
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
}

TEST(DSparkDecodeStepTest, SamplingConstraintsDoNotDisableAsyncOutput) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = false;

  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));

  input.sampling_params.all_greedy_sample = true;
  input.sampling_params.repetition_penalties = torch::tensor({1.1f});
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
}

enum class DSparkDecodeFallback {
  DEVICE_INPUTS_UNSUPPORTED,
  NEW_REQUEST,
  BOOTSTRAP
};

class DSparkDecodeStepFallbackTest
    : public ::testing::TestWithParam<DSparkDecodeFallback> {};

TEST_P(DSparkDecodeStepFallbackTest,
       DraftSelectionLeavesPendingForExplicitCommit) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  auto cache = std::make_shared<EmbeddingCache>(1, true);
  ForwardInput previous;
  previous.input_params.meta.num_sequences = 1;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {0};
  previous.input_params.embedding.request_ids = {"previous"};
  previous.sampling_params.all_greedy_sample = true;
  torch::Tensor expected = torch::full({1, 16}, -1, torch::kLong);
  expected[0][0] = 42;
  expected[0][1] = 43;
  torch::Tensor accepted_device = expected.to(device.unwrap());
  torch::Tensor accepted_host =
      torch::empty(expected.sizes(), expected.options().pinned_memory(true));
  torch::Tensor embeddings =
      torch::ones({1, 16, 8}, torch::kFloat).to(device.unwrap());
  torch::Tensor positions =
      torch::tensor({127}, torch::kLong).to(device.unwrap());
  auto compute = device.current_stream();
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    accepted_host.copy_(accepted_device, true);
    ready = compute->record_event();
  }
  DSparkWorkerImplTestPeer::install_pending_context(worker,
                                                    previous,
                                                    accepted_device,
                                                    accepted_host,
                                                    embeddings,
                                                    positions,
                                                    ready,
                                                    cache);
  ForwardInput input = previous;
  switch (GetParam()) {
    case DSparkDecodeFallback::DEVICE_INPUTS_UNSUPPORTED:
      DSparkWorkerImplTestPeer::initialize_decode_modes(worker, false);
      break;

    case DSparkDecodeFallback::NEW_REQUEST:
      input.input_params.embedding.request_ids = {"new"};
      break;
    case DSparkDecodeFallback::BOOTSTRAP:
      input.input_params.embedding.mtp_bootstrap_embeddings = embeddings[0][0];
      break;
  }

  std::vector<int64_t> pending_rows;
  if (DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(worker,
                                                                  input)) {
    pending_rows = DSparkWorkerImplTestPeer::pending_rows(worker, input);
  }
  EXPECT_TRUE(pending_rows.empty());
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::has_pending(worker));
  DSparkWorkerImplTestPeer::commit_pending_context(worker);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
  EXPECT_TRUE(torch::equal(accepted_host, expected));
  const auto states = cache->read_decode_states({0}, {"previous"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 43);
  EXPECT_EQ(states[0].position_offset, 1);
  EXPECT_EQ(DSparkWorkerImplTestPeer::consumed_tensor_count(worker), 3);
}

INSTANTIATE_TEST_SUITE_P(
    DeviceInputsUnsupportedAndUnmatched,
    DSparkDecodeStepFallbackTest,
    ::testing::Values(DSparkDecodeFallback::DEVICE_INPUTS_UNSUPPORTED,

                      DSparkDecodeFallback::NEW_REQUEST,
                      DSparkDecodeFallback::BOOTSTRAP));

TEST(DSparkDecodeStepTest, AutomaticDecodeModesRespectConfigAndDeviceInputs) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  struct TestCase {
    std::string name;
    bool overlap;
    bool adaptive;
    int32_t cp_size;
    int32_t kv_split_size;
    std::string sampling_mode;
    bool device_inputs;
    bool async_output;
    bool prelaunch;
  };
  const std::vector<TestCase> cases = {
      {"supported", true, false, 1, 1, "greedy", true, true, true},
      {"async_only", true, false, 1, 1, "greedy", false, true, false},
      {"no_overlap", false, false, 1, 1, "greedy", true, false, false},
      {"adaptive", true, true, 1, 1, "greedy", true, false, false},
      {"context_parallel", true, false, 2, 1, "greedy", true, true, false},
      {"kv_split", true, false, 1, 2, "greedy", true, true, false},
      {"probabilistic", true, false, 1, 1, "probabilistic", true, true, false},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.name);
    ParallelArgs parallel_args(0, 2, nullptr);
    runtime::Options options;
    options.enable_schedule_overlap(test_case.overlap)
        .enable_adaptive_speculative_decode(test_case.adaptive)
        .num_speculative_tokens(15)
        .draft_sampling_mode(test_case.sampling_mode);
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    DSparkWorkerImplTestPeer::set_parallel_sizes(
        worker, test_case.cp_size, test_case.kv_split_size);
    DSparkWorkerImplTestPeer::initialize_decode_modes(worker,
                                                      test_case.device_inputs);
    EXPECT_EQ(DSparkWorkerImplTestPeer::async_output_enabled(worker),
              test_case.async_output);
    EXPECT_EQ(
        DSparkWorkerImplTestPeer::draft_prelaunch_enabled_for_init(worker),
        test_case.prelaunch);
    EXPECT_FALSE(
        DSparkWorkerImplTestPeer::can_prelaunch_draft_for_init(worker, false));
    EXPECT_EQ(
        DSparkWorkerImplTestPeer::draft_prelaunch_enabled_for_init(worker),
        test_case.prelaunch);
  }
}

TEST(DSparkDecodeStepTest, PdDecodeDoesNotDisableAsyncOutputOrPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_disagg_pd(true)
      .instance_role(InstanceRole::DECODE)
      .num_speculative_tokens(15)
      .draft_sampling_mode("greedy");
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = true;
  for (const bool supports_device_inputs : {false, true}) {
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    DSparkWorkerImplTestPeer::initialize_decode_modes(worker,
                                                      supports_device_inputs);
    EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
    EXPECT_EQ(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
                  worker, input),
              supports_device_inputs);
  }
}

TEST(DSparkDecodeStepTest, BatchConstraintsSelectNormalPathAndAllowRecovery) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = true;

  auto expect_normal_path = [&]() {
    EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
        worker, input));
    EXPECT_FALSE(
        DSparkWorkerImplTestPeer::can_prepare_without_wait(worker, input));
    EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
    EXPECT_TRUE(
        DSparkWorkerImplTestPeer::draft_prelaunch_enabled_for_init(worker));
  };
  auto expect_prelaunch = [&]() {
    EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
        worker, input));
    EXPECT_TRUE(
        DSparkWorkerImplTestPeer::can_prepare_without_wait(worker, input));
    EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  };
  expect_prelaunch();
  input.json_object_states.emplace_back();
  expect_normal_path();
  input.json_object_states.clear();
  expect_prelaunch();
  input.json_object_state_snapshots.emplace_back();
  expect_normal_path();
  input.json_object_state_snapshots.clear();
  expect_prelaunch();

  input.input_params.block_copy.src_block_indices = torch::tensor({0});
  expect_normal_path();
  input.input_params.block_copy.src_block_indices = torch::Tensor();
  expect_prelaunch();

  input.input_params.block_copy.swap_blocks.emplace_back(0, 1);
  expect_normal_path();
  input.input_params.block_copy.swap_blocks.clear();
  expect_prelaunch();

  input.sampling_params.all_greedy_sample = false;
  expect_normal_path();
  input.sampling_params.all_greedy_sample = true;
  expect_prelaunch();
  for (torch::Tensor* parameter : {&input.sampling_params.unique_token_ids,
                                   &input.sampling_params.frequency_penalties,
                                   &input.sampling_params.presence_penalties,
                                   &input.sampling_params.repetition_penalties,
                                   &input.sampling_params.filter_mask,
                                   &input.sampling_params.filter_bitmask}) {
    *parameter = torch::tensor({1.1f});
    expect_normal_path();
    *parameter = torch::Tensor();
    expect_prelaunch();
  }
}

TEST(DSparkDecodeStepTest, RestoreAndMultiTableDoNotDisableAsyncOrPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15).block_size(
      4);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 2;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.meta.requires_host_restore = true;
  input.input_params.embedding.embedding_ids = {0, 1};
  input.input_params.embedding.request_ids = {"first", "second"};
  input.sampling_params.all_greedy_sample = true;
  input.positions_host = torch::tensor({7, 31}, torch::kInt32);
  input.input_params.attention.host.kv_seq_lens = {8, 32};
  input.input_params.multi_block_tables = {
      torch::tensor({{7, 3}, {5, 9}}, torch::kInt32),
      torch::tensor({{11}, {13}}, torch::kInt32)};
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));

  auto compute = device.current_stream();
  auto guard = compute->set_stream_guard();
  dflash_async::DraftCacheBlockTable table =
      DSparkWorkerImplTestPeer::prepare_context_block_table(worker, input);
  ASSERT_TRUE(table.is_circular);
  ASSERT_EQ(table.blocks.device(), device.unwrap());
  torch::Tensor accepted =
      torch::tensor({{0, 42, -1}, {43, 44, 45}}, torch::kLong);
  dflash_async::ContextKvInputs context =
      dflash_async::prepare_context_kv(accepted.to(device.unwrap()),
                                       input.positions_host.to(device.unwrap()),
                                       table,
                                       4);
  torch::Tensor slots = context.slots.to(torch::kCPU).reshape({2, 3});
  const auto row_context = specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers expected;
  for (int32_t sequence = 0; sequence < 2; ++sequence) {
    for (int32_t offset = 0; offset < 3; ++offset) {
      if (accepted[sequence][offset].item<int64_t>() < 0) {
        EXPECT_EQ(slots[sequence][offset].item<int32_t>(), -1);
        break;
      }
      specBuilder::RowSpec row;
      row.seq_id = sequence;
      row.position_offset = offset;
      row.append_token = false;
      row.append_kv_len = false;
      specBuilder::append_decode_row(row_context, row, 4, expected);
      EXPECT_EQ(slots[sequence][offset].item<int32_t>(),
                expected.out_new_cache_slots.back());
    }
  }
  EXPECT_EQ(input.input_params.multi_block_tables.size(), 2);
  EXPECT_TRUE(input.input_params.meta.requires_host_restore);
}

TEST(DSparkDecodeStepTest,
     JsonDecodePreparationUsesNormalPathAndAllowsRecovery) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  ForwardInput input;
  input.token_ids = torch::tensor({0}, torch::kInt32);
  input.positions = torch::tensor({0}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.meta.q_max_seq_len = 1;
  input.input_params.meta.kv_max_seq_len = 1;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.input_params.attention.host.q_seq_lens = {1};
  input.input_params.attention.host.kv_seq_lens = {1};
  input.input_params.attention.host.new_cache_slots = {0};
  input.input_params.attention.host.block_tables =
      torch::tensor({{0}}, torch::kInt32);
  input.sampling_params.all_greedy_sample = true;
  JsonObjectGrammar grammar({"{", "}", "stop"}, {2});
  input.json_object_states = {grammar.initial_state()};

  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::initialize_decode_modes(worker, true);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  ForwardInput processed;
  worker.prepare_work_before_execute(input, processed);
  ASSERT_NE(processed.metadata_ready_event, nullptr);
  ASSERT_TRUE(processed.metadata_ready_event->synchronize());
  ASSERT_EQ(processed.json_object_states.size(), 1);
  EXPECT_TRUE(processed.json_object_states.front().initialized());
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(
      DSparkWorkerImplTestPeer::draft_prelaunch_enabled_for_init(worker));

  input.json_object_states.clear();
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
      worker, input));
  ForwardInput unconstrained;
  worker.prepare_work_before_execute(input, unconstrained);
  ASSERT_NE(unconstrained.metadata_ready_event, nullptr);
  ASSERT_TRUE(unconstrained.metadata_ready_event->synchronize());
  EXPECT_TRUE(unconstrained.json_object_states.empty());
}

TEST(DSparkDecodeStepTest,
     DeviceSamplingIndicesPreserveValuesWithoutPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  for (const int32_t num_tokens : {0, 1, 4, 64}) {
    SCOPED_TRACE(num_tokens);
    std::vector<int32_t> legacy_indices;
    legacy_indices.reserve(num_tokens);
    for (int32_t i = 0; i < num_tokens; i++) {
      legacy_indices.emplace_back(i);
    }
    torch::Tensor expected = torch::tensor(legacy_indices);
    SamplingParameters parameters =
        DSparkWorkerImplTestPeer::prepare_sampling_indices(worker, num_tokens);
    EXPECT_EQ(parameters.selected_token_idxes.scalar_type(),
              expected.scalar_type());
    EXPECT_EQ(parameters.selected_token_idxes.sizes(), expected.sizes());
    EXPECT_EQ(parameters.selected_token_idxes.device(), device.unwrap());
    EXPECT_TRUE(torch::equal(parameters.selected_token_idxes.cpu(), expected));
    EXPECT_TRUE(
        parameters.sample_idxes.is_same(parameters.selected_token_idxes));
  }
}

class ContextOnlyModel final : public CausalLM {
 public:
  explicit ContextOnlyModel(torch::TensorOptions options)
      : options_(std::move(options)) {}

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    LOG(FATAL) << "Forward is outside the async output test.";
    return {};
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    LOG(FATAL) << "Logits are outside the async output test.";
    return {};
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  ModelOutput write_context_kv(const torch::Tensor& hidden,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               std::vector<KVCache>&,
                               const ModelInputParams&) override {
    ModelOutput output;
    output.hidden_states = hidden;
    return output;
  }

 private:
  torch::TensorOptions options_;
};

class ContextOnlyWorker final : public LLMWorkerImpl {
 public:
  ContextOnlyWorker(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options)
      : LLMWorkerImpl(parallel_args, device, options) {
    model_ = std::make_unique<ContextOnlyModel>(
        torch::TensorOptions().device(device).dtype(torch::kFloat));
  }
};

struct DecodeStepTrace {
  std::vector<bool> pending_at_draft;
  std::vector<torch::Tensor> draft_tokens;
  std::vector<torch::Tensor> draft_positions;
  std::vector<torch::Tensor> draft_slots;
  std::vector<torch::Tensor> context_slots;
};

class DecodeStepDraftModel final : public CausalLM {
 public:
  DecodeStepDraftModel(torch::TensorOptions options,
                       std::shared_ptr<DecodeStepTrace> trace,
                       std::function<bool()> has_pending)
      : options_(std::move(options)),
        trace_(std::move(trace)),
        has_pending_(std::move(has_pending)) {}

  ModelOutput forward(const torch::Tensor& token_ids,
                      const torch::Tensor& positions,
                      std::vector<KVCache>&,
                      const ModelInputParams& params) override {
    trace_->pending_at_draft.emplace_back(has_pending_());
    trace_->draft_tokens.emplace_back(token_ids.clone());
    trace_->draft_positions.emplace_back(positions.clone());
    trace_->draft_slots.emplace_back(
        params.attention.device.new_cache_slots.clone());
    torch::Tensor anchors =
        token_ids.view({params.meta.num_sequences, -1}).select(1, 0);
    ModelOutput output;
    output.draft_proposal.emplace(
        torch::stack({anchors + 1, anchors + 2}, 1).to(torch::kLong));
    return output;
  }

  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    LOG(FATAL) << "The test draft returns a completed proposal.";
    return {};
  }

  ModelOutput write_context_kv(const torch::Tensor& hidden,
                               const torch::Tensor&,
                               const torch::Tensor& slots,
                               std::vector<KVCache>&,
                               const ModelInputParams&) override {
    trace_->context_slots.emplace_back(slots.clone());
    return ModelOutput(hidden);
  }

  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}

 private:
  torch::TensorOptions options_;
  std::shared_ptr<DecodeStepTrace> trace_;
  std::function<bool()> has_pending_;
};

class DecodeStepDraftWorker final : public LLMWorkerImpl {
 public:
  DecodeStepDraftWorker(const ParallelArgs& parallel_args,
                        const torch::Device& device,
                        const runtime::Options& options,
                        std::shared_ptr<DecodeStepTrace> trace,
                        std::function<bool()> has_pending)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat;
    auto tensor_options = torch::TensorOptions().device(device).dtype(dtype_);
    ModelArgs args;
    args.model_type("dspark_decode_step_test")
        .n_layers(1)
        .hidden_size(4)
        .draft_query_uses_decode(true);
    context_ = ModelContext(parallel_args, args, QuantArgs(), tensor_options);
    model_ = std::make_unique<DecodeStepDraftModel>(
        tensor_options, std::move(trace), std::move(has_pending));
    model_executor_ =
        std::make_unique<Executor>(model_.get(), args, device, options);
    is_spec_draft_ = true;
  }

  bool owns_npu_parallel_input_prepare() const override { return false; }
};

class DecodeStepTargetWorker final : public LLMWorkerImpl {
 public:
  DecodeStepTargetWorker(const ParallelArgs& parallel_args,
                         const torch::Device& device,
                         const runtime::Options& options)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kFloat;
    ModelArgs args;
    args.model_type("dspark_decode_step_test").n_layers(1).hidden_size(4);
    context_ =
        ModelContext(parallel_args,
                     args,
                     QuantArgs(),
                     torch::TensorOptions().device(device).dtype(dtype_));
  }

  bool owns_npu_parallel_input_prepare() const override { return false; }

  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& input,
      Stream& compute_stream) override {
    auto guard = compute_stream.set_stream_guard();
    CHECK(compute_stream.wait_event(input.metadata_ready_event));
    ForwardOutput output;
    output.sample_output.next_tokens = input.token_ids.to(torch::kLong) + 1;
    output.logits =
        torch::zeros({input.token_ids.numel(), 64},
                     input.token_ids.options().dtype(torch::kFloat));
    output.logits.scatter_(
        1, output.sample_output.next_tokens.unsqueeze(1), 100.0);
    output.sample_output.embeddings = input.positions.to(torch::kFloat)
                                          .unsqueeze(1)
                                          .expand({-1, 4})
                                          .contiguous();
    output.retained_inputs.emplace_back(std::make_shared<ForwardInput>(input));
    output.ready_event = compute_stream.record_event();
    return output;
  }
};

class DSparkDecodeExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_model_impl_ = ModelConfig::get_instance().model_impl();
    saved_manual_loader_ = LoadConfig::get_instance().enable_manual_loader();
    ModelConfig::get_instance().model_impl("native");
    LoadConfig::get_instance().enable_manual_loader(false);
    device_.set_device();
    device_.init_device_context();
  }

  void TearDown() override {
    EXPECT_EQ(device_.synchronize_default_stream(), 0);
    ModelConfig::get_instance().model_impl(saved_model_impl_);
    LoadConfig::get_instance().enable_manual_loader(saved_manual_loader_);
  }

  Device device_{0};
  std::string saved_model_impl_;
  bool saved_manual_loader_ = false;
};

TEST_F(DSparkDecodeExecutionTest, PrelaunchFallsBackForPenaltyAndRecovers) {
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_graph(false)
      .backend("llm")
      .enable_schedule_overlap(true)
      .enable_speculative_decode(true)
      .speculative_algorithm("DSpark")
      .draft_sampling_mode("greedy")
      .num_speculative_tokens(2)
      .block_size(4);
  DSparkWorkerImpl worker(parallel_args, device_.unwrap(), options);
  auto cache = std::make_shared<EmbeddingCache>(2, true);
  auto trace = std::make_shared<DecodeStepTrace>();
  trace->pending_at_draft.reserve(3);
  trace->draft_tokens.reserve(3);
  trace->draft_positions.reserve(3);
  trace->draft_slots.reserve(3);
  trace->context_slots.reserve(3);
  runtime::Options leaf_options = options;
  leaf_options.enable_schedule_overlap(false);
  DSparkWorkerImplTestPeer::configure_execution(
      worker,
      std::make_unique<DecodeStepDraftWorker>(
          parallel_args,
          device_.unwrap(),
          leaf_options,
          trace,
          [&worker]() {
            return DSparkWorkerImplTestPeer::has_pending(worker);
          }),
      std::make_unique<DecodeStepTargetWorker>(
          parallel_args, device_.unwrap(), leaf_options),
      cache);

  ForwardInput previous;
  previous.input_params.meta.num_sequences = 2;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {0, 1};
  previous.input_params.embedding.request_ids = {"first", "second"};
  previous.sample_sequence_ids = {"first-sequence", "second-sequence"};
  previous.positions_host = torch::tensor({2, 10}, torch::kInt);
  previous.input_params.attention.host.block_tables = torch::tensor(
      {{1, 3, 5, 7, 9, 11, 13, 15}, {2, 4, 6, 8, 10, 12, 14, 16}}, torch::kInt);
  torch::Tensor accepted_host =
      torch::tensor({{2, 3, 4}, {12, 13, 14}}, torch::kLong);
  torch::Tensor accepted_device = accepted_host.to(device_.unwrap());
  torch::Tensor embeddings =
      torch::zeros({2, 3, 4}, torch::TensorOptions().device(device_.unwrap()));
  torch::Tensor base_positions = previous.positions_host.to(device_.unwrap());
  auto ready = device_.current_stream()->record_event();
  ASSERT_NE(ready, nullptr);
  DSparkWorkerImplTestPeer::install_pending_context(worker,
                                                    previous,
                                                    accepted_device,
                                                    accepted_host,
                                                    embeddings,
                                                    base_positions,
                                                    ready,
                                                    cache);

  const std::vector<torch::Tensor> scheduler_positions = {
      torch::tensor({3, 11}, torch::kInt),
      torch::tensor({6, 14}, torch::kInt),
      torch::tensor({17, 9}, torch::kInt)};
  const std::vector<torch::Tensor> expected_anchors = {
      torch::tensor({4, 14}, torch::kLong),
      torch::tensor({7, 17}, torch::kLong),
      torch::tensor({20, 10}, torch::kLong)};
  std::vector<ForwardInput> step_inputs;
  std::vector<ForwardOutput> outputs;
  step_inputs.reserve(scheduler_positions.size());
  outputs.reserve(scheduler_positions.size());
  for (size_t step = 0; step < scheduler_positions.size(); ++step) {
    SCOPED_TRACE(step);
    ForwardInput input = previous;
    input.input_params.meta.batch_id = static_cast<uint64_t>(step + 1);
    input.token_ids_host = torch::full({2}, -1, torch::kInt);
    input.positions_host = scheduler_positions[step];
    input.token_ids = input.token_ids_host.to(device_.unwrap());
    input.positions = input.positions_host.to(device_.unwrap());
    input.input_params.attention.host.q_seq_lens = {1, 1};
    input.input_params.attention.host.kv_seq_lens = {
        input.positions_host[0].item<int32_t>() + 1,
        input.positions_host[1].item<int32_t>() + 1};
    input.input_params.meta.kv_max_seq_len =
        input.positions_host.max().item<int32_t>() + 1;
    ExecutionBatchMetadata& execution_batch =
        input.input_params.execution_batch;
    execution_batch.num_reqs = 2;
    execution_batch.num_tokens = 2;
    execution_batch.num_scheduled_tokens = {1, 1};
    execution_batch.num_computed_tokens = {
        input.positions_host[0].item<int32_t>(),
        input.positions_host[1].item<int32_t>()};
    execution_batch.query_start_loc = {0, 1, 2};
    execution_batch.is_prefilling = {0, 0};
    input.sampling_params.all_greedy_sample = true;
    input.sampling_params.do_sample =
        torch::zeros({2}, input.token_ids.options().dtype(torch::kBool));
    if (step == 1) {
      input.sampling_params.repetition_penalties = torch::tensor({1.1f, 1.1f});
    }
    if (step == 2) {
      input.input_params.embedding.embedding_ids = {1, 0};
      input.input_params.embedding.request_ids = {"second", "first"};
      input.sample_sequence_ids = {"second-sequence", "first-sequence"};
      input.input_params.attention.host.block_tables =
          previous.input_params.attention.host.block_tables.flip({0});
    }
    input.input_params.attention.device.block_tables =
        input.input_params.attention.host.block_tables.to(device_.unwrap());
    input.metadata_ready_event = device_.current_stream()->record_event();
    EXPECT_EQ(DSparkWorkerImplTestPeer::can_prelaunch_draft_for_this_step(
                  worker, input),
              step != 1);
    auto output = DSparkWorkerImplTestPeer::run_decode_step(worker, input);
    ASSERT_TRUE(output.has_value());
    ASSERT_NE(output->ready_event, nullptr);
    EXPECT_TRUE(output->requires_host_retirement);
    EXPECT_TRUE(output->sample_output.next_tokens.device().is_cpu());
    ASSERT_EQ(trace->pending_at_draft.size(), step + 1);
    EXPECT_EQ(trace->pending_at_draft.back(), step != 1);
    ASSERT_EQ(trace->context_slots.size(), step + 1);
    EXPECT_TRUE(DSparkWorkerImplTestPeer::has_pending(worker));
    EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
    const auto states =
        cache->read_decode_states(input.input_params.embedding.embedding_ids,
                                  input.input_params.embedding.request_ids);
    ASSERT_EQ(states.size(), 2);
    for (size_t row = 0; row < states.size(); ++row) {
      EXPECT_EQ(states[row].token_id,
                expected_anchors[step][row].item<int64_t>());
      EXPECT_EQ(states[row].position_offset, 2);
    }
    step_inputs.emplace_back(std::move(input));
    outputs.emplace_back(std::move(*output));
  }
  for (size_t step = 0; step < outputs.size(); ++step) {
    SCOPED_TRACE(step);
    ASSERT_TRUE(outputs[step].ready_event->synchronize());
    torch::Tensor actual_tokens = trace->draft_tokens[step].to(torch::kCPU);
    EXPECT_TRUE(
        torch::equal(actual_tokens.view({2, 2}).select(1, 0).to(torch::kLong),
                     expected_anchors[step]));
    torch::Tensor expected_positions =
        (scheduler_positions[step] + 2).unsqueeze(1) +
        torch::arange(2, torch::kInt).unsqueeze(0);
    EXPECT_TRUE(
        torch::equal(trace->draft_positions[step].to(torch::kCPU).view({2, 2}),
                     expected_positions));
    torch::Tensor block_indices =
        torch::floor_divide(expected_positions, 4).to(torch::kLong);
    const torch::Tensor& block_table =
        step_inputs[step].input_params.attention.host.block_tables;
    torch::Tensor expected_slots = block_table.gather(1, block_indices) * 4 +
                                   expected_positions.remainder(4);
    EXPECT_TRUE(torch::equal(
        trace->draft_slots[step].to(torch::kCPU).view({2, 2}), expected_slots));
    torch::Tensor context_positions =
        (scheduler_positions[step] + 2).unsqueeze(1) +
        torch::arange(3, torch::kInt).unsqueeze(0);
    torch::Tensor context_block_indices =
        torch::floor_divide(context_positions, 4).to(torch::kLong);
    torch::Tensor expected_context_slots =
        block_table.gather(1, context_block_indices) * 4 +
        context_positions.remainder(4);
    EXPECT_TRUE(
        torch::equal(trace->context_slots[step].to(torch::kCPU).view({2, 3}),
                     expected_context_slots));
    torch::Tensor expected_accepted =
        expected_anchors[step].unsqueeze(1) + torch::arange(1, 4, torch::kLong);
    EXPECT_TRUE(
        torch::equal(outputs[step].sample_output.next_tokens.to(torch::kLong),
                     expected_accepted));
  }
  DSparkWorkerImplTestPeer::commit_pending_context(worker);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
  const auto states = cache->read_decode_states({0, 1}, {"first", "second"});
  ASSERT_EQ(states.size(), 2);
  EXPECT_EQ(states[0].token_id, 13);
  EXPECT_EQ(states[1].token_id, 23);
  DSparkWorkerImplTestPeer::commit_pending_context(worker);
  EXPECT_EQ(trace->context_slots.size(), 3);
}

TEST(DSparkDecodeStepTest, AsyncAcceptedOutputReplaysJsonForMixedRows) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(2).block_size(
      128);
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  ForwardInput input;
  input.positions_host = torch::tensor({0, 0, 0}, torch::kInt32);
  input.input_params.meta.num_sequences = 3;
  input.input_params.attention.device.block_tables =
      torch::tensor({{0}, {1}, {2}}, torch::kInt32).to(device.unwrap());
  input.json_object_states = {grammar->initial_state(),
                              grammar->initial_state(),
                              JsonObjectGrammarState()};
  input.sample_sequence_ids = {"valid", "invalid", "plain"};
  const torch::Tensor accepted =
      torch::tensor({{0, 1, -1}, {0, 2, -1}, {2, 2, -1}}, torch::kLong);
  for (const auto dtype : {torch::kInt32, torch::kInt64}) {
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    SampleOutput sample;
    sample.next_tokens = accepted.to(device.unwrap(), dtype);
    sample.embeddings =
        torch::zeros({3, 3, 4}, torch::TensorOptions().device(device.unwrap()));
    ForwardOutput output = DSparkWorkerImplTestPeer::finalize_output(
        worker,
        input,
        sample,
        std::make_unique<ContextOnlyWorker>(
            parallel_args, device.unwrap(), options));
    ASSERT_EQ(output.json_object_errors.size(), 1);
    EXPECT_EQ(output.json_object_errors.front().sample_sequence_id, "invalid");
    ASSERT_NE(output.ready_event, nullptr);
    ASSERT_TRUE(output.ready_event->synchronize());
    EXPECT_TRUE(torch::equal(output.sample_output.next_tokens.to(torch::kLong),
                             accepted));
    EXPECT_TRUE(output.requires_host_retirement);
    EXPECT_TRUE(DSparkWorkerImplTestPeer::has_pending(worker));
    // Replaying the accepted block must not commit the input grammar state.
    EXPECT_FALSE(input.json_object_states.front().is_complete());
  }
}

TEST(DSparkDecodeStepTest, DraftQueryPreservesNonJsonSamplingMasks) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  const ParallelArgs parallel_args(0, 1, nullptr);
  for (const std::string& algorithm : {"DFlash2", "DSpark"}) {
    SCOPED_TRACE(algorithm);
    runtime::Options options;
    options.speculative_algorithm(algorithm)
        .num_speculative_tokens(2)
        .block_size(16);
    std::unique_ptr<DFlashWorkerImpl> worker;
    if (algorithm == "DSpark") {
      worker = std::make_unique<DSparkWorkerImpl>(
          parallel_args, device.unwrap(), options);
    } else {
      worker = std::make_unique<DFlash2WorkerImpl>(
          parallel_args, device.unwrap(), options);
    }
    ForwardInput input;
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.input_params.execution_batch.num_reqs = 1;
    input.input_params.execution_batch.num_tokens = 1;
    input.input_params.execution_batch.num_scheduled_tokens = {1};
    input.input_params.execution_batch.num_computed_tokens = {3};
    input.input_params.execution_batch.query_start_loc = {0, 1};
    input.input_params.execution_batch.is_prefilling = {0};
    input.token_ids_host = torch::tensor({0}, torch::kInt32);
    input.positions_host = torch::tensor({3}, torch::kInt32);
    input.token_ids = input.token_ids_host.to(device.unwrap());
    input.positions = input.positions_host.to(device.unwrap());
    input.input_params.attention.host.q_seq_lens = {1};
    input.input_params.attention.host.kv_seq_lens = {4};
    input.input_params.attention.host.block_tables =
        torch::arange(8, torch::kInt32).reshape({1, 8});
    input.sampling_params.all_greedy_sample = true;
    input.sampling_params.filter_mask =
        torch::tensor({{0.0f, 1.0f, 0.0f, 1.0f}});
    input.sampling_params.filter_bitmask = torch::tensor({{10}}, torch::kInt32);
    ForwardInput plain =
        DFlashWorkerImplTestPeer::prepare_query(*worker, input);
    EXPECT_TRUE(plain.sampling_params.filter_mask.is_same(
        input.sampling_params.filter_mask));
    EXPECT_TRUE(plain.sampling_params.filter_bitmask.is_same(
        input.sampling_params.filter_bitmask));
    input.json_object_states = {grammar->initial_state()};
    ForwardInput constrained =
        DFlashWorkerImplTestPeer::prepare_query(*worker, input);
    EXPECT_FALSE(constrained.sampling_params.filter_mask.defined());
    EXPECT_FALSE(constrained.sampling_params.filter_bitmask.defined());
    EXPECT_TRUE(input.sampling_params.filter_mask.defined());
    EXPECT_TRUE(input.sampling_params.filter_bitmask.defined());
  }
}

TEST(DSparkDecodeStepTest,
     SharedJsonReplayWaitsForCopyAndHandlesStridedTokens) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  ForwardInput input;
  input.input_params.meta.num_sequences = 3;
  input.json_object_states = {grammar->initial_state(),
                              grammar->initial_state(),
                              JsonObjectGrammarState()};
  JsonObjectGrammarState open = grammar->initial_state();
  ASSERT_TRUE(open.accept_token(0));
  JsonObjectGrammarState complete = open;
  ASSERT_TRUE(complete.accept_token(1));
  const std::vector<uint64_t> expected_fingerprints = {
      input.json_object_states[0].fingerprint(),
      open.fingerprint(),
      complete.fingerprint(),
      input.json_object_states[1].fingerprint(),
      open.fingerprint(),
      open.fingerprint()};
  const ParallelArgs parallel_args(0, 1, nullptr);
  for (const std::string& algorithm : {"DFlash2", "DSpark"}) {
    for (const auto dtype : {torch::kInt32, torch::kInt64}) {
      SCOPED_TRACE(algorithm);
      runtime::Options options;
      options.speculative_algorithm(algorithm).num_speculative_tokens(2);
      std::unique_ptr<DFlashWorkerImpl> worker;
      if (algorithm == "DSpark") {
        worker = std::make_unique<DSparkWorkerImpl>(
            parallel_args, device.unwrap(), options);
      } else {
        worker = std::make_unique<DFlash2WorkerImpl>(
            parallel_args, device.unwrap(), options);
      }
      // A transposed pinned destination exercises both dtype conversion and
      // contiguous normalization, without relying on the usual staging layout.
      torch::Tensor host =
          torch::full({2, 3},
                      3,
                      torch::TensorOptions().dtype(dtype).pinned_memory(true))
              .transpose(0, 1);
      ASSERT_FALSE(host.is_contiguous());
      const torch::Tensor drafted =
          torch::tensor({{0, 1}, {0, 2}, {2, 2}}, torch::kLong)
              .to(device.unwrap(), dtype);
      auto copy_stream = device.get_stream_from_pool();
      StreamEventPtr ready;
      {
        auto guard = copy_stream->set_stream_guard();
        torch::Tensor matrix = torch::ones({2048, 2048},
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat16)
                                               .device(device.unwrap()));
        for (int32_t iteration = 0; iteration < 16; ++iteration) {
          matrix = torch::matmul(matrix, matrix);
        }
        host.copy_(drafted, true);
        ready = copy_stream->record_event();
      }
      ASSERT_NE(ready, nullptr);
      ForwardInput validation = DFlashWorkerImplTestPeer::prepare_json_states(
          *worker, input, host, ready);
      ASSERT_EQ(validation.json_object_states.size(), 9);
      EXPECT_EQ(validation.json_object_invalid_draft,
                (std::vector<uint8_t>{0, 0, 0, 1, 0, 0}));
      for (size_t row = 0; row < expected_fingerprints.size(); ++row) {
        EXPECT_EQ(validation.json_object_states[row].fingerprint(),
                  expected_fingerprints[row]);
      }
      for (size_t row = 6; row < 9; ++row) {
        EXPECT_FALSE(validation.json_object_states[row].initialized());
      }
      EXPECT_TRUE(validation.sampling_params.filter_bitmask.defined());
      EXPECT_FALSE(validation.sampling_params.filter_mask.defined());
      EXPECT_FALSE(input.json_object_states.front().is_complete());
    }
  }
}

TEST(DSparkDecodeStepTest, JsonConsumesHostTokensOnlyAfterReadyEvent) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "stop"},
      std::unordered_set<int32_t>{2});
  ForwardInput input;
  input.json_object_states = {grammar->initial_state()};
  input.sample_sequence_ids = {"request"};
  input.sample_prior_output_rows = {0};
  torch::Tensor host_tokens = torch::full(
      {1, 3},
      -1,
      torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
  auto copy_stream = device.get_stream_from_pool();
  ForwardOutput output;
  torch::Tensor device_tokens =
      torch::tensor({{0, 1, -1}}, torch::kLong).to(device.unwrap());
  {
    auto copy_guard = copy_stream->set_stream_guard();
    torch::Tensor matrix = torch::ones(
        {2048, 2048},
        torch::TensorOptions().dtype(torch::kFloat16).device(device.unwrap()));
    for (int32_t iteration = 0; iteration < 16; ++iteration) {
      matrix = torch::matmul(matrix, matrix);
    }
    host_tokens.copy_(device_tokens, true);
    output.ready_event = copy_stream->record_event();
  }
  output.sample_output.next_tokens = host_tokens;
  output.requires_host_retirement = true;
  DSparkWorkerImplTestPeer::install_output(worker, std::move(output));
  worker.update_json_object_states_by_last_step_output(input);
  ASSERT_TRUE(input.json_object_errors.empty());
  EXPECT_TRUE(input.json_object_states.front().is_complete());
  EXPECT_TRUE(
      torch::equal(host_tokens, torch::tensor({{0, 1, -1}}, torch::kLong)));
}

}  // namespace
}  // namespace xllm
