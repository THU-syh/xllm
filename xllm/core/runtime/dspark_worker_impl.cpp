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

#include "runtime/dspark_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>

#include <algorithm>
#include <iterator>
#include <optional>
#include <unordered_map>
#include <utility>

#include "common/metrics.h"
#include "core/framework/model/model_sampling_input.h"
#include "core/framework/speculative/adaptive_speculative_controller.h"
#include "core/framework/speculative/dflash_async_input_builder.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/process_group.h"
#include "runtime/llm_worker_impl.h"
#include "util/timer.h"

namespace xllm {

DSparkWorkerImpl::DSparkWorkerImpl(const ParallelArgs& parallel_args,
                                   const torch::Device& device,
                                   const runtime::Options& options)
    : DFlashWorkerImpl(parallel_args, device, options),
      sampling_process_group_(parallel_args.tp_group_ != nullptr
                                  ? parallel_args.tp_group_
                                  : parallel_args.process_group_) {}

DSparkWorkerImpl::~DSparkWorkerImpl() {
  if (idle_decode_output_.ready_event != nullptr) {
    CHECK(idle_decode_output_.ready_event->synchronize());
  }
  if (pending_context_.has_value()) {
    CHECK(pending_context_->context_ready->synchronize());
  }
  if (!consumed_context_tensors_.empty()) {
    compute_stream_->synchronize();
  }
}

bool DSparkWorkerImpl::init_model(const std::string& model_weights_path,
                                  int32_t random_seed,
                                  MasterStatus master_status) {
  bool result = DFlashWorkerImpl::init_model(
      model_weights_path, random_seed, master_status);

  // Reduced-vocabulary drafts ship a "d2t" table mapping each draft id to a
  // target id via diffs (target_id = d2t[i] + i). Precompute the full map so
  // sampled draft ids can be remapped to target space. Full-vocab drafts have
  // no "d2t"; hot_token_id_ stays undefined and no remapping is applied.
  if (draft_impl_->get_status() == WorkerImpl::Status::LOADED) {
    CHECK_GT(options_.num_speculative_tokens(), 0)
        << "DSpark requires num_speculative_tokens > 0.";
    async_output_enabled_ = enable_schedule_overlap() &&
                            !options_.enable_adaptive_speculative_decode();
    const bool supports_device_inputs =
        draft_impl_->supports_device_only_decode_inputs();
    draft_prelaunch_enabled_for_init_ =
        can_prelaunch_draft_for_init(supports_device_inputs);
    draft_impl_->set_device_only_decode_inputs(
        draft_prelaunch_enabled_for_init_);
    LOG(INFO) << "DSpark async output enabled=" << async_output_enabled_;
    LOG(INFO) << "DSpark device-only decode inputs supported="
              << supports_device_inputs << ", draft prelaunch enabled="
              << draft_prelaunch_enabled_for_init_;
    auto model_loader = ModelLoader::create(model_weights_path);
    for (const auto& state_dict : model_loader->get_state_dicts()) {
      torch::Tensor d2t_tensor = state_dict->get_tensor("d2t");
      if (d2t_tensor.defined()) {
        torch::Tensor arange_tensor = torch::arange(d2t_tensor.size(0));
        hot_token_id_ = (d2t_tensor + arange_tensor).to(device_, torch::kLong);
        LOG(INFO) << "DSparkWorkerImpl: loaded d2t, draft vocab size "
                  << hot_token_id_.size(0);
        break;
      }
    }
  }

  return result;
}

bool DSparkWorkerImpl::can_prelaunch_draft_for_init(
    bool supports_device_inputs) const {
  const bool draft_prelaunch_pending_support_gate =
      parallel_args_.cp_size() == 1 &&
      parallel_args_.kv_split_size_effective() == 1;
  return async_output_enabled_ && draft_prelaunch_pending_support_gate &&
         draft_sampling_mode_ == DraftSamplingMode::GREEDY &&
         supports_device_inputs;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_decode(
    const ForwardInput& raw_input) {
  ForwardInput input = raw_input;
  ForwardInput validate_input;
  std::vector<int64_t> pending_rows;
  if (can_prelaunch_draft_for_this_step(input)) {
    pending_rows = build_batch_to_pending_row_mapping(input);
  }

  DraftBlock draft_block;
  if (!pending_rows.empty()) {
    draft_block = run_decode_draft_from_pending(input, pending_rows);
    commit_pending_context();
    prepare_decode_input(input);
    prepare_validate_inputs(input, validate_input);
  } else {
    commit_pending_context();
    prepare_decode_input(input);
    draft_block = run_decode_draft(input, validate_input);
  }

  const bool async_output =
      async_output_enabled_ && !input.input_params.meta.is_graph_warmup;
  auto output = run_validate(input, draft_block, validate_input, async_output);
  if (output.has_value()) {
    output->failed_request_ids.insert(output->failed_request_ids.end(),
                                      draft_block.failed_request_ids.begin(),
                                      draft_block.failed_request_ids.end());
  }
  return output;
}

DSparkWorkerImpl::DraftBlock DSparkWorkerImpl::run_decode_draft(
    const ForwardInput& input,
    ForwardInput& validate_input) {
  // Same input build as DFlash, but sample_from_anchor()==true makes the query
  // block N-wide and every position predicts a draft token.
  ForwardInput query_input;
  prepare_query_inputs(input, query_input);
  query_input.input_params.is_dflash_proposal = true;

  return execute_draft_query(
      input, std::move(query_input), false, &validate_input);
}

DSparkWorkerImpl::DraftBlock DSparkWorkerImpl::execute_draft_query(
    const ForwardInput& input,
    ForwardInput query_input,
    bool prepared,
    ForwardInput* validate_input) {
  Timer timer;
  const int32_t batch_size = input.input_params.meta.num_sequences;
  const int32_t num_speculative_tokens = options_.num_speculative_tokens();
  CHECK_GT(num_speculative_tokens, 0)
      << "DSpark requires num_speculative_tokens > 0.";
  const bool want_confidence = draft_impl_->has_dspark_confidence_head() &&
                               adaptive_spec_controller_ != nullptr &&
                               adaptive_spec_controller_->enabled();
  query_input.skip_sampling_for_logits_only = true;
  query_input.return_selected_hidden = want_confidence;

  ForwardInput processed_input;
  if (prepared) {
    processed_input = std::move(query_input);
  } else {
    draft_impl_->prepare_work_before_execute_on_stream(
        query_input, processed_input, *prepare_stream_);
  }
  ModelSamplingInput sampling;
  sampling.parameters = input.sampling_params;
  if (draft_sampling_mode_ == DraftSamplingMode::GREEDY) {
    force_greedy_draft_sampling(sampling.parameters);
  }
  sampling.parameters.return_probs = draft_probs_required(
      draft_sampling_mode_, sampling.parameters.all_greedy_sample);
  sampling.return_confidence = want_confidence;
  processed_input.input_params.sampling = std::move(sampling);

  std::optional<ForwardOutput> draft_output =
      draft_impl_->execute_no_sync_on_stream(processed_input,
                                             *compute_stream_,
                                             /*record_ready_event=*/false);
  CHECK(draft_output.has_value())
      << "DSpark draft forward must return an output.";
  if (validate_input != nullptr) {
    prepare_validate_inputs(input, *validate_input);
  }

  DraftBlock draft_block;
  if (draft_output->draft_proposal.has_value()) {
    draft_block.proposal = std::move(*draft_output->draft_proposal);
    draft_block.confidence_probs =
        std::move(draft_output->draft_confidence_probs);
  } else {
    // Native fallback: the model returns logits, so sample the proposal here.
    CHECK(draft_output->logits.defined())
        << "DSpark draft forward must return logits.";
    const int64_t num_rows = draft_output->logits.size(/*dim=*/0);
    CHECK_EQ(num_rows % num_speculative_tokens, 0)
        << "DSpark draft logits rows must be divisible by "
           "num_speculative_tokens.";
    CHECK_EQ(num_rows / num_speculative_tokens, batch_size)
        << "DSpark draft logits batch must match the decode batch.";
    torch::Tensor base_logits = draft_output->logits.view(
        {batch_size, num_speculative_tokens, draft_output->logits.size(-1)});
    torch::Tensor last_hidden;
    if (want_confidence) {
      if (draft_output->selected_hidden.defined() &&
          draft_output->selected_hidden.size(0) == num_rows) {
        last_hidden = draft_output->selected_hidden.view(
            {batch_size,
             num_speculative_tokens,
             draft_output->selected_hidden.size(-1)});
      } else {
        LOG_FIRST_N(WARNING, 1)
            << "DSpark: draft did not expose selected_hidden; ConfidenceHead "
            << "disabled for adaptive pruning, using proposal probs.";
      }
    }

    BlockSample sampled = [&]() {
      c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
      torch::Tensor anchor_token_ids =
          processed_input.token_ids.view({batch_size, num_speculative_tokens})
              .select(1, 0)
              .to(device_, torch::kLong);
      SamplingParameters sampling_params_on_device =
          prepared ? input.sampling_params
                   : input.sampling_params.to(base_logits.device(),
                                              base_logits.scalar_type());
      if (draft_sampling_mode_ == DraftSamplingMode::GREEDY) {
        force_greedy_draft_sampling(sampling_params_on_device);
      }
      return sample_block(base_logits,
                          last_hidden,
                          anchor_token_ids,
                          sampling_params_on_device);
    }();
    draft_block.proposal = std::move(sampled.proposal);
    draft_block.confidence_probs = std::move(sampled.confidence_probs);
  }
  draft_block.failed_request_ids = std::move(draft_output->failed_request_ids);
  draft_block.retained_inputs = take_retained_inputs(*draft_output);
  COUNTER_ADD(speculative_execution_latency_seconds_draft,
              timer.elapsed_seconds());
  return draft_block;
}

bool DSparkWorkerImpl::can_prepare_without_compute_stream_wait(
    const ForwardInput& input) const {
  return can_prelaunch_draft_for_this_step(input);
}

bool DSparkWorkerImpl::can_prelaunch_draft_for_this_step(
    const ForwardInput& input) const {
  // Draft sampling consumes JSON grammar state, which is not available before
  // the previous accepted-token D2H and Host state update.
  if (!draft_prelaunch_enabled_for_init_ ||
      input.input_params.meta.is_graph_warmup ||
      !input.input_params.meta.batch_forward_type.is_decode() ||
      input.input_params.meta.num_sequences <= 0 ||
      input.input_params.embedding.embedding_ids.empty() ||
      input.input_params.embedding.request_ids.size() !=
          input.input_params.embedding.embedding_ids.size() ||
      input.input_params.embedding.mtp_bootstrap_embeddings.defined()) {
    return false;
  }
  const auto& block_copy = input.input_params.block_copy;
  if (!block_copy.swap_blocks.empty() ||
      (block_copy.src_block_indices.defined() &&
       block_copy.src_block_indices.numel() > 0)) {
    return false;
  }
  if (!input.json_object_states.empty() ||
      !input.json_object_state_snapshots.empty()) {
    return false;
  }
  const auto& sampling = input.sampling_params;
  return sampling.all_greedy_sample && !sampling.unique_token_ids.defined() &&
         !sampling.frequency_penalties.defined() &&
         !sampling.presence_penalties.defined() &&
         !sampling.repetition_penalties.defined() &&
         !sampling.filter_mask.defined() && !sampling.filter_bitmask.defined();
}

std::vector<int64_t> DSparkWorkerImpl::build_batch_to_pending_row_mapping(
    const ForwardInput& input) const {
  if (!pending_context_.has_value()) {
    return {};
  }
  const ForwardInput& previous = pending_context_->identity;
  const auto& previous_requests = previous.input_params.embedding.request_ids;
  const auto& current_requests = input.input_params.embedding.request_ids;
  const auto& previous_embeddings =
      previous.input_params.embedding.embedding_ids;
  const auto& current_embeddings = input.input_params.embedding.embedding_ids;
  const bool use_sequence_ids = !previous.sample_sequence_ids.empty() ||
                                !input.sample_sequence_ids.empty();
  if (use_sequence_ids &&
      (previous.sample_sequence_ids.size() != previous_requests.size() ||
       input.sample_sequence_ids.size() != current_requests.size())) {
    return {};
  }
  std::unordered_map<std::string, int64_t> previous_rows;
  previous_rows.reserve(previous_requests.size());
  for (size_t row = 0; row < previous_requests.size(); ++row) {
    const std::string& identity = use_sequence_ids
                                      ? previous.sample_sequence_ids[row]
                                      : previous_requests[row];
    if (!previous_rows.emplace(identity, static_cast<int64_t>(row)).second) {
      return {};
    }
  }
  std::vector<int64_t> rows;
  rows.reserve(current_requests.size());
  for (size_t row = 0; row < current_requests.size(); ++row) {
    const std::string& identity = use_sequence_ids
                                      ? input.sample_sequence_ids[row]
                                      : current_requests[row];
    const auto found = previous_rows.find(identity);
    if (found == previous_rows.end()) {
      return {};
    }
    const size_t previous_row = static_cast<size_t>(found->second);
    if (previous_requests[previous_row] != current_requests[row] ||
        previous_embeddings[previous_row] != current_embeddings[row]) {
      return {};
    }
    rows.emplace_back(found->second);
  }
  return rows;
}

void DSparkWorkerImpl::commit_pending_context() {
  if (!pending_context_.has_value()) {
    return;
  }
  CHECK(pending_context_->d2h_ready->synchronize())
      << "Failed to synchronize DSpark accepted tokens";
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  embedding_cache_->write_target_context(
      pending_context_->identity.input_params.embedding.embedding_ids,
      pending_context_->identity.input_params.embedding.request_ids,
      pending_context_->accepted_host,
      pending_context_->embeddings,
      options_.num_speculative_tokens());
  consumed_context_tensors_.reserve(consumed_context_tensors_.size() + 3);
  consumed_context_tensors_.emplace_back(pending_context_->accepted_device);
  consumed_context_tensors_.emplace_back(pending_context_->embeddings);
  consumed_context_tensors_.emplace_back(pending_context_->base_positions);
  pending_context_.reset();
}

DFlashWorkerImpl::DraftBlock DSparkWorkerImpl::run_decode_draft_from_pending(
    const ForwardInput& input,
    const std::vector<int64_t>& rows) {
  ForwardInput query;
  prepare_query_inputs(input, query);
  query.input_params.is_dflash_proposal = true;
  query.skip_sampling_for_logits_only = true;
  query.return_selected_hidden = false;
  ForwardInput prepared;
  draft_impl_->prepare_work_before_execute_on_stream(
      query, prepared, *prepare_stream_, true, false);
  ForwardInput device_input = input;
  dflash_async::DraftCacheBlockTable block_table;
  torch::Tensor row_indices;
  torch::Tensor host_row_indices;
  const bool block_parallel_rows =
      draft_sas_mode_ == dflash_detail::DSparkSasMode::COMPATIBILITY;
  {
    c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
    // A block-parallel proposal exports one metadata row per query token.
    // Device preparation consumes one ring table per logical sequence.
    std::vector<torch::Tensor> tables =
        prepared.input_params.multi_block_tables;
    if (block_parallel_rows) {
      const int32_t width = options_.num_speculative_tokens();
      for (torch::Tensor& table : tables) {
        table = table.slice(/*dim=*/0, /*start=*/0, table.size(0), width);
      }
    }
    block_table = dflash_async::prepare_draft_cache_block_table(
        prepared.input_params.attention.device.block_tables,
        tables,
        input.input_params.meta.num_sequences,
        device_);
    if (block_parallel_rows) {
      // Reuse the staged SWA table for Python graph refresh. A CPU table
      // would introduce a blocking H2D copy on the compute stream while
      // submitting the proposal, serializing it behind the target verify.
      CHECK(!prepared.input_params.multi_block_tables.empty());
      prepared.input_params.multi_block_tables.front() =
          block_table.blocks.repeat_interleave(
              options_.num_speculative_tokens(),
              /*dim=*/0);
    }
    device_input.sampling_params =
        input.sampling_params.to(draft_impl_->device(), torch::kFloat32);
    force_greedy_draft_sampling(device_input.sampling_params);
    host_row_indices = torch::empty(
        {static_cast<int64_t>(rows.size())},
        torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
    std::copy(rows.begin(), rows.end(), host_row_indices.data_ptr<int64_t>());
    row_indices = host_row_indices.to(device_, torch::kLong, true);
    prepared.metadata_ready_event = prepare_stream_->record_event();
  }
  CHECK(prepared.metadata_ready_event != nullptr);
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(prepared.metadata_ready_event));
  torch::Tensor accepted =
      pending_context_->accepted_device.index_select(0, row_indices);
  torch::Tensor base_positions =
      pending_context_->base_positions.index_select(0, row_indices);
  dflash_async::NextDraftInputs next =
      dflash_async::prepare_next_draft(accepted,
                                       base_positions,
                                       block_table,
                                       mask_token_id_,
                                       options_.block_size(),
                                       options_.num_speculative_tokens());
  prepared.token_ids.copy_(next.query_tokens);
  prepared.positions.copy_(next.query_positions);
  prepared.input_params.attention.device.new_cache_slots.copy_(
      next.query_slots);
  if (block_parallel_rows) {
    // Match build_query_rows: every query row sees the whole block.
    prepared.input_params.attention.device.kv_seq_lens.copy_(
        next.kv_lengths.repeat_interleave(options_.num_speculative_tokens()));
  } else {
    prepared.input_params.attention.device.kv_seq_lens.copy_(next.kv_lengths);
  }
  prepared.input_params.attn_metadata.reset();
  prepared.metadata_ready_event.reset();
  prepared.device_tensors_ready = true;
  DraftBlock block =
      execute_draft_query(device_input, std::move(prepared), true);
  block.retained_tensors.reserve(block.retained_tensors.size() + 8);
  block.retained_tensors.emplace_back(std::move(block_table.blocks));
  block.retained_tensors.emplace_back(std::move(row_indices));
  block.retained_tensors.emplace_back(std::move(host_row_indices));
  block.retained_tensors.emplace_back(std::move(accepted));
  block.retained_tensors.emplace_back(std::move(base_positions));
  block.retained_tensors.emplace_back(std::move(next.anchor_tokens));
  block.retained_tensors.emplace_back(pending_context_->accepted_device);
  block.retained_tensors.emplace_back(pending_context_->base_positions);
  VLOG(1) << "DSpark step-local draft submitted: batch_id="
          << input.input_params.meta.batch_id << ", sequences=" << rows.size();
  return block;
}

void DSparkWorkerImpl::prepare_validate_output_async(
    const ForwardInput& input) {
  c10::StreamGuard prepare_guard = prepare_stream_->set_stream_guard();
  torch::Tensor host_base_positions = torch::empty(
      input.positions_host.sizes(),
      input.positions_host.options().dtype(torch::kLong).pinned_memory(true));
  host_base_positions.copy_(input.positions_host);
  prepared_base_positions_ = host_base_positions.to(device_, true);
  prepared_context_block_table_ = dflash_async::prepare_draft_cache_block_table(
      input.input_params.attention.device.block_tables,
      input.input_params.multi_block_tables,
      input.input_params.meta.num_sequences,
      device_);
  prepared_context_ready_ = prepare_stream_->record_event();
  CHECK(prepared_context_ready_ != nullptr);
}

void DSparkWorkerImpl::finalize_validate_output_async(
    const ForwardInput& input,
    const SampleOutput& output,
    ForwardOutput& target_output) {
  CHECK(!pending_context_.has_value());
  CHECK(prepared_context_ready_ != nullptr);
  c10::StreamGuard compute_guard = compute_stream_->set_stream_guard();
  CHECK(compute_stream_->wait_event(prepared_context_ready_));
  PendingContext pending;
  pending.identity = input;
  pending.accepted_device = output.next_tokens;
  pending.embeddings = output.embeddings;
  pending.base_positions = std::move(prepared_base_positions_);
  pending.accepted_host = torch::empty(
      output.next_tokens.sizes(),
      output.next_tokens.options().device(torch::kCPU).pinned_memory(true));
  pending.accepted_host.copy_(pending.accepted_device, true);
  pending.d2h_ready = compute_stream_->record_event();
  CHECK(pending.d2h_ready != nullptr);
  dflash_async::ContextKvInputs context =
      dflash_async::prepare_context_kv(pending.accepted_device,
                                       pending.base_positions,
                                       prepared_context_block_table_,
                                       options_.block_size());
  ModelOutput written = draft_impl_->write_context_kv(
      output.embeddings.reshape({output.next_tokens.numel(), -1}),
      context.positions,
      context.slots,
      input.input_params);
  CHECK(written.hidden_states.defined());
  pending.context_ready = compute_stream_->record_event();
  CHECK(pending.context_ready != nullptr);

  if (!input.json_object_states.empty()) {
    // Replay only after the asynchronous accepted-token copy is visible on
    // host. The context KV write can finish independently through
    // context_ready.
    CHECK(pending.d2h_ready->synchronize());
    check_json_accepted_tokens(
        input,
        pending.accepted_host.to(torch::kInt64).contiguous(),
        target_output);
  }

  auto retained = std::make_shared<ForwardInput>(input);
  retained->retained_device_tensors.reserve(
      retained->retained_device_tensors.size() + 6 +
      consumed_context_tensors_.size());
  retained->retained_device_tensors.emplace_back(
      std::move(prepared_context_block_table_.blocks));
  retained->retained_device_tensors.emplace_back(pending.accepted_device);
  retained->retained_device_tensors.emplace_back(pending.embeddings);
  retained->retained_device_tensors.emplace_back(pending.base_positions);
  retained->retained_device_tensors.emplace_back(std::move(context.positions));
  retained->retained_device_tensors.emplace_back(std::move(context.slots));
  retained->retained_device_tensors.insert(
      retained->retained_device_tensors.end(),
      std::make_move_iterator(consumed_context_tensors_.begin()),
      std::make_move_iterator(consumed_context_tensors_.end()));
  consumed_context_tensors_.clear();
  target_output.retained_inputs.emplace_back(std::move(retained));
  target_output.sample_output = output;
  target_output.sample_output.next_tokens = pending.accepted_host;
  target_output.sample_output.embeddings = torch::Tensor();
  target_output.ready_event = pending.context_ready;
  target_output.requires_host_retirement = true;
  target_output.defer_block_speculative_metrics = true;
  pending_context_ = std::move(pending);
  prepared_context_ready_.reset();
  VLOG(1) << "DSpark step-local output submitted: batch_id="
          << input.input_params.meta.batch_id
          << ", sequences=" << input.input_params.meta.num_sequences;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_prefill(
    const ForwardInput& input) {
  commit_pending_context();
  std::optional<ForwardOutput> output = DFlashWorkerImpl::step_prefill(input);
  consumed_context_tensors_.clear();
  return output;
}

std::optional<ForwardOutput> DSparkWorkerImpl::step_empty(
    const ForwardInput& input) {
  commit_pending_context();
  if (!input.input_params.meta.batch_forward_type.is_decode()) {
    return DFlashWorkerImpl::step_empty(input);
  }
  ForwardInput empty_input = input;
  empty_input.input_params.is_dflash_proposal = true;
  ModelSamplingInput sampling;
  force_greedy_draft_sampling(sampling.parameters);
  empty_input.input_params.sampling = std::move(sampling);
  return DFlashWorkerImpl::step_empty(empty_input);
}

void DSparkWorkerImpl::finish_empty_decode(
    const ForwardInput& input,
    ForwardOutput& output,
    std::optional<ForwardOutput>& draft_output) {
  if (!draft_prelaunch_enabled_for_init_ ||
      input.input_params.meta.is_graph_warmup) {
    DFlashWorkerImpl::finish_empty_decode(input, output, draft_output);
    return;
  }
  StreamEventPtr ready = compute_stream_->record_event();
  CHECK(ready != nullptr);
  // The next draft must reach DP graph admission while the current target is
  // running, including on idle ranks. Retire the previous idle inputs only
  // after this step's draft and target have both been submitted.
  if (idle_decode_output_.ready_event != nullptr) {
    CHECK(idle_decode_output_.ready_event->synchronize());
  }
  idle_decode_output_.ready_event = std::move(ready);
  idle_decode_output_.retained_inputs = take_retained_inputs(output);
  output.ready_event.reset();
  if (draft_output.has_value()) {
    transfer_retained_inputs(idle_decode_output_, draft_output.value());
  }
  // This worker owns retirement. The empty output carries no token that the
  // generic scheduler must wait to read, and no input it must keep alive.
}

DSparkWorkerImpl::BlockSample DSparkWorkerImpl::sample_block(
    const torch::Tensor& base_logits,
    const torch::Tensor& last_hidden,
    const torch::Tensor& anchor_token_ids,
    const SamplingParameters& sampling_params) const {
  CHECK_EQ(base_logits.dim(), 3)
      << "DSpark base_logits must be [num_reqs, n_spec, draft_vocab].";
  CHECK_EQ(anchor_token_ids.dim(), 1)
      << "DSpark anchor_token_ids must be [num_reqs].";
  CHECK_EQ(anchor_token_ids.size(0), base_logits.size(0))
      << "DSpark anchor token batch must match base_logits.";
  CHECK_EQ(anchor_token_ids.scalar_type(), torch::kLong)
      << "DSpark anchor_token_ids must use int64.";
  CHECK_EQ(anchor_token_ids.device(), base_logits.device())
      << "DSpark anchor_token_ids and base_logits must share a device.";

  const int64_t num_reqs = base_logits.size(/*dim=*/0);
  const int64_t num_speculative_tokens = base_logits.size(/*dim=*/1);
  const int64_t draft_vocab_size = base_logits.size(/*dim=*/2);
  const bool with_confidence =
      last_hidden.defined() && draft_impl_->has_dspark_confidence_head();
  if (with_confidence) {
    CHECK_EQ(last_hidden.dim(), 3)
        << "DSpark last_hidden must be [num_reqs, n_spec, hidden_size].";
    CHECK_EQ(last_hidden.size(0), num_reqs)
        << "DSpark last_hidden batch mismatch.";
    CHECK_EQ(last_hidden.size(1), num_speculative_tokens)
        << "DSpark last_hidden n_spec mismatch.";
  }
  const bool need_draft_probs = draft_probs_required(
      draft_sampling_mode_, sampling_params.all_greedy_sample);
  SamplingParameters step_sampling_params = sampling_params;
  const torch::TensorOptions index_options =
      torch::TensorOptions().dtype(torch::kInt).device(base_logits.device());
  step_sampling_params.selected_token_idxes = torch::empty({0}, index_options);
  step_sampling_params.sample_idxes = torch::empty({0}, index_options);
  step_sampling_params.return_probs = need_draft_probs;
  step_sampling_params.logprobs = false;
  step_sampling_params.max_top_logprobs = 0;
  step_sampling_params.use_beam_search = false;

  torch::Tensor token_ids = torch::empty(
      {num_reqs, num_speculative_tokens},
      torch::TensorOptions().dtype(torch::kLong).device(base_logits.device()));
  std::vector<torch::Tensor> probs_steps;
  if (need_draft_probs) {
    probs_steps.reserve(num_speculative_tokens);
  }
  // Filled after the sample loop by a single batched ConfidenceHead call; stays
  // undefined when confidence is disabled.
  torch::Tensor confidence_probs;

  using ISlice = torch::indexing::Slice;
  Sampler sampler;
  // Loop-invariant: only needed when remapping reduced-vocab draft ids.
  const int64_t target_vocab_size =
      hot_token_id_.defined() ? context_.get_model_args().vocab_size() : 0;
  torch::Tensor previous_token_ids = anchor_token_ids;
  for (int64_t token_idx = 0; token_idx < num_speculative_tokens; ++token_idx) {
    torch::Tensor markov_bias =
        draft_impl_->dspark_markov_bias(previous_token_ids);
    CHECK_EQ(markov_bias.dim(), 2)
        << "DSpark Markov bias must be [num_reqs, draft_vocab].";
    CHECK_EQ(markov_bias.size(0), num_reqs)
        << "DSpark Markov bias batch must match base_logits.";
    CHECK_EQ(markov_bias.size(1), draft_vocab_size)
        << "DSpark Markov bias width must equal the draft vocabulary.";

    torch::Tensor step_logits =
        base_logits.select(/*dim=*/1, /*index=*/token_idx) + markov_bias;
    SampleOutput sample_output =
        sampler.forward(step_logits, step_sampling_params);
    torch::Tensor sampled_token_ids = sample_output.next_tokens;
    synchronize_sampled_token_ids(sampled_token_ids, step_sampling_params);

    // Reduced-vocab draft: map draft ids to target ids before they reach the
    // proposal and the Markov feedback embedding, both indexed in target space.
    // Scatter the draft probs into the full target vocab so the verifier's
    // (p-q)+ rejection residual is computed in target space.
    if (hot_token_id_.defined()) {
      sampled_token_ids = hot_token_id_.index_select(0, sampled_token_ids);
      if (need_draft_probs && sample_output.probs.defined()) {
        torch::Tensor full_vocab_probs =
            torch::zeros({sample_output.probs.size(0), target_vocab_size},
                         sample_output.probs.options());
        full_vocab_probs.index_put_({ISlice(), hot_token_id_},
                                    sample_output.probs);
        sample_output.probs = full_vocab_probs;
      }
    }

    token_ids.index_put_({ISlice(), token_idx}, sampled_token_ids);
    if (need_draft_probs) {
      probs_steps.emplace_back(std::move(sample_output.probs));
    }
    previous_token_ids = sampled_token_ids;
  }

  // ConfidenceHead over the whole block, batched. Equivalent to computing it
  // per step inside the loop on (last_hidden[:, k], prev_k) — prev_k is the
  // anchor for step 0 and the draft token sampled at step k-1 otherwise — but
  // as a single linear+sigmoid instead of gamma small kernel launches. The
  // inputs don't depend on the sampling of the same step, so hoisting is exact.
  if (with_confidence) {
    torch::Tensor prev_matrix = torch::cat(
        {anchor_token_ids.view({num_reqs, 1}),
         token_ids.slice(/*dim=*/1, /*start=*/0, num_speculative_tokens - 1)},
        /*dim=*/1);  // [num_reqs, num_spec]
    confidence_probs =
        draft_impl_->dspark_confidence_probs(last_hidden, prev_matrix);
    CHECK_EQ(confidence_probs.dim(), 2)
        << "batched confidence probs must be [num_reqs, num_spec]";
    CHECK_EQ(confidence_probs.size(0), num_reqs)
        << "batched confidence probs batch mismatch";
    CHECK_EQ(confidence_probs.size(1), num_speculative_tokens)
        << "batched confidence probs n_spec mismatch";
  }

  return {need_draft_probs ? DraftProposal(std::move(token_ids),
                                           torch::stack(probs_steps, /*dim=*/1))
                           : DraftProposal(std::move(token_ids)),
          std::move(confidence_probs)};
}

void DSparkWorkerImpl::synchronize_sampled_token_ids(
    torch::Tensor& sampled_token_ids,
    const SamplingParameters& sampling_params) const {
  if (sampling_params.all_greedy_sample || sampling_process_group_ == nullptr ||
      sampling_process_group_->world_size() <= 1) {
    return;
  }
  sampled_token_ids = sampled_token_ids.contiguous();
  sampling_process_group_->broadcast(sampled_token_ids, /*root_rank=*/0);
}

}  // namespace xllm
