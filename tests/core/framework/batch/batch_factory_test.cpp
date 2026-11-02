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

#include "core/framework/batch/batch_factory.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/metrics.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request(int32_t dp_rank,
                                      size_t num_sequences = 1) {
  RequestState state(std::vector<int32_t>(4, 1));
  state.seq_capacity = 16;
  state.n = state.best_of = num_sequences;
  state.logprobs = false;
  state.stopping_checker.set_max_generated_tokens(4);
  auto request = std::make_shared<Request>(std::to_string(dp_rank),
                                           /*x_request_id=*/"",
                                           /*x_request_time=*/"",
                                           std::move(state));
  request->expand_sequences(/*share_prefix=*/false);
  for (const auto& sequence : request->sequences()) {
    sequence->set_dp_rank(dp_rank);
  }
  return request;
}

TEST(BatchFactoryWorkloadTest, SplitsBudgetsAndDeduplicatesRequestsPerDp) {
  auto multi = make_request(/*dp_rank=*/0, /*num_sequences=*/3);
  auto decode = make_request(/*dp_rank=*/1);
  const std::vector<std::shared_ptr<Request>> requests = {multi, decode};
  const std::vector<Sequence*> sequences = {multi->sequences()[0].get(),
                                            multi->sequences()[1].get(),
                                            multi->sequences()[2].get(),
                                            decode->sequences()[0].get()};
  sequences[0]->kv_state().set_kv_cache_tokens_num(1);
  sequences[1]->kv_state().set_kv_cache_tokens_num(4);
  sequences[2]->kv_state().set_kv_cache_tokens_num(3);
  sequences[3]->kv_state().set_kv_cache_tokens_num(4);
  const std::vector<size_t> budgets = {2, 1, 2, 2};
  BatchFactory* factory = BatchFactory::get_instance(/*dp_size=*/2);
  for (const bool rec : {false, true}) {
    SCOPED_TRACE(rec);
    const double prompt_before =
        COUNTER_VALUE(num_processing_tokens_total_prompt);
    const double decode_before =
        COUNTER_VALUE(num_processing_tokens_total_generated);
    auto sample = std::make_shared<SchedulingSample>();
    const auto batches =
        rec ? factory->create_rec_batches(
                  requests, sequences, budgets, nullptr, sample)
            : factory->create_batches(
                  requests, sequences, budgets, nullptr, sample);
    ASSERT_EQ(batches.size(), 2u);
    ASSERT_EQ(sample->workloads.size(), 2u);
    const SchedulingWorkload expected[] = {{3, 2, 1, 1}, {0, 2, 0, 1}};
    for (size_t dp = 0; dp < batches.size(); ++dp) {
      EXPECT_EQ(batches[dp].scheduling_sample(), sample);
      const SchedulingWorkload& actual = sample->workloads[dp];
      EXPECT_EQ(actual.prompt_tokens, expected[dp].prompt_tokens);
      EXPECT_EQ(actual.decode_tokens, expected[dp].decode_tokens);
      EXPECT_EQ(actual.prefill_requests, expected[dp].prefill_requests);
      EXPECT_EQ(actual.decode_requests, expected[dp].decode_requests);
    }
    EXPECT_DOUBLE_EQ(
        COUNTER_VALUE(num_processing_tokens_total_prompt) - prompt_before, 3);
    EXPECT_DOUBLE_EQ(
        COUNTER_VALUE(num_processing_tokens_total_generated) - decode_before,
        4);
  }
}

TEST(SchedulingSampleTest, RecordsOuterRoundOnceAcrossRequestSplits) {
  for (const int32_t dp : {0, 1}) {
    SCOPED_TRACE(dp);
    auto first = make_request(dp);
    auto second = make_request(1 - dp);
    second->sequences().front()->set_dp_rank(dp);
    auto sample = std::make_shared<SchedulingSample>();
    sample->metrics = std::make_shared<SchedulingMetricState>();
    const std::string server = std::to_string(745 + dp);
    sample->metrics->server = server;
    const auto batches =
        BatchFactory::get_instance(/*dp_size=*/2)
            ->create_batches({first, second},
                             {first->sequences().front().get(),
                              second->sequences().front().get()},
                             {2, 4},
                             nullptr,
                             sample);
    ASSERT_TRUE(batches[1 - dp].empty());
    const auto parts = batches[dp].split_by_request();
    ASSERT_EQ(parts.size(), 2u);
    const double mismatches_before =
        COUNTER_VALUE(scheduling_sample_mismatch_total);
    for (const Batch& part : parts) {
      EXPECT_EQ(part.size(), 1u);
      record_batch_scheduling(batches);
    }
    EXPECT_TRUE(sample->recorded);
    EXPECT_EQ(MULTI_HISTOGRAM_scheduling_latency_microseconds
                  .get_stats({server, "submitted"})
                  ->count(),
              1);
    EXPECT_DOUBLE_EQ(
        MULTI_GAUGE_scheduling_workload_tokens_latest
            .get_stats({server, std::to_string(dp), "prompt_tokens"})
            ->get_value(),
        6);
    EXPECT_DOUBLE_EQ(COUNTER_VALUE(scheduling_sample_mismatch_total),
                     mismatches_before);
  }
}

TEST(SchedulingSampleTest, SkipsMismatchedBatchSamples) {
  auto request = make_request(/*dp_rank=*/0);
  BatchFactory* factory = BatchFactory::get_instance(/*dp_size=*/2);
  const double mismatches_before =
      COUNTER_VALUE(scheduling_sample_mismatch_total);
  for (int32_t mismatch = 0; mismatch < 3; ++mismatch) {
    SCOPED_TRACE(mismatch);
    auto sample = std::make_shared<SchedulingSample>();
    sample->metrics = std::make_shared<SchedulingMetricState>();
    sample->metrics->server = "743";
    auto batches = factory->create_batches(
        {request}, {request->sequences().front().get()}, {4}, nullptr, sample);
    if (mismatch == 0) {
      sample->workloads.pop_back();
    } else if (mismatch == 1) {
      batches.back() = Batch{};
    } else {
      ++sample->workloads.front().prompt_tokens;
    }
    record_batch_scheduling(batches);
    EXPECT_FALSE(sample->recorded);
    EXPECT_FALSE(MULTI_HISTOGRAM_scheduling_latency_microseconds.has_stats(
        {"743", "submitted"}));
    EXPECT_DOUBLE_EQ(COUNTER_VALUE(scheduling_sample_mismatch_total),
                     mismatches_before + mismatch + 1);
  }
}

TEST(SchedulingSampleTest, SubmitsOnceAndKeepsRecBudget) {
  auto request = make_request(/*dp_rank=*/0);
  auto sample = std::make_shared<SchedulingSample>();
  sample->metrics = std::make_shared<SchedulingMetricState>();
  sample->metrics->server = "7";
  auto batches = BatchFactory::get_instance(/*dp_size=*/2)
                     ->create_rec_batches({request},
                                          {request->sequences().front().get()},
                                          {4},
                                          nullptr,
                                          sample);
  batches.front().refresh_sequences_from_groups();
  auto& latency = MULTI_HISTOGRAM_scheduling_latency_microseconds;
  record_batch_scheduling(
      batches, /*is_graph_warmup=*/true, /*use_outer_budget=*/true);
  EXPECT_FALSE(latency.has_stats({"7", "submitted"}));
  record_batch_scheduling(
      batches, /*is_graph_warmup=*/false, /*use_outer_budget=*/true);
  record_batch_scheduling(
      batches, /*is_graph_warmup=*/false, /*use_outer_budget=*/true);
  EXPECT_EQ(latency.get_stats({"7", "submitted"})->count(), 1);
  EXPECT_DOUBLE_EQ(MULTI_GAUGE_scheduling_workload_tokens_latest
                       .get_stats({"7", "0", "prompt_tokens"})
                       ->get_value(),
                   4);
  EXPECT_EQ(MULTI_HISTOGRAM_scheduling_workload_tokens
                .get_stats({"7", "1", "decode_tokens"})
                ->count(),
            1);
  EXPECT_DOUBLE_EQ(MULTI_GAUGE_scheduling_workload_requests_latest
                       .get_stats({"7", "0", "prefill_requests"})
                       ->get_value(),
                   1);
  EXPECT_EQ(MULTI_HISTOGRAM_scheduling_workload_requests
                .get_stats({"7", "0", "prefill_requests"})
                ->count(),
            1);
  SchedulingSample empty;
  empty.metrics = sample->metrics;
  empty.workloads.resize(2);
  SchedulingSample late = empty;
  record_scheduling_metrics(empty, /*empty=*/true);
  EXPECT_EQ(latency.get_stats({"7", "empty"})->count(), 1);
  close_scheduling_metrics(*sample->metrics);
  record_scheduling_metrics(late);
  EXPECT_EQ(latency.get_stats({"7", "submitted"})->count(), 1);
}

}  // namespace
}  // namespace xllm
