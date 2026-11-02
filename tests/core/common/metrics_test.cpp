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

#include "common/metrics.h"

#include <brpc/channel.h>
#include <brpc/server.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstddef>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "util/scope_guard.h"

namespace xllm {
namespace {

class PausedMetricDumper final : public bvar::Dumper {
 public:
  PausedMetricDumper(std::promise<void>& entered,
                     std::shared_future<void> resume)
      : entered_(entered), resume_(std::move(resume)) {}
  bool dump(const std::string& name, const butil::StringPiece&) override {
    if (name ==
        "scheduling_latency_microseconds_latency_server_740_round_submitted") {
      entered_.set_value();
      resume_.wait();
    }
    return true;
  }

 private:
  std::promise<void>& entered_;
  std::shared_future<void> resume_;
};

TEST(SchedulingMetricsTest, SkipsInvalidSamples) {
  auto state = std::make_shared<SchedulingMetricState>();
  state->server = "742";
  SchedulingSample sample;
  sample.workloads = {{4, 0, 1, 0}};
  record_scheduling_metrics(sample);
  EXPECT_FALSE(sample.recorded);

  sample.metrics = state;
  sample.workloads.clear();
  record_scheduling_metrics(sample);
  EXPECT_FALSE(sample.recorded);
  auto* recorder = MULTI_HISTOGRAM_scheduling_latency_microseconds.get_stats(
      {"742", "submitted"});
  ASSERT_NE(recorder, nullptr);
  EXPECT_EQ(recorder->count(), 0);

  sample.workloads = {{4, 0, 1, 0}};
  sample.started_at = std::chrono::steady_clock::now() + std::chrono::hours(1);
  record_scheduling_metrics(sample);
  EXPECT_FALSE(sample.recorded);
  EXPECT_EQ(recorder->count(), 0);

  sample.started_at = std::chrono::steady_clock::now();
  sample.excluded_wait = std::chrono::hours(1);
  record_scheduling_metrics(sample);
  EXPECT_FALSE(sample.recorded);
  EXPECT_EQ(recorder->count(), 0);

  sample.excluded_wait = {};
  sample.workloads = {{}, {4, 0, 1, 0}};
  record_scheduling_metrics(sample, /*empty=*/true);
  EXPECT_FALSE(sample.recorded);
  EXPECT_FALSE(MULTI_HISTOGRAM_scheduling_latency_microseconds.has_stats(
      {"742", "empty"}));
  EXPECT_EQ(recorder->count(), 0);

  sample.workloads = {{4, 0, 1, 0}};
  record_scheduling_metrics(sample);
  EXPECT_TRUE(sample.recorded);
  EXPECT_EQ(recorder->count(), 1);
}

TEST(SchedulingMetricsTest, KeepsScrapeReadersAliveAndExportsNativeMetrics) {
  auto state = std::make_shared<SchedulingMetricState>();
  state->server = "740";
  SchedulingSample sample;
  sample.started_at -= std::chrono::milliseconds(2);
  sample.metrics = state;
  sample.workloads = {{4, 0, 1, 0}, {0, 2, 0, 1}};
  SchedulingSample late = sample;
  record_scheduling_metrics(sample);
  record_scheduling_metrics(sample);
  auto& latency = MULTI_HISTOGRAM_scheduling_latency_microseconds;
  auto* recorder = latency.get_stats({"740", "submitted"});
  std::promise<void> entered;
  auto entered_future = entered.get_future();
  std::promise<void> resume;
  PausedMetricDumper dumper(entered, resume.get_future().share());
  auto dump = std::async(std::launch::async, [&] {
    return bvar::MVariable::dump_exposed(&dumper, nullptr);
  });
  ScopeGuard unblock([&] { resume.set_value(); });
  ASSERT_EQ(entered_future.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  close_scheduling_metrics(*state);
  resume.set_value();
  unblock.dismiss();
  EXPECT_GT(dump.get(), 0);
  record_scheduling_metrics(late);
  EXPECT_EQ(latency.get_stats({"740", "submitted"}), recorder);
  EXPECT_EQ(recorder->count(), 1);

  brpc::Server server;
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  ScopeGuard stop_server([&] {
    server.Stop(0);
    server.Join();
  });
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = 5000;
  brpc::Channel channel;
  ASSERT_EQ(channel.Init(server.listen_address(), &options), 0);
  brpc::Controller controller;
  controller.http_request().uri() = "/brpc_metrics";
  channel.CallMethod(nullptr, &controller, nullptr, nullptr, nullptr);
  ASSERT_FALSE(controller.Failed()) << controller.ErrorText();
  ASSERT_EQ(controller.http_response().status_code(), 200);
  const std::string body = controller.response_attachment().to_string();
  ::testing::Test::RecordProperty("brpc_metrics", body);
  for (const std::string& dp : {std::string("0"), std::string("1")}) {
    for (const char* kind : {"prompt_tokens",
                             "decode_tokens",
                             "prefill_requests",
                             "decode_requests"}) {
      const std::string family =
          std::string(kind).find("tokens") != std::string::npos ? "tokens"
                                                                : "requests";
      const std::string name = "scheduling_workload_" + family +
                               "_count_server_740_dp_" + dp + "_workload_" +
                               kind;
      // Pinned BRPC declares count as gauge; the value is cumulative rounds.
      EXPECT_NE(body.find("# TYPE " + name + " gauge\n"), std::string::npos);
      EXPECT_NE(body.find("\n" + name + " 1\n"), std::string::npos);
      const std::string labels = "_server_740_dp_" + dp + "_workload_" + kind;
      const std::string prefix = "scheduling_workload_" + family;
      for (const char* suffix : {"_latency", "_qps"}) {
        EXPECT_NE(body.find("\n" + prefix + suffix + labels + " "),
                  std::string::npos);
      }
      EXPECT_NE(
          body.find("\n" + prefix + "_latency" + labels + "_quantile_99 "),
          std::string::npos);
    }
  }
  EXPECT_NE(body.find("# TYPE "
                      "scheduling_latency_microseconds_count_server_740_round_"
                      "submitted gauge\n"),
            std::string::npos);
  EXPECT_NE(body.find("# TYPE "
                      "scheduling_latency_microseconds_latency_server_740_"
                      "round_submitted_quantile_99 gauge\n"),
            std::string::npos);
  const std::string latest =
      "scheduling_latency_microseconds_latest_server_740_round_submitted";
  EXPECT_NE(body.find("# TYPE " + latest + " gauge\n"), std::string::npos);
  const size_t value_at = body.find("\n" + latest + " ");
  ASSERT_NE(value_at, std::string::npos);
  EXPECT_GE(std::stod(body.substr(value_at + latest.size() + 2)), 2000.0);
}

TEST(PrefixCacheHitMetricsTest, UpdatesCumulativeTokenHitRate) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/100, /*hit_tokens=*/40);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/50, /*hit_tokens=*/30);

  const double expected_prompt = prompt_before + 150.0;
  const double expected_hit = hit_before + 70.0;
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

TEST(PrefixCacheHitMetricsTest, ClampsHitsAndIgnoresZeroPrompt) {
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);

  record_prefix_cache_hit_metrics(/*prompt_tokens=*/10, /*hit_tokens=*/20);
  const double rate_after_clamp = GAUGE_VALUE(prefix_cache_token_hit_rate_perc);
  record_prefix_cache_hit_metrics(/*prompt_tokens=*/0, /*hit_tokens=*/20);

  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   prompt_before + 10.0);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total),
                   hit_before + 10.0);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   rate_after_clamp);
}

TEST(PrefixCacheHitMetricsTest, KeepsCountersAndGaugeConsistentConcurrently) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kUpdatesPerThread = 100;
  const double prompt_before = COUNTER_VALUE(prefix_cache_prompt_tokens_total);
  const double hit_before = COUNTER_VALUE(prefix_cache_hit_tokens_total);
  std::vector<std::thread> threads;
  threads.reserve(kThreadCount);
  for (size_t thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([]() {
      for (size_t update = 0; update < kUpdatesPerThread; ++update) {
        record_prefix_cache_hit_metrics(/*prompt_tokens=*/4, /*hit_tokens=*/1);
      }
    });
  }
  for (std::thread& thread : threads) {
    thread.join();
  }

  const double expected_prompt =
      prompt_before + static_cast<double>(kThreadCount * kUpdatesPerThread * 4);
  const double expected_hit =
      hit_before + static_cast<double>(kThreadCount * kUpdatesPerThread);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_prompt_tokens_total),
                   expected_prompt);
  EXPECT_DOUBLE_EQ(COUNTER_VALUE(prefix_cache_hit_tokens_total), expected_hit);
  EXPECT_DOUBLE_EQ(GAUGE_VALUE(prefix_cache_token_hit_rate_perc),
                   expected_hit * 100.0 / expected_prompt);
}

}  // namespace
}  // namespace xllm
