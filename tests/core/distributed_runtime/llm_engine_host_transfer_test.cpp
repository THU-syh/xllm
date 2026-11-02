/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/metrics.h"
#include "core/distributed_runtime/comm_channel.h"
#include "core/distributed_runtime/llm_engine.h"
#include "core/distributed_runtime/rec_engine.h"
#include "core/distributed_runtime/vlm_engine.h"
#include "framework/batch/batch.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/stopping_checker.h"

namespace xllm {
class EngineKVCacheMetricsTestPeer final {
 public:
  static std::shared_ptr<DistManager> with_clients(
      std::vector<std::shared_ptr<WorkerClient>> clients) {
    auto manager = std::shared_ptr<DistManager>(new DistManager());
    manager->worker_clients_ = std::move(clients);
    return manager;
  }

  static void configure_llm_rec(
      RecEngine& engine,
      std::vector<std::shared_ptr<WorkerClient>> clients,
      int32_t dp_size) {
    engine.worker_clients_ = std::move(clients);
    engine.worker_clients_num_ = engine.worker_clients_.size();
    engine.dp_size_ = dp_size;
    engine.dp_local_tp_size_ = engine.worker_clients_num_ / dp_size;
    engine.pipeline_ =
        std::make_unique<RecEngine::LlmRecEnginePipeline>(engine);
  }

  template <typename Workers>
  static void register_local_layout(LLMEngine& engine,
                                    const Workers& workers,
                                    int32_t dp_size) {
    BlockManagerPool::Options options;
    options.num_blocks(8)
        .block_size(1)
        .enable_prefix_cache(false)
        .model_id(engine.options_.model_id())
        .server_idx(engine.options_.server_idx());
    engine.kv_cache_manager_ =
        std::make_unique<BlockManagerPool>(options, dp_size);
    engine.register_kv_cache_pool_layout(workers, dp_size);
  }

  template <typename EngineType>
  static bool allocate(EngineType& engine,
                       ModelArgs args,
                       const KVCacheCapacity& capacity) {
    engine.args_ = std::move(args);
    return engine.allocate_kv_cache(capacity);
  }
};

namespace {

class RecordingWorker final : public WorkerClient {
 public:
  void prefetch_from_storage(
      const std::shared_ptr<const StoragePrefetchRequest>& request,
      std::shared_ptr<PrefetchResult> result,
      size_t worker_index) override {
    prefetch_request_ = request;
    prefetch_result_ = std::move(result);
    prefetch_index_ = worker_index;
  }

  folly::SemiFuture<uint32_t> transfer_kv_blocks(
      const std::vector<BlockTransferInfo>& infos) override {
    ++offload_count_;
    offload_infos_ = infos;
    return folly::makeSemiFuture(static_cast<uint32_t>(infos.size()));
  }

  void transfer_kv_blocks(
      uint64_t batch_id,
      const std::vector<BlockTransferInfo>& infos) override {
    ++load_count_;
    load_batch_id_ = batch_id;
    load_infos_ = infos;
  }

  uint32_t offload_count_ = 0;
  uint32_t load_count_ = 0;
  uint64_t load_batch_id_ = 0;
  std::vector<BlockTransferInfo> offload_infos_;
  std::vector<BlockTransferInfo> load_infos_;
  std::shared_ptr<const StoragePrefetchRequest> prefetch_request_;
  std::shared_ptr<PrefetchResult> prefetch_result_;
  size_t prefetch_index_ = 0;
};

class ClientEngine final : public LLMEngine {
 public:
  ClientEngine(runtime::Options options,
               std::vector<std::shared_ptr<WorkerClient>> clients)
      : LLMEngine(std::move(options), std::move(clients)) {}
};

class LayoutWorker final : public WorkerClient {
 public:
  explicit LayoutWorker(std::vector<KVCachePoolLayout> layout)
      : layout_(std::move(layout)) {}

  folly::SemiFuture<bool> allocate_kv_cache_async(
      const KVCacheShape& /*shape*/) override {
    ++allocations_;
    return folly::makeSemiFuture(true);
  }

  folly::SemiFuture<folly::Unit> process_group_test_async() override {
    return folly::makeSemiFuture(folly::unit);
  }

  std::vector<KVCachePoolLayout> kv_cache_pool_layout() const override {
    CHECK_GT(allocations_, 0);
    return layout_;
  }

  int32_t allocations_ = 0;

 private:
  std::vector<KVCachePoolLayout> layout_;
};

class EngineKVCacheMetricsTest : public ::testing::Test {
 protected:
  runtime::Options options(const std::string& model, int32_t dp_size = 1) {
    runtime::Options value;
    value.model_id(model)
        .server_idx(623)
        .devices({torch::Device(torch::kPrivateUse1, 0)})
        .master_node_addr("127.0.0.1:1")
        .dp_size(dp_size)
        .cp_size(1)
        .enable_prefix_cache(false)
        .max_seqs_per_batch(4);
    return value;
  }

  ModelArgs model_args(bool hybrid) {
    ModelArgs args;
    args.model_type("qwen3")
        .n_layers(hybrid ? 2 : 1)
        .n_heads(2)
        .head_dim(8)
        .layer_types(hybrid ? std::vector<std::string>{"linear_attention",
                                                       "full_attention"}
                            : std::vector<std::string>{"full_attention"})
        .linear_num_key_heads(2)
        .linear_num_value_heads(2)
        .linear_key_head_dim(2)
        .linear_value_head_dim(2)
        .linear_conv_kernel_dim(4);
    return args;
  }

  KVCacheCapacity capacity(bool hybrid) {
    KVCacheCapacity value;
    value.n_blocks(8).block_size(1).num_linear_state_blocks(hybrid ? 4 : 0);
    return value;
  }

  std::shared_ptr<LayoutWorker> worker(bool draft = false, bool hybrid = true) {
    std::vector<KVCachePoolLayout> layout = {{BlockType::KV, 8, 64, "target"}};
    if (hybrid) {
      layout.emplace_back(
          KVCachePoolLayout{BlockType::LINEAR, 4, 32, "target"});
    }
    if (draft) {
      layout.emplace_back(KVCachePoolLayout{BlockType::KV, 8, 16, "draft"});
    }
    return std::make_shared<LayoutWorker>(std::move(layout));
  }

  Sequence sequence(int32_t dp_rank) {
    sampling_.beam_width = 0;
    SequenceParams params;
    params.seq_capacity = 16;
    params.sampling_param = &sampling_;
    params.stopping_checker = &stopping_;
    const std::vector<int32_t> tokens = {1, 2};
    IncrementalDecoder decoder(/*prompt=*/"prompt",
                               /*num_prompt_tokens=*/tokens.size(),
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    Sequence value(
        /*index=*/0, tokens, torch::Tensor(), MMData(), decoder, params);
    value.set_dp_rank(dp_rank);
    return value;
  }

  double occupancy(const std::string& model,
                   const char* role,
                   int32_t dp_rank,
                   int32_t worker_rank,
                   const char* type) {
    return MULTI_GAUGE_kv_cache_pool_occupied_bytes
        .get_stats({model,
                    "623",
                    role,
                    std::to_string(dp_rank),
                    std::to_string(worker_rank),
                    type})
        ->get_value();
  }

  void check_allocation(Engine& engine,
                        const std::string& model,
                        bool hybrid = true) {
    Sequence active = sequence(/*dp_rank=*/1);
    auto* pool = engine.block_manager_pool();
    ASSERT_TRUE(pool->try_allocate(&active));
    pool->report_pool_occupancy();
    EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 1, "kv"), 128.0);
    if (hybrid) {
      EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 1, "linear"), 32.0);
    }
    EXPECT_DOUBLE_EQ(occupancy(model, "target", 0, 0, "kv"), 0.0);
    pool->deallocate(&active);
    pool->report_pool_occupancy();
    EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 1, "kv"), 0.0);
    if (hybrid) {
      EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 1, "linear"), 0.0);
    }
  }

  RequestSamplingParam sampling_;
  StoppingChecker stopping_;
};

TEST_F(EngineKVCacheMetricsTest, WarnsWhenWorkerLayoutIsUnavailable) {
  const std::string model = "kv_missing_layout_review";
  ClientEngine engine(
      options(model, /*dp_size=*/2),
      {std::make_shared<LayoutWorker>(std::vector<KVCachePoolLayout>{}),
       worker()});
  testing::internal::CaptureStderr();
  const bool allocated = EngineKVCacheMetricsTestPeer::allocate(
      engine, model_args(/*hybrid=*/true), capacity(/*hybrid=*/true));
  const std::string log = testing::internal::GetCapturedStderr();
  ASSERT_TRUE(allocated);
  EXPECT_NE(log.find("KV occupancy layout unavailable: model=" + model),
            std::string::npos);
  EXPECT_NE(log.find("worker_rank=0, dp_rank=0"), std::string::npos);
  Sequence active = sequence(/*dp_rank=*/1);
  auto* pool = engine.block_manager_pool();
  ASSERT_TRUE(pool->try_allocate(&active));
  pool->report_pool_occupancy();
  EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 1, "kv"), 128.0);
  pool->deallocate(&active);
}

class LayoutService final : public proto::DistributeWorker {
 public:
  void AllocateKVCache(google::protobuf::RpcController* /*controller*/,
                       const proto::AllocateKVCacheRequest* request,
                       proto::AllocateKVCacheResponse* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard guard(done);
    response->set_ok(request->kv_cache_shape().key_cache_shape(0) > 0);
    for (int32_t type : {0, 8}) {
      auto* pool = response->add_pool_layout();
      pool->set_block_type(static_cast<proto::BlockType>(type));
      pool->set_resource_count(4);
      pool->set_bytes_per_resource(64);
      pool->set_cache_role("target");
    }
  }

  void AllocateKVCacheWithTransfer(google::protobuf::RpcController* controller,
                                   const proto::AllocateKVCacheRequest* request,
                                   proto::AllocateKVCacheResponse* response,
                                   google::protobuf::Closure* done) override {
    AllocateKVCache(controller, request, response, done);
  }
};

TEST(CommChannelKVCacheMetricsTest, ReadsKnownPoolsAndClearsFailedOutput) {
  LayoutService service;
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  CommChannel channel;
  ASSERT_TRUE(channel.init_brpc("127.0.0.1:" +
                                std::to_string(server.listen_address().port)));
  proto::KVCacheShape proto_shape;
  proto_shape.add_key_cache_shape(4);
  proto_shape.add_key_cache_shape(2);
  const KVCacheShape shape = KVCacheShape::from_proto(proto_shape);
  proto_shape.set_key_cache_shape(0, 0);
  const KVCacheShape failed_shape = KVCacheShape::from_proto(proto_shape);
  for (bool transfer : {false, true}) {
    auto allocate = [&](const KVCacheShape& value,
                        std::vector<KVCachePoolLayout>& layout) {
      return transfer ? channel.allocate_kv_cache_with_transfer(value, layout)
                      : channel.allocate_kv_cache(value, layout);
    };
    std::vector<KVCachePoolLayout> layout;
    ASSERT_TRUE(allocate(shape, layout));
    ASSERT_EQ(layout.size(), 1u);
    EXPECT_EQ(layout[0].block_type, BlockType::KV);
    EXPECT_EQ(layout[0].resource_count, 4);
    EXPECT_EQ(layout[0].bytes_per_resource, 64);
    EXPECT_EQ(layout[0].cache_role, "target");
    const auto saved = layout;
    EXPECT_FALSE(allocate(KVCacheShape(), layout));
    EXPECT_TRUE(layout.empty());
    layout = saved;
    EXPECT_FALSE(allocate(failed_shape, layout));
    EXPECT_TRUE(layout.empty());
  }
}

TEST(CommChannelKVCacheMetricsTest, PreservesStatusWireCompatibility) {
  proto::Status old_response;
  old_response.set_ok(true);
  proto::AllocateKVCacheResponse new_response;
  ASSERT_TRUE(new_response.ParseFromString(old_response.SerializeAsString()));
  EXPECT_TRUE(new_response.ok());
  EXPECT_EQ(new_response.pool_layout_size(), 0);
  new_response.add_pool_layout()->set_resource_count(4);
  old_response.Clear();
  ASSERT_TRUE(old_response.ParseFromString(new_response.SerializeAsString()));
  EXPECT_TRUE(old_response.ok());
}

TEST_F(EngineKVCacheMetricsTest, LocalWorkersMapToDpPools) {
  const std::string model = "kv_local_workers_review";
  ClientEngine engine(options(model, /*dp_size=*/2),
                      {worker(), worker(), worker(), worker()});
  std::vector<std::unique_ptr<LayoutWorker>> local_workers;
  local_workers.reserve(4);
  for (int32_t rank = 0; rank < 4; ++rank) {
    local_workers.emplace_back(std::make_unique<LayoutWorker>(
        std::vector<KVCachePoolLayout>{{BlockType::KV, 8, 64, "target"}}));
    ASSERT_TRUE(
        local_workers.back()->allocate_kv_cache_async(KVCacheShape()).get());
  }
  EngineKVCacheMetricsTestPeer::register_local_layout(
      engine, local_workers, /*dp_size=*/2);
  Sequence active = sequence(/*dp_rank=*/1);
  auto* pool = engine.block_manager_pool();
  ASSERT_TRUE(pool->try_allocate(&active));
  pool->report_pool_occupancy();
  for (int32_t rank = 0; rank < 4; ++rank) {
    EXPECT_DOUBLE_EQ(occupancy(model, "target", rank / 2, rank, "kv"),
                     rank < 2 ? 0.0 : 128.0);
  }
  pool->deallocate(&active);
  pool->report_pool_occupancy();
  EXPECT_DOUBLE_EQ(occupancy(model, "target", 1, 3, "kv"), 0.0);
}

TEST_F(EngineKVCacheMetricsTest, SharedLayoutBelongsOnlyToTargetPool) {
  const std::string model = "kv_spec_review";
  auto client = worker(/*draft=*/true);
  ClientEngine target(options(model), {client});
  ASSERT_TRUE(EngineKVCacheMetricsTestPeer::allocate(
      target, model_args(/*hybrid=*/true), capacity(/*hybrid=*/true)));
  Sequence active = sequence(/*dp_rank=*/0);
  auto* pool = target.block_manager_pool();
  ASSERT_TRUE(pool->try_allocate(&active));
  pool->report_pool_occupancy();
  EXPECT_DOUBLE_EQ(occupancy(model, "target", 0, 0, "linear"), 32.0);
  EXPECT_DOUBLE_EQ(occupancy(model, "draft", 0, 0, "kv"), 32.0);
  {
    auto draft_options = options(model);
    draft_options.is_draft_engine(true);
    ClientEngine draft(std::move(draft_options), {client});
    ASSERT_TRUE(EngineKVCacheMetricsTestPeer::allocate(
        draft, model_args(/*hybrid=*/false), capacity(/*hybrid=*/false)));
    EXPECT_EQ(client->allocations_, 2);
    EXPECT_DOUBLE_EQ(occupancy(model, "target", 0, 0, "linear"), 32.0);
    EXPECT_DOUBLE_EQ(occupancy(model, "draft", 0, 0, "kv"), 32.0);
  }
  EXPECT_DOUBLE_EQ(occupancy(model, "draft", 0, 0, "kv"), 32.0);
  pool->deallocate(&active);
  pool->report_pool_occupancy();
  EXPECT_DOUBLE_EQ(occupancy(model, "target", 0, 0, "linear"), 0.0);
  EXPECT_DOUBLE_EQ(occupancy(model, "draft", 0, 0, "kv"), 0.0);
}

TEST_F(EngineKVCacheMetricsTest, VlmRegistersLayoutOnAllocation) {
  const std::string model = "kv_vlm_review";
  VLMEngine engine(
      options(model, /*dp_size=*/2),
      EngineKVCacheMetricsTestPeer::with_clients({worker(), worker()}));
  ASSERT_TRUE(EngineKVCacheMetricsTestPeer::allocate(
      engine, model_args(/*hybrid=*/true), capacity(/*hybrid=*/true)));
  check_allocation(engine, model);
}

TEST_F(EngineKVCacheMetricsTest, LlmRecRegistersLayoutOnAllocation) {
  const std::string model = "kv_rec_review";
  RecEngine engine(options(model, /*dp_size=*/2));
  EngineKVCacheMetricsTestPeer::configure_llm_rec(
      engine,
      {worker(/*draft=*/false, /*hybrid=*/false),
       worker(/*draft=*/false, /*hybrid=*/false)},
      /*dp_size=*/2);
  ASSERT_TRUE(EngineKVCacheMetricsTestPeer::allocate(
      engine, model_args(/*hybrid=*/false), capacity(/*hybrid=*/false)));
  EXPECT_FALSE(engine.block_manager_pool()->options().enable_linear_state());
  check_allocation(engine, model, /*hybrid=*/false);
}

struct TransferTopology {
  uint32_t workers_;
  uint32_t dp_;
  uint32_t cp_;
  uint32_t rank_;
  std::vector<uint32_t> expected_;
};

class EngineHostTransferTest
    : public ::testing::TestWithParam<TransferTopology> {};

TEST_P(EngineHostTransferTest, TransfersEveryShardInOnlyTheRequestedDpGroup) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(topology.workers_);
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_));
  ClientEngine engine(std::move(options), std::move(clients));
  EXPECT_EQ(engine.host_transfer_worker_count(), topology.expected_.size());
  BlockTransferInfo info(/*src_block_id=*/3, /*dst_block_id=*/7);
  info.transfer_type = TransferType::D2H2G;
  info.block_type = BlockType::C4;
  std::fill(std::begin(info.hash_key), std::end(info.hash_key), 42);
  auto futures = engine.transfer_kv_blocks(topology.rank_, {info});
  EXPECT_EQ(futures.size(), topology.expected_.size());
  for (auto& future : futures) {
    EXPECT_EQ(std::move(future).get(), 1U);
  }
  info.transfer_type = TransferType::H2D;
  engine.transfer_kv_blocks(topology.rank_, /*batch_id=*/123, {info});
  std::vector<uint32_t> offloaded;
  std::vector<uint32_t> loaded;
  offloaded.reserve(topology.workers_);
  loaded.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    const auto& worker = *workers[i];
    EXPECT_LE(worker.offload_count_, 1U);
    EXPECT_LE(worker.load_count_, 1U);
    if (worker.offload_count_ != 0) {
      offloaded.emplace_back(i);
      ASSERT_EQ(worker.offload_infos_.size(), 1U);
      EXPECT_EQ(worker.offload_infos_[0].src_block_id, 3);
      EXPECT_EQ(worker.offload_infos_[0].dst_block_id, 7);
      EXPECT_EQ(worker.offload_infos_[0].transfer_type, TransferType::D2H2G);
    }
    if (worker.load_count_ == 0) {
      continue;
    }
    loaded.emplace_back(i);
    EXPECT_EQ(worker.load_batch_id_, 123U);
    ASSERT_EQ(worker.load_infos_.size(), 1U);
    EXPECT_EQ(worker.load_infos_[0].src_block_id, 3);
    EXPECT_EQ(worker.load_infos_[0].dst_block_id, 7);
    EXPECT_EQ(worker.load_infos_[0].block_type, BlockType::C4);
    EXPECT_EQ(worker.load_infos_[0].transfer_type, TransferType::H2D);
    EXPECT_TRUE(std::equal(std::begin(info.hash_key),
                           std::end(info.hash_key),
                           std::begin(worker.load_infos_[0].hash_key)));
  }
  EXPECT_EQ(offloaded, topology.expected_);
  EXPECT_EQ(loaded, topology.expected_);
}

TEST_P(EngineHostTransferTest, PrefetchWaitsForEveryShardAndStopsAtFirstMiss) {
  const auto& topology = GetParam();
  std::vector<std::shared_ptr<WorkerClient>> clients;
  std::vector<std::shared_ptr<RecordingWorker>> workers;
  clients.reserve(topology.workers_);
  workers.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    auto worker = std::make_shared<RecordingWorker>();
    clients.emplace_back(worker);
    workers.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(static_cast<int32_t>(topology.workers_))
      .dp_size(static_cast<int32_t>(topology.dp_))
      .cp_size(static_cast<int32_t>(topology.cp_))
      .prefetch_timeout(0)
      .prefetch_batch_size(5);
  ClientEngine engine(std::move(options), std::move(clients));
  auto request = std::make_shared<StoragePrefetchRequest>();
  for (int32_t i = 0; i < 5; ++i) {
    BlockTransferInfo info(/*src_block_id=*/i, /*dst_block_id=*/i);
    info.transfer_type = TransferType::G2H;
    PrefetchUnit unit;
    unit.gated_blocks.emplace_back(info);
    request->units.emplace_back(std::move(unit));
  }
  size_t callbacks = 0;
  size_t common_prefix = 0;
  engine.prefetch_from_storage(
      topology.rank_,
      request,
      [] { return false; },
      [&](PrefetchSummary summary) {
        ++callbacks;
        common_prefix = 0;
        while (common_prefix < summary.gated_hits.size() &&
               summary.gated_hits[common_prefix] == summary.rank_count) {
          ++common_prefix;
        }
      });
  std::vector<uint32_t> prefetched;
  prefetched.reserve(topology.workers_);
  for (uint32_t i = 0; i < topology.workers_; ++i) {
    if (workers[i]->prefetch_request_ != nullptr) {
      prefetched.emplace_back(i);
    }
  }
  ASSERT_EQ(prefetched, topology.expected_);
  for (size_t i = 0; i < topology.expected_.size(); ++i) {
    const auto& worker = *workers[topology.expected_[i]];
    EXPECT_EQ(worker.prefetch_request_, request);
    EXPECT_EQ(worker.prefetch_index_, i);
    EXPECT_EQ(worker.prefetch_result_->worker_count(),
              topology.expected_.size());
    EXPECT_EQ(callbacks, 0U);
    const bool last = i + 1 == topology.expected_.size();
    const std::vector<uint8_t> gated_hits =
        last ? std::vector<uint8_t>{1, 1, 1, 0, 1}
             : std::vector<uint8_t>{1, 1, 1, 1, 1};
    const std::vector<uint8_t> non_gated_hits(5, 1);
    EXPECT_EQ(worker.prefetch_result_->record_batch_result(
                  i, gated_hits, non_gated_hits),
              PrefetchControl::STOP);
    EXPECT_EQ(callbacks, 0U);
    worker.prefetch_result_->mark_worker_ended(i, /*worker_ok=*/true);
  }
  EXPECT_EQ(callbacks, 1U);
  EXPECT_EQ(common_prefix, 3U);
}

INSTANTIATE_TEST_SUITE_P(
    DpCp,
    EngineHostTransferTest,
    ::testing::Values(
        TransferTopology{8, 1, 1, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{8, 1, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 1, 1, {8, 9, 10, 11, 12, 13, 14, 15}},
        TransferTopology{16, 2, 4, 0, {0, 1, 2, 3, 4, 5, 6, 7}},
        TransferTopology{16, 2, 4, 1, {8, 9, 10, 11, 12, 13, 14, 15}}));

// Overlap batches borrow the sequence cache. Serial worker execution does not
// require additional Block owners in the batch.

class StepCaptureWorker final : public WorkerClient {
 public:
  folly::SemiFuture<std::optional<RawForwardOutput>> step_remote_async(
      const ForwardInput& inputs) override {
    ++step_count_;
    captured_token_ids_ = inputs.token_ids.clone();
    RawForwardOutput output;
    RawSampleOutput sample;
    RawToken token;
    token.id = 5;
    sample.tokens.emplace_back(std::move(token));
    output.outputs.emplace_back(std::move(sample));
    return folly::makeSemiFuture(
        std::optional<RawForwardOutput>(std::move(output)));
  }

  uint32_t step_count() const { return step_count_; }
  const torch::Tensor& captured_token_ids() const {
    return captured_token_ids_;
  }

 private:
  uint32_t step_count_ = 0;
  torch::Tensor captured_token_ids_;
};

std::shared_ptr<Request> make_overlap_step_request() {
  const std::vector<int32_t> prompt{1, 2, 3, 4};
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(32);
  stopping_checker.set_max_context_len(30000);
  stopping_checker.set_ignore_eos(true);
  RequestState req_state("x",
                         prompt,
                         sampling_param,
                         scheduler_param,
                         stopping_checker,
                         prompt.size() + 30000,
                         /*n=*/1,
                         /*best_of=*/1,
                         /*logprobs=*/false,
                         /*stream=*/false,
                         /*echo=*/false,
                         /*skip_special_tokens=*/false,
                         /*enable_schedule_overlap=*/false,
                         /*output_func=*/nullptr,
                         /*outputs_func=*/nullptr);
  return std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
}

TEST(LLMEngineOverlapOwnershipTest, MtpOverlapStepDoesNotCopyCpPrefillBlocks) {
  // Two workers form one DP group with cp_size=2: the engine must dispatch
  // the same forward_inputs[dp_rank] to both CP ranks (llm_engine.cpp: "Engine
  // sends full global tokens; model-side CP shards inside the worker").
  std::vector<std::shared_ptr<StepCaptureWorker>> workers;
  std::vector<std::shared_ptr<WorkerClient>> clients;
  workers.reserve(2);
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    auto worker = std::make_shared<StepCaptureWorker>();
    workers.emplace_back(worker);
    clients.emplace_back(std::move(worker));
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  options.speculative_algorithm("mtp");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  // A prefill batch processes every prompt token: no KV tokens yet.
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  engine.step(batches);

  // Both CP ranks ran the same batch with byte-identical token inputs.
  for (const auto& worker : workers) {
    ASSERT_EQ(worker->step_count(), 1u);
  }
  EXPECT_TRUE(torch::equal(workers[0]->captured_token_ids(),
                           workers[1]->captured_token_ids()));

  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  batches.clear();
  EXPECT_EQ(kv_blocks.front().ref_count(), 1u);
}

TEST(LLMEngineOverlapOwnershipTest, DFlash2StepDoesNotCopyCpBlocks) {
  std::vector<std::shared_ptr<WorkerClient>> clients;
  clients.reserve(2);
  for (int32_t worker_rank = 0; worker_rank < 2; ++worker_rank) {
    clients.emplace_back(std::make_shared<StepCaptureWorker>());
  }
  runtime::Options options;
  options.world_size(2).dp_size(1).cp_size(2).enable_schedule_overlap(true);
  options.speculative_algorithm("DFlash2");
  ClientEngine engine(std::move(options), std::move(clients));

  BlockManagerPool::Options pool_options;
  pool_options.num_blocks_ = 32;
  pool_options.block_size_ = 4;
  pool_options.max_seqs_per_batch_ = 1024;
  BlockManagerPool pool(pool_options, /*dp_size=*/1);
  auto request = make_overlap_step_request();
  Sequence* sequence = request->sequences().front().get();
  ASSERT_TRUE(pool.try_allocate(sequence));
  ASSERT_GT(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  sequence->kv_state().set_kv_cache_tokens_num(0);

  std::vector<Batch> batches;
  batches.emplace_back(sequence);
  const auto kv_blocks = sequence->kv_state().blocks(BlockType::KV);
  ASSERT_FALSE(kv_blocks.empty());
  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  engine.step(batches);

  EXPECT_EQ(kv_blocks.front().ref_count(), 2u);
  // Dropping the batch preserves the sequence's owner.
  batches.clear();
  EXPECT_EQ(sequence->kv_state().blocks(BlockType::KV).front().ref_count(), 1u);
}

}  // namespace
}  // namespace xllm
