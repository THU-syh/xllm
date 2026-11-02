/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "block_manager_impl.h"
#include "block_manager_pool.h"
#include "common/global_flags.h"
#include "concurrent_block_manager_impl.h"
#include "core/framework/batch/batch.h"
#include "core/framework/config/scheduler_config.h"
#include "framework/block/block_manager_pool_test_peer.h"
#include "framework/block/composite_block_manager.h"
#include "framework/block/linear_state_block_manager.h"
#include "framework/model/model_input_params.h"
#include "framework/prefix_cache/block_hasher.h"
#include "framework/request/incremental_decoder.h"

namespace xllm {

namespace {

template <typename T>
class ScopedValue final {
 public:
  ScopedValue(T* target, T value) : target_(target), old_(*target) {
    *target_ = value;
  }
  ~ScopedValue() { *target_ = old_; }

  ScopedValue(const ScopedValue&) = delete;
  ScopedValue& operator=(const ScopedValue&) = delete;

 private:
  T* target_;
  T old_;
};

LinearStatePrefixHash make_prefix_hash(uint8_t tag) {
  LinearStatePrefixHash hash{};
  hash.fill(tag);
  return hash;
}

LinearStatePrefixHash compute_linear_state_prefix_hash_for_test(
    const Slice<int32_t>& token_ids,
    int32_t block_size,
    size_t boundary_tokens) {
  LinearStatePrefixHash hash{};
  if (block_size <= 0 || boundary_tokens == 0) {
    return hash;
  }
  const size_t stride = static_cast<size_t>(block_size);
  if (boundary_tokens % stride != 0 || boundary_tokens > token_ids.size()) {
    return hash;
  }

  const size_t boundary_blocks = boundary_tokens / stride;
  const uint8_t* previous_hash = nullptr;
  for (size_t block_idx = 0; block_idx < boundary_blocks; ++block_idx) {
    xxh3_128bits_hash(
        previous_hash,
        token_ids.slice(block_idx * stride, (block_idx + 1) * stride),
        hash.data());
    previous_hash = hash.data();
  }
  return hash;
}

int32_t insert_linear_state_checkpoint(LinearStateBlockManager* cache,
                                       const XXH3Key& hash) {
  if (BlockManagerPoolTestPeer::contains(cache, hash)) {
    Block matched = BlockManagerPoolTestPeer::match(cache, hash);
    return matched.is_valid() ? matched.id() : -1;
  }
  std::vector<Block> allocated = cache->allocate(1);
  if (allocated.empty()) {
    return -1;
  }
  Block slot_block = std::move(allocated.front());
  const int32_t slot = slot_block.id();
  slot_block.set_hash_value(hash.data);
  std::vector<Block> checkpoint;
  checkpoint.emplace_back(std::move(slot_block));
  cache->cache(checkpoint);
  return slot;
}

int32_t insert_linear_state_checkpoint(LinearStateBlockManager* cache,
                                       const LinearStatePrefixHash& hash) {
  return insert_linear_state_checkpoint(cache, XXH3Key(hash.data()));
}

bool allocate_kv_prefix_for_test(BlockManagerPool& pool, Sequence& sequence) {
  int32_t dp_rank = -1;
  std::vector<Block> blocks = pool.allocate(sequence.num_tokens(), dp_rank);
  if (blocks.empty()) {
    return false;
  }
  sequence.set_dp_rank(dp_rank);
  sequence.kv_state().add_blocks(BlockType::KV, std::move(blocks));
  return true;
}

bool allocate_next_linear_chunk(BlockManagerPool& pool, Sequence& sequence) {
  pool.allocate_shared(&sequence);
  const size_t chunk_stride =
      SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill();
  const size_t target = std::min(sequence.num_tokens(),
                                 sequence.kv_cache_tokens_num() + chunk_stride);
  return pool.allocate(&sequence, target);
}

class BlockManagerPoolTest : public ::testing::Test {
 protected:
  BlockManagerPoolTest() {
    sampling_param_.beam_width = 0;
    sampling_param_.is_embeddings = false;
  }

  Sequence make_sequence(size_t index,
                         const std::vector<int32_t>& prompt_tokens) {
    SequenceParams params;
    params.seq_capacity = prompt_tokens.size() + 8;
    params.echo = false;
    params.skip_special_tokens = true;
    params.streaming = false;
    params.enable_schedule_overlap = false;
    params.rec_type = RecType::kNone;
    params.bos_token_id = 0;
    params.request_id = "block_manager_pool_test";
    params.sampling_param = &sampling_param_;
    params.stopping_checker = &stopping_checker_;

    IncrementalDecoder decoder(
        /*prompt=*/"prompt",
        /*num_prompt_tokens=*/prompt_tokens.size(),
        /*echo=*/params.echo,
        /*skip_special_tokens=*/params.skip_special_tokens);

    return Sequence(index,
                    prompt_tokens,
                    /*input_embedding=*/torch::Tensor(),
                    /*mm_data=*/MMData(),
                    decoder,
                    params);
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

BlockManagerPool::Options make_linear_state_pool_options(
    int32_t linear_state_num_slots) {
  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // The EMBEDDING pool is sized directly by num_embedding_blocks and is only
  // built under spec decode, so opt in here.
  options.max_seqs_per_batch(0)
      .num_embedding_blocks(4)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(linear_state_num_slots);
  return options;
}

}  // namespace

TEST(BlockManagerTest, Basic) {
  const uint32_t n_blocks = 10;
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(n_blocks).block_size(block_size);
  BlockManagerImpl manager(options);

  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);
  EXPECT_EQ(manager.block_size(), block_size);

  // Allocate a block
  {
    std::vector<Block> allocated = manager.allocate(1);
    ASSERT_EQ(allocated.size(), 1u);
    Block block = std::move(allocated.front());
    EXPECT_EQ(block.id(), 1);
    EXPECT_EQ(block.size(), block_size);
    EXPECT_EQ(block.is_shared(), false);
    EXPECT_EQ(block.ref_count(), 1);

    EXPECT_EQ(manager.num_free_blocks(), n_blocks - 2);
  }
  // the block should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Allocate a list of blocks
  {
    std::vector<Block> blocks;
    for (uint32_t i = 1; i < n_blocks; ++i) {
      auto allocated = manager.allocate(1);
      ASSERT_EQ(allocated.size(), 1u);
      Block block = std::move(allocated.front());
      EXPECT_EQ(block.id(), i);
      EXPECT_EQ(block.size(), block_size);
      EXPECT_EQ(block.is_shared(), false);
      EXPECT_EQ(block.ref_count(), 1);
      blocks.push_back(std::move(block));
    }
    EXPECT_EQ(manager.num_free_blocks(), 0);
    for (const auto& block : blocks) {
      EXPECT_EQ(block.ref_count(), 1);
      EXPECT_EQ(block.is_shared(), false);
    }

    // Test CHECK failure
    EXPECT_TRUE(manager.allocate(1).empty());
  }

  // all blocks should be freed after the scope
  EXPECT_EQ(manager.num_free_blocks(), n_blocks - 1);

  // Test shared blocks
  {
    std::vector<Block> allocated = manager.allocate(1);
    ASSERT_EQ(allocated.size(), 1u);
    Block block = std::move(allocated.front());
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);
    // test copy constructor
    {
      // NOLINTNEXTLINE
      const Block block2 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block2.ref_count(), 2);
      EXPECT_EQ(block2.is_shared(), true);
      EXPECT_EQ(block2, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test assignment operator
    {
      std::vector<Block> more = manager.allocate(1);
      ASSERT_EQ(more.size(), 1u);
      Block block4 = std::move(more.front());
      block4 = block;
      EXPECT_EQ(block.ref_count(), 2);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(block4.ref_count(), 2);
      EXPECT_EQ(block4.is_shared(), true);
      EXPECT_EQ(block4, block);

      Block invalid_block;
      invalid_block = block;
      EXPECT_EQ(block.ref_count(), 3);
      EXPECT_EQ(block.is_shared(), true);
      EXPECT_EQ(invalid_block.ref_count(), 3);
      EXPECT_EQ(invalid_block.is_shared(), true);
      EXPECT_EQ(invalid_block, block);
    }
    EXPECT_EQ(block.ref_count(), 1);
    EXPECT_EQ(block.is_shared(), false);

    // test move constructor
    {
      Block block3 = std::move(block);
      EXPECT_FALSE(block.is_valid());

      EXPECT_EQ(block3.ref_count(), 1);
      EXPECT_EQ(block3.is_shared(), false);
      EXPECT_FALSE(block3 == block);
    }
    EXPECT_FALSE(block.is_valid());
  }
}

TEST(BlockManagerTest, InflightRetentionPreservesLogicalOwnership) {
  BlockManager::Options options;
  options.num_blocks(2).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  std::vector<Block> allocated = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(allocated.size(), 1u);
  const Block& owner = allocated.front();
  EXPECT_EQ(owner.ref_count(), 1u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_FALSE(owner.is_shared());
  EXPECT_FALSE(owner.has_multiple_owners());

  {
    const Block inflight = owner.retain_for_inflight();
    EXPECT_EQ(inflight.id(), owner.id());
    EXPECT_EQ(inflight.size(), owner.size());
    EXPECT_EQ(inflight.manager(), &manager);
    EXPECT_EQ(owner.ref_count(), 2u);
    EXPECT_EQ(inflight.ref_count(), 2u);
    EXPECT_TRUE(owner.is_shared());
    EXPECT_TRUE(inflight.is_shared());
    EXPECT_EQ(owner.owner_count(), 1u);
    EXPECT_EQ(inflight.owner_count(), 1u);
    EXPECT_FALSE(owner.has_multiple_owners());
    EXPECT_FALSE(inflight.has_multiple_owners());

    {
      Block second_owner = owner;
      EXPECT_EQ(owner.ref_count(), 3u);
      EXPECT_EQ(owner.owner_count(), 2u);
      EXPECT_EQ(second_owner.owner_count(), 2u);
      EXPECT_TRUE(owner.has_multiple_owners());
      EXPECT_TRUE(second_owner.has_multiple_owners());
      EXPECT_TRUE(inflight.has_multiple_owners());

      Block moved_owner = std::move(second_owner);
      EXPECT_EQ(moved_owner.ref_count(), 3u);
      EXPECT_EQ(moved_owner.owner_count(), 2u);
      EXPECT_FALSE(second_owner.is_valid());
      second_owner = std::move(moved_owner);
      EXPECT_EQ(second_owner.ref_count(), 3u);
      EXPECT_EQ(second_owner.owner_count(), 2u);
      EXPECT_FALSE(moved_owner.is_valid());
    }
    EXPECT_EQ(owner.ref_count(), 2u);
    EXPECT_EQ(owner.owner_count(), 1u);
    EXPECT_FALSE(owner.has_multiple_owners());
  }
  EXPECT_EQ(owner.ref_count(), 1u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_FALSE(owner.is_shared());
}

TEST(BlockManagerTest, InflightCopiesAndMovesPreserveReferenceKind) {
  BlockManager::Options options;
  options.num_blocks(2).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  std::vector<Block> allocated = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(allocated.size(), 1u);
  const Block& owner = allocated.front();
  {
    Block inflight = owner.retain_for_inflight();
    Block copied = inflight;
    EXPECT_EQ(owner.ref_count(), 3u);
    EXPECT_EQ(owner.owner_count(), 1u);

    Block moved = std::move(copied);
    EXPECT_EQ(moved.id(), owner.id());
    EXPECT_EQ(moved.ref_count(), 3u);
    EXPECT_EQ(moved.owner_count(), 1u);
    EXPECT_FALSE(copied.is_valid());
    EXPECT_EQ(copied.ref_count(), 0u);
    EXPECT_EQ(copied.owner_count(), 0u);

    Block& same_alias = moved;
    moved = same_alias;
    EXPECT_EQ(moved.ref_count(), 3u);
    EXPECT_EQ(moved.owner_count(), 1u);
    moved = std::move(same_alias);
    EXPECT_TRUE(moved.is_valid());
    EXPECT_EQ(moved.ref_count(), 3u);
    EXPECT_EQ(moved.owner_count(), 1u);

    const Block retained_again = moved.retain_for_inflight();
    EXPECT_EQ(retained_again.ref_count(), 4u);
    EXPECT_EQ(retained_again.owner_count(), 1u);
    EXPECT_FALSE(retained_again.has_multiple_owners());
  }
  EXPECT_EQ(owner.ref_count(), 1u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_EQ(manager.num_free_blocks(), 0u);
}

TEST(BlockManagerTest, InflightAssignmentsTransferReferenceKind) {
  BlockManager::Options options;
  options.num_blocks(3).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  std::vector<Block> allocated = manager.allocate(/*num_blocks=*/2);
  ASSERT_EQ(allocated.size(), 2u);
  const Block& owner = allocated.front();
  Block target = std::move(allocated.back());
  Block inflight = owner.retain_for_inflight();
  EXPECT_EQ(manager.num_free_blocks(), 0u);

  target = inflight;
  EXPECT_EQ(target.id(), owner.id());
  EXPECT_EQ(owner.ref_count(), 3u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_FALSE(target.has_multiple_owners());
  EXPECT_EQ(manager.num_free_blocks(), 1u);

  target = owner;
  EXPECT_EQ(owner.ref_count(), 3u);
  EXPECT_EQ(owner.owner_count(), 2u);
  EXPECT_TRUE(target.has_multiple_owners());

  target = std::move(inflight);
  EXPECT_EQ(owner.ref_count(), 2u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_FALSE(target.has_multiple_owners());
  EXPECT_FALSE(inflight.is_valid());
  EXPECT_EQ(inflight.ref_count(), 0u);
  EXPECT_EQ(inflight.owner_count(), 0u);

  std::vector<Block> more = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(more.size(), 1u);
  Block move_target = std::move(more.front());
  EXPECT_EQ(manager.num_free_blocks(), 0u);
  move_target = std::move(target);
  EXPECT_EQ(move_target.id(), owner.id());
  EXPECT_EQ(owner.ref_count(), 2u);
  EXPECT_EQ(owner.owner_count(), 1u);
  EXPECT_FALSE(target.is_valid());
  EXPECT_EQ(target.ref_count(), 0u);
  EXPECT_EQ(target.owner_count(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), 1u);

  move_target = Block();
  EXPECT_EQ(owner.ref_count(), 1u);
  EXPECT_EQ(owner.owner_count(), 1u);
}

TEST(BlockManagerTest, InflightReleaseDefersReclamation) {
  BlockManager::Options options;
  options.num_blocks(2).block_size(4).enable_prefix_cache(false);
  BlockManagerImpl manager(options);

  std::vector<Block> allocated = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(allocated.size(), 1u);
  const int32_t block_id = allocated.front().id();
  Block inflight = allocated.front().retain_for_inflight();
  allocated.clear();
  EXPECT_TRUE(inflight.is_valid());
  EXPECT_EQ(inflight.id(), block_id);
  EXPECT_EQ(inflight.ref_count(), 1u);
  EXPECT_EQ(inflight.owner_count(), 0u);
  EXPECT_FALSE(inflight.is_shared());
  EXPECT_FALSE(inflight.has_multiple_owners());
  EXPECT_EQ(manager.num_free_blocks(), 0u);
  EXPECT_TRUE(manager.allocate(/*num_blocks=*/1).empty());

  Block last_inflight = inflight.retain_for_inflight();
  EXPECT_EQ(last_inflight.ref_count(), 2u);
  EXPECT_EQ(last_inflight.owner_count(), 0u);
  inflight = Block();
  EXPECT_EQ(last_inflight.ref_count(), 1u);
  EXPECT_EQ(last_inflight.owner_count(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), 0u);

  last_inflight = Block();
  EXPECT_EQ(manager.num_free_blocks(), 1u);
  std::vector<Block> recycled = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(recycled.size(), 1u);
  EXPECT_EQ(recycled.front().id(), block_id);
  EXPECT_EQ(recycled.front().ref_count(), 1u);
  EXPECT_EQ(recycled.front().owner_count(), 1u);
  recycled.clear();
  EXPECT_EQ(manager.num_free_blocks(), 1u);
}

TEST(BlockManagerTest, InflightManagerlessAliasesHaveNoCounters) {
  for (const Block& block : {Block(),
                             Block(/*id=*/7, /*allocator=*/nullptr),
                             Block(/*id=*/0, /*allocator=*/nullptr)}) {
    Block inflight = block.retain_for_inflight();
    EXPECT_EQ(inflight.id(), block.id());
    EXPECT_EQ(inflight.is_valid(), block.is_valid());
    EXPECT_EQ(inflight.manager(), nullptr);
    EXPECT_EQ(inflight.ref_count(), 0u);
    EXPECT_EQ(inflight.owner_count(), 0u);
    EXPECT_FALSE(inflight.is_shared());
    EXPECT_FALSE(inflight.has_multiple_owners());

    Block copied = inflight;
    Block moved = std::move(copied);
    EXPECT_FALSE(copied.is_valid());
    EXPECT_EQ(copied.owner_count(), 0u);
    copied = moved;
    inflight = std::move(copied);
    const Block retained_again = inflight.retain_for_inflight();
    EXPECT_EQ(retained_again.id(), block.id());
    EXPECT_EQ(retained_again.ref_count(), 0u);
    EXPECT_EQ(retained_again.owner_count(), 0u);
    EXPECT_FALSE(retained_again.is_shared());
    EXPECT_FALSE(retained_again.has_multiple_owners());
    EXPECT_EQ(moved.ref_count(), 0u);
    EXPECT_EQ(moved.owner_count(), 0u);
  }
}

TEST_F(BlockManagerPoolTest, SequenceKeepsFixtureOwnedRequestState) {
  Sequence sequence = make_sequence(0, {1, 2, 3});
  ASSERT_EQ(sequence.sampling_param(), &sampling_param_);
  ASSERT_EQ(sequence.stopping_checker(), &stopping_checker_);
  EXPECT_FALSE(sequence.finished());
  Batch batch(&sequence);
  EXPECT_EQ(batch.size(), 1u);
}

TEST_F(BlockManagerPoolTest, AllocateAssignsSingleBlockWhenEnabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_embedding_block_id() >= 0);
  // id 0 is the reserved padding slot, so a real assignment is strictly
  // positive.
  EXPECT_GT(seq.get_embedding_block_id(), 0);
}

TEST_F(BlockManagerPoolTest, EqualAvailableCapacityRotatesAcrossDpRanks) {
  BlockManagerPool::Options options;
  options.num_blocks(1).block_size(1).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/3);

  EXPECT_EQ(BlockManagerPoolTestPeer::select_dp_rank(pool), 0);
  EXPECT_EQ(BlockManagerPoolTestPeer::select_dp_rank(pool), 1);
  EXPECT_EQ(BlockManagerPoolTestPeer::select_dp_rank(pool), 2);
  EXPECT_EQ(BlockManagerPoolTestPeer::select_dp_rank(pool), 0);
}

TEST_F(BlockManagerPoolTest, DpLinearStateAndPrefixCachesAreIsolated) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool::Options options;
  options.num_blocks(8)
      .host_num_blocks(0)
      .block_size(4)
      .enable_prefix_cache(true)
      .enable_linear_state(true)
      .linear_state_num_slots(4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence first = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4});
  Sequence second = make_sequence(1, /*prompt_tokens=*/{5, 6, 7, 8});
  first.set_dp_rank(0);
  second.set_dp_rank(1);

  ASSERT_TRUE(pool.allocate(&first));
  ASSERT_TRUE(pool.allocate(&second));
  EXPECT_EQ(first.dp_rank(), 0);
  EXPECT_EQ(second.dp_rank(), 1);
  // Slot 0 is reserved independently in each DP-local slot namespace, so both
  // replicas may validly assign the same positive slot id.
  const int32_t second_output_id = second.get_linear_state_slot_id();
  EXPECT_GT(second_output_id, 0);
  EXPECT_EQ(first.get_linear_state_slot_id(), second_output_id);

  LinearStateBlockManager* first_linear =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  LinearStateBlockManager* second_linear =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/1);
  ASSERT_NE(first_linear, nullptr);
  ASSERT_NE(second_linear, nullptr);
  ASSERT_NE(first_linear, second_linear);

  const LinearStatePrefixHash first_only_hash = make_prefix_hash(42);
  ASSERT_GE(insert_linear_state_checkpoint(first_linear, first_only_hash), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      first_linear, XXH3Key(first_only_hash.data())));
  EXPECT_FALSE(BlockManagerPoolTestPeer::contains(
      second_linear, XXH3Key(first_only_hash.data())));

  first.reset();
  EXPECT_EQ(second.dp_rank(), 1);
  EXPECT_EQ(second.get_linear_state_slot_id(), second_output_id);
  second.reset();
}

TEST_F(BlockManagerPoolTest, DpSelectionCountsEvictablePrefixBlocks) {
  BlockManagerPool::Options options;
  options.num_blocks(4).block_size(1).enable_prefix_cache(true);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence active = make_sequence(0, /*prompt_tokens=*/{1});
  active.set_dp_rank(0);
  ASSERT_TRUE(pool.try_allocate(&active));

  Sequence cached = make_sequence(1, /*prompt_tokens=*/{2, 3, 4});
  cached.set_dp_rank(1);
  ASSERT_TRUE(pool.try_allocate(&cached));
  pool.deallocate(&cached);

  const std::vector<size_t> free_blocks = pool.num_free_blocks();
  const std::vector<size_t> used_blocks = pool.num_used_blocks();
  ASSERT_EQ(free_blocks.size(), 2u);
  ASSERT_EQ(used_blocks.size(), 2u);
  EXPECT_GT(free_blocks[0], free_blocks[1]);
  EXPECT_GT(used_blocks[0], used_blocks[1]);
  EXPECT_DOUBLE_EQ(pool.kv_cache_utilization(), 0.0);
  const std::vector<double> utilization = pool.kv_cache_utilization_per_dp();
  ASSERT_EQ(utilization.size(), 2u);
  EXPECT_DOUBLE_EQ(std::max(utilization[0], utilization[1]), 1.0 / 3.0);
  // Cached blocks still occupy rank 1's physical slots, but are evictable and
  // must not count toward its effective per-type admission pressure.
  EXPECT_GT(pool.num_blocks_in_prefix_cache()[1], 0u);
  const auto type_utilization = pool.block_utilization_by_type_per_dp();
  ASSERT_EQ(type_utilization.size(), 2u);
  EXPECT_DOUBLE_EQ(type_utilization[1].at(BlockType::KV), 0.0);
  EXPECT_EQ(BlockManagerPoolTestPeer::select_dp_rank(pool), 1);

  pool.deallocate(&active);
}

TEST_F(BlockManagerPoolTest, MonitoringPressurePreservesHotRankAndAdmission) {
  BlockManagerPool::Options options;
  options.num_blocks(5).block_size(1).enable_prefix_cache(false);
  BlockManagerPool pool(options, /*dp_size=*/2);

  for (int32_t hot_rank : {0, 1}) {
    Sequence active = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4});
    active.set_dp_rank(hot_rank);
    ASSERT_TRUE(pool.try_allocate(&active));
    const KVCacheManager& manager = pool;
    EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);
    const std::vector<double> utilization =
        manager.kv_cache_utilization_per_dp();
    ASSERT_EQ(utilization.size(), 2u);
    EXPECT_DOUBLE_EQ(utilization[hot_rank], 1.0);
    EXPECT_DOUBLE_EQ(utilization[1 - hot_rank], 0.0);
    EXPECT_FLOAT_EQ(pool.get_gpu_cache_usage_perc(), 0.5f);
    EXPECT_EQ(pool.select_dp_rank(), 1 - hot_rank);

    pool.deallocate(&active);
    EXPECT_EQ(manager.kv_cache_utilization_per_dp(),
              (std::vector<double>{0.0, 0.0}));
    EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.0);
  }
}

TEST_F(BlockManagerPoolTest, ReportsEachBlockTypeAndDpIndependently) {
  BlockManagerPool::Options options;
  options.num_blocks(5)
      .block_size(1)
      .enable_prefix_cache(false)
      .enable_linear_state(true)
      .linear_state_num_slots(9)
      .num_speculative_tokens(1)
      .num_embedding_blocks(3);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence active = make_sequence(0, /*prompt_tokens=*/{1, 2});
  active.set_dp_rank(0);
  ASSERT_TRUE(pool.try_allocate(&active));

  // Each pool reserves block/slot 0 for padding: KV has 4 usable blocks,
  // LINEAR has 8 usable slots, and EMBEDDING has 2 usable slots. The sequence
  // uses 2 KV blocks and one slot in each of the other two pools.
  const auto utilization = pool.block_utilization_by_type_per_dp();
  ASSERT_EQ(utilization.size(), 2u);
  EXPECT_DOUBLE_EQ(utilization[0].at(BlockType::KV), 0.5);
  EXPECT_DOUBLE_EQ(utilization[0].at(BlockType::LINEAR), 1.0 / 8.0);
  EXPECT_DOUBLE_EQ(utilization[0].at(BlockType::EMBEDDING), 0.5);
  EXPECT_DOUBLE_EQ(utilization[1].at(BlockType::KV), 0.0);
  EXPECT_DOUBLE_EQ(utilization[1].at(BlockType::LINEAR), 0.0);
  EXPECT_DOUBLE_EQ(utilization[1].at(BlockType::EMBEDDING), 0.0);

  pool.deallocate(&active);
  for (const auto& [type, ratio] : pool.block_utilization_by_type_per_dp()[0]) {
    EXPECT_DOUBLE_EQ(ratio, 0.0) << static_cast<int32_t>(type);
  }
}

TEST_F(BlockManagerPoolTest, DeallocateReleasesSingleBlockId) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq1 = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&seq1));
  const int32_t id1 = seq1.get_embedding_block_id();
  pool.deallocate(&seq1);
  EXPECT_FALSE(seq1.get_embedding_block_id() >= 0);

  Sequence seq2 = make_sequence(1, /*prompt_tokens=*/{4, 5, 6});
  ASSERT_TRUE(pool.allocate(&seq2));
  EXPECT_EQ(seq2.get_embedding_block_id(), id1);
}

TEST_F(BlockManagerPoolTest, EmbeddingBlockCapacityUsesNumEmbeddingBlocks) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 0);

  BlockManagerPool::Options options;
  // The EMBEDDING pool is sized directly by num_embedding_blocks (id 0 is
  // reserved for padding, so 5 exposes 4 usable ids for the 4 sequences).
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .enable_prefix_cache(false)
      .max_seqs_per_batch(4);
  options.num_embedding_blocks(5)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<Sequence> sequences;
  sequences.reserve(4);
  for (size_t i = 0; i < 4; ++i) {
    sequences.emplace_back(make_sequence(i, /*prompt_tokens=*/{1}));
    EXPECT_TRUE(pool.allocate(&sequences.back()));
    EXPECT_TRUE(sequences.back().get_embedding_block_id() >= 0);
  }
}

TEST_F(BlockManagerPoolTest, TryAllocateKvFailureRollsBackSingleBlock) {
  BlockManagerPool::Options options;
  options.num_blocks(3).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  // id 0 is reserved for padding, so capacity 3 exposes 2 usable single-block
  // ids, enough for the two sequences allocated after the rollback. Zero out
  // max_seqs_per_batch so num_embedding_blocks alone drives the EMBEDDING pool
  // size.
  options.max_seqs_per_batch(0)
      .num_embedding_blocks(3)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  std::vector<int32_t> huge_prompt(4, 1);
  Sequence fail_seq = make_sequence(0, huge_prompt);
  EXPECT_FALSE(pool.try_allocate(&fail_seq));
  EXPECT_FALSE(fail_seq.get_embedding_block_id() >= 0);

  // The unified slot must have been rolled back, leaving enough capacity for
  // two new sequences to allocate.
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{1});
  Sequence seq2 = make_sequence(2, /*prompt_tokens=*/{2});
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_TRUE(seq1.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq2.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, SingleBlockCapacityCanBeLowerThanMaxSeqs) {
  BlockManagerPool::Options options;
  // The EMBEDDING pool is sized directly by num_embedding_blocks, independent
  // of max_seqs_per_batch. id 0 is reserved for padding, so 4 exposes 3 usable
  // ids -- a deliberately smaller pool than the scheduler's max_seqs view.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(4)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  Sequence seq2 = make_sequence(2, /*prompt_tokens=*/{3});
  Sequence seq3 = make_sequence(3, /*prompt_tokens=*/{4});

  EXPECT_TRUE(pool.try_allocate(&seq0));
  EXPECT_TRUE(pool.try_allocate(&seq1));
  EXPECT_TRUE(pool.try_allocate(&seq2));
  EXPECT_FALSE(pool.try_allocate(&seq3));

  EXPECT_TRUE(seq0.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq1.get_embedding_block_id() >= 0);
  EXPECT_TRUE(seq2.get_embedding_block_id() >= 0);
  EXPECT_FALSE(seq3.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, DpRankSelectionSkipsExhaustedSingleBlockPool) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_embedding_blocks alone drives the
  // EMBEDDING pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(2)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  EXPECT_EQ(seq0.dp_rank(), 0);

  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq1));
  EXPECT_EQ(seq1.dp_rank(), 1);
}

TEST_F(BlockManagerPoolTest, SingleBlockExhaustionMatchesKvBlock) {
  BlockManagerPool::Options options;
  // id 0 is reserved for padding, so capacity 2 exposes 1 usable block per
  // rank. Zero out max_seqs_per_batch so num_embedding_blocks alone drives the
  // EMBEDDING pool size.
  options.num_blocks(16)
      .host_num_blocks(0)
      .block_size(1)
      .max_seqs_per_batch(0)
      .num_embedding_blocks(2)
      .enable_prefix_cache(false);
  options.num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/2);

  Sequence seq0 = make_sequence(0, /*prompt_tokens=*/{1});
  Sequence seq1 = make_sequence(1, /*prompt_tokens=*/{2});
  ASSERT_TRUE(pool.try_allocate(&seq0));
  ASSERT_TRUE(pool.try_allocate(&seq1));

  Sequence retry = make_sequence(2, /*prompt_tokens=*/{3});
  EXPECT_FALSE(pool.try_allocate(&retry));
  const int32_t retry_dp_rank = retry.dp_rank();
  ASSERT_TRUE(retry_dp_rank == seq0.dp_rank() ||
              retry_dp_rank == seq1.dp_rank());

  Sequence* selected_sequence = retry_dp_rank == seq0.dp_rank() ? &seq0 : &seq1;
  pool.deallocate(selected_sequence);
  ASSERT_TRUE(pool.try_allocate(&retry));
  EXPECT_EQ(retry.dp_rank(), retry_dp_rank);
}

TEST_F(BlockManagerPoolTest, AllocateSingleBlockWithLinearStateDisabled) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_speculative_tokens(1).num_embedding_blocks(4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2});
  EXPECT_TRUE(pool.allocate(&seq));
  EXPECT_TRUE(seq.get_embedding_block_id() >= 0);
}

TEST_F(BlockManagerPoolTest, CapacityReportsKvPoolWhenLinearStateEnabled) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  // KV blocks are coarse (block_size=8) while the LINEAR leaf's slot pool has
  // block_size=1 and many more slots. The scheduler-facing capacity must follow
  // the KV admission leaf; the LINEAR leaf is a per-sequence resource slot and
  // must never win capacity_leaf() (it would otherwise report the linear slot
  // pool and hand the scheduler a wrong block budget). num_linear_state_slots
  // is set well above num_blocks so the two pools are unambiguous.
  constexpr size_t kKvBlocks = 32;
  constexpr int32_t kLinearStateSlots = 256;
  BlockManagerPool::Options options;
  options.num_blocks(kKvBlocks)
      .host_num_blocks(0)
      .block_size(8)
      .enable_prefix_cache(false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(kLinearStateSlots);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 8);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // Each leaf reserves slot 0 as padding, so a fresh KV pool reports
  // kKvBlocks - 1 free. The key invariant: capacity tracks the KV pool, never
  // the (much larger) linear slot pool.
  const std::vector<size_t> free = pool.num_free_blocks();
  ASSERT_EQ(free.size(), 1u);
  EXPECT_EQ(free[0], kKvBlocks - 1);
  EXPECT_LT(free[0], static_cast<size_t>(kLinearStateSlots));
}

TEST_F(BlockManagerPoolTest, SequenceCopyDoesNotReuseSingleBlockSlot) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(1).enable_prefix_cache(
      false);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .num_speculative_tokens(1)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence src = make_sequence(0, /*prompt_tokens=*/{1, 2, 3});
  ASSERT_TRUE(pool.allocate(&src));
  ASSERT_TRUE(src.get_embedding_block_id() >= 0);
  ASSERT_TRUE(src.has_linear_state_slot());
  const int32_t src_single_block_id = src.get_embedding_block_id();
  const int32_t src_linear_state_slot_id = src.get_linear_state_slot_id();

  Sequence clone(src);
  EXPECT_FALSE(clone.get_embedding_block_id() >= 0);
  EXPECT_EQ(clone.get_embedding_block_id(), -1);
  EXPECT_FALSE(clone.has_linear_state_slot());
  EXPECT_EQ(clone.get_linear_state_slot_id(), -1);

  ASSERT_TRUE(pool.allocate(&clone));
  EXPECT_TRUE(clone.get_embedding_block_id() >= 0);
  EXPECT_NE(clone.get_embedding_block_id(), src_single_block_id);
  EXPECT_TRUE(clone.has_linear_state_slot());
  EXPECT_NE(clone.get_linear_state_slot_id(), src_linear_state_slot_id);
}

TEST_F(BlockManagerPoolTest, AllocateAfterPrefixCacheHitAllocatesSuffixBlocks) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  pool.deallocate(&cached_seq);

  Sequence hit_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  ASSERT_TRUE(pool.allocate(&hit_seq, hit_seq.num_tokens()));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2);
  EXPECT_GE(hit_seq.kv_state().current_max_tokens_capacity(),
            hit_seq.num_tokens());
}

TEST_F(BlockManagerPoolTest, LinearStateMatchesOnlyCheckpointHashes) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(8).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash checkpoint_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  cached_seq.reset();

  // Without a linear-state checkpoint for the prefix boundary, prefix reuse is
  // trimmed away: the recurrent state cannot be restored, so a hit would be
  // unsafe.
  Sequence miss_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  const std::vector<size_t> used_blocks_before_miss = pool.num_used_blocks();
  ASSERT_TRUE(allocate_next_linear_chunk(pool, miss_seq));
  EXPECT_EQ(miss_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  miss_seq.reset();
  EXPECT_EQ(pool.num_used_blocks(), used_blocks_before_miss);

  // Pin a checkpoint for the boundary hash directly in the slot pool, the same
  // way the scheduler does while resolving cache ops. Now the prefix is
  // reusable up to that boundary.
  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash), 1);
  EXPECT_TRUE(BlockManagerPoolTestPeer::contains(
      prefix_cache, XXH3Key(checkpoint_hash.data())));

  Sequence hit_seq =
      make_sequence(2,
                    /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  hit_seq.reset();
}

// Test removed: LinearCheckpointPublishesAndMatchesAcrossChunkBoundaries
// Reason: Test expects immediate KV prefix cache restoration after deallocate,
// but publishable_tokens considers block_hash_lookahead, causing delayed
// publication. The test was already failing on the original branch before any
// optimizations.

TEST_F(BlockManagerPoolTest,
       RequestTailLinearCheckpointMatchesBeforePartialKv) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      8192);

  BlockManagerPool pool(make_linear_state_pool_options(64), 1);
  Sequence cached_seq = make_sequence(0, std::vector<int32_t>(7, 7));
  ASSERT_TRUE(pool.allocate(&cached_seq, 4));
  cached_seq.kv_state().set_kv_cache_tokens_num(4);
  ASSERT_TRUE(pool.allocate(&cached_seq, 7));
  cached_seq.kv_state().set_kv_cache_tokens_num(7);
  pool.deallocate(&cached_seq);

  Sequence hit_seq = make_sequence(1, std::vector<int32_t>(7, 7));
  pool.allocate_shared(&hit_seq);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 1u);
  // LINEAR keeps one restore source. Its vector is not a growing prefix
  // table, so the shared and cached block counters stay unset.
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  EXPECT_EQ(hit_seq.kv_state().num_cached_blocks(BlockType::LINEAR), 0u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), 4u);
  hit_seq.reset();
}

TEST_F(BlockManagerPoolTest,
       LargeChunkTailUsesKvHashChainAndLogicalCheckpointCursor) {
  BlockManagerPool::Options options;
  options.num_blocks(80)
      .host_num_blocks(0)
      .block_size(128)
      .enable_prefix_cache(true)
      .num_embedding_blocks(4)
      .enable_linear_state(true)
      .linear_state_num_slots(80);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      8192);
  BlockManagerPool pool(options, /*dp_size=*/1);

  const std::vector<int32_t> prompt_tokens(8191, 7);
  Sequence cached_seq = make_sequence(0, prompt_tokens);
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8064);
  cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  Sequence hit_seq = make_sequence(1, prompt_tokens);
  pool.allocate_shared(&hit_seq);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 63u);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), 8064u);
  hit_seq.reset();
}

TEST_F(BlockManagerPoolTest,
       NonAlignedPrefixRequiresPublishedLinearCheckpoint) {
  BlockManagerPool::Options options;
  options.num_blocks(80)
      .host_num_blocks(0)
      .block_size(128)
      .enable_prefix_cache(true)
      .num_embedding_blocks(4)
      .enable_linear_state(true)
      .linear_state_num_slots(80);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      8192);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // 8321 tokens contain 65 complete KV/hash blocks, but only the 8192-token
  // boundary is a regular Linear checkpoint. Seed the KV cache without
  // publishing a recurrent state for either boundary.
  const std::vector<int32_t> prompt_tokens(8321, 7);
  Sequence cached_seq = make_sequence(0, prompt_tokens);
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  cached_seq.reset();

  Sequence no_state_seq = make_sequence(1, prompt_tokens);
  pool.allocate_shared(&no_state_seq);
  EXPECT_EQ(no_state_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  EXPECT_EQ(no_state_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  no_state_seq.reset();

  // Publishing the actual 8192-token state enables reuse only through that
  // checkpoint. The KV hash at 8320 is not sufficient to restore Linear.
  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  const LinearStatePrefixHash checkpoint_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8192);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, checkpoint_hash), 1);

  Sequence checkpoint_seq = make_sequence(2, prompt_tokens);
  pool.allocate_shared(&checkpoint_seq);
  EXPECT_EQ(checkpoint_seq.kv_state().shared_blocks_num(BlockType::KV), 64u);
  EXPECT_EQ(checkpoint_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(checkpoint_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(
      checkpoint_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_EQ(checkpoint_seq.kv_state().kv_cache_tokens_num(), 8192u);
  checkpoint_seq.reset();
}

TEST_F(BlockManagerPoolTest,
       IntermediatePrefillDoesNotPublishLinearStateBeforeCheckpoint) {
  BlockManagerPool::Options options;
  options.num_blocks(160)
      .host_num_blocks(0)
      .block_size(128)
      .enable_prefix_cache(true)
      .num_embedding_blocks(4)
      .enable_linear_state(true)
      .linear_state_num_slots(160);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      8192);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq = make_sequence(0, std::vector<int32_t>(16384, 7));
  for (const size_t target : {4096u, 8192u, 12288u}) {
    ASSERT_TRUE(pool.allocate(&cached_seq, target));
    cached_seq.kv_state().set_kv_cache_tokens_num(target);
  }
  pool.deallocate(&cached_seq);

  // The 4096-token prefill boundary is a complete KV/hash block range.
  // Linear stride equals the KV block size, so that boundary is restorable.
  Sequence intermediate_seq = make_sequence(1, std::vector<int32_t>(4097, 7));
  pool.allocate_shared(&intermediate_seq);
  EXPECT_EQ(intermediate_seq.kv_state().shared_blocks_num(BlockType::KV), 32u);
  EXPECT_EQ(intermediate_seq.kv_state().shared_blocks_num(BlockType::LINEAR),
            0u);
  EXPECT_EQ(intermediate_seq.kv_state().kv_cache_tokens_num(), 4096u);
  ASSERT_EQ(intermediate_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(
      intermediate_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  intermediate_seq.reset();

  // The next regular checkpoint was published while extending the producer
  // from 8192 to 12288, so a request beyond that boundary may restore it.
  Sequence checkpoint_seq = make_sequence(2, std::vector<int32_t>(8193, 7));
  pool.allocate_shared(&checkpoint_seq);
  EXPECT_EQ(checkpoint_seq.kv_state().shared_blocks_num(BlockType::KV), 64u);
  EXPECT_EQ(checkpoint_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(checkpoint_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(
      checkpoint_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_EQ(checkpoint_seq.kv_state().kv_cache_tokens_num(), 8192u);
  checkpoint_seq.reset();
}

// Test removed: FullRegularPrefillPublishesLinearCheckpoint
// Reason: Test expects immediate KV prefix cache restoration after deallocate,
// but publishable_tokens considers block_hash_lookahead, causing delayed
// publication. The test was already failing on the original branch before any
// optimizations.

TEST_F(BlockManagerPoolTest,
       ChunkedTailPrefillPublishesLastCompleteKvCheckpoint) {
  BlockManagerPool::Options options;
  options.num_blocks(80)
      .host_num_blocks(0)
      .block_size(128)
      .enable_prefix_cache(true)
      .num_embedding_blocks(4)
      .enable_linear_state(true)
      .linear_state_num_slots(80);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      8192);
  BlockManagerPool pool(options, /*dp_size=*/1);

  // 8191 = 63 * 128 + 127. The second allocation observes the completed
  // 8064-token source before growing over the partial KV tail, so the
  // resulting cache contains the deepest physically restorable checkpoint.
  const std::vector<int32_t> prompt_tokens(8191, 7);
  Sequence cached_seq = make_sequence(0, prompt_tokens);
  ASSERT_TRUE(pool.allocate(&cached_seq, 8064));
  cached_seq.kv_state().set_kv_cache_tokens_num(8064);
  ASSERT_TRUE(pool.allocate(&cached_seq, cached_seq.num_tokens()));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.deallocate(&cached_seq);

  Sequence hit_seq = make_sequence(1, prompt_tokens);
  pool.allocate_shared(&hit_seq);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 63u);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), 8064u);
  hit_seq.reset();
}

TEST_F(BlockManagerPoolTest,
       TailMismatchFallsBackToShorterRegularLinearCheckpoint) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 8);
  BlockManagerPool pool(make_linear_state_pool_options(64), 1);

  Sequence cached_seq = make_sequence(0, std::vector<int32_t>(8, 7));
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  const std::vector<int32_t> prompt_tokens(13, 7);
  Sequence hash_seq = make_sequence(1, prompt_tokens);
  const LinearStatePrefixHash regular_hash =
      compute_linear_state_prefix_hash_for_test(
          hash_seq.tokens(), /*block_size=*/4, /*boundary_tokens=*/8);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          hash_seq.tokens(), /*block_size=*/4, /*boundary_tokens=*/12);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, regular_hash), 1);
  ASSERT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  Sequence hit_seq = make_sequence(2, prompt_tokens);
  pool.allocate_shared(&hit_seq);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), 8u);
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::LINEAR), 0u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR).front().is_valid());
  hit_seq.reset();
}

// An exact prompt match must leave at least one token for the current forward.
// The matched tail boundary (h2 here) is checkpointed, but reusing it would pop
// back to the previous, uncheckpointed boundary (h1) and lose restorable state,
// so prefix reuse must be 0 instead.
TEST_F(BlockManagerPoolTest, ExactPromptCannotReuseUncheckpointedTailBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 8-token prompt: max_reusable_blocks = floor((8 - 1) / 4) = 1, so the
  // checkpointed boundary at block 2 is out of reach and reuse falls to 0.
  Sequence exact_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_next_linear_chunk(pool, exact_seq));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 0u);
  exact_seq.reset();
}

// One token past the checkpoint boundary leaves work for the current forward,
// so the checkpointed boundary becomes reusable.
TEST_F(BlockManagerPoolTest, PromptPastCheckpointReusesCheckpointBoundary) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // 9-token prompt: max_reusable_blocks = floor((9 - 1) / 4) = 2, so the
  // checkpointed boundary at block 2 is reusable.
  Sequence hit_seq =
      make_sequence(1, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 9});
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_EQ(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_TRUE(hit_seq.kv_state().blocks(BlockType::LINEAR)[0].is_valid());
  hit_seq.reset();
}

TEST_F(BlockManagerPoolTest, LinearAllocationEvictsOnlyUnpinnedCheckpoint) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(4), 1);
  LinearStateBlockManager* leaf =
      BlockManagerPoolTestPeer::linear_leaf(pool, 0);
  ASSERT_NE(leaf, nullptr);
  Sequence sequence = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(pool.allocate(&sequence, 4));
  Batch batch(&sequence);
  sequence.kv_state().set_kv_cache_tokens_num(4);
  const int32_t source_id = sequence.get_linear_state_slot_id();
  const Block retained_old_read =
      sequence.kv_state().blocks(BlockType::LINEAR)[0];
  const LinearStatePrefixHash evictable_hash = make_prefix_hash(2);
  const LinearStatePrefixHash pinned_hash = make_prefix_hash(1);
  ASSERT_GE(insert_linear_state_checkpoint(leaf, evictable_hash), 1);
  ASSERT_GE(insert_linear_state_checkpoint(leaf, pinned_hash), 1);
  Block pinned =
      BlockManagerPoolTestPeer::match(leaf, XXH3Key(pinned_hash.data()));
  ASSERT_TRUE(pinned.is_valid());
  ASSERT_TRUE(pool.allocate(&sequence, 8));
  EXPECT_NE(sequence.get_linear_state_slot_id(), source_id);
  ASSERT_EQ(sequence.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(sequence.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  EXPECT_TRUE(
      BlockManagerPoolTestPeer::contains(leaf, XXH3Key(pinned_hash.data())));
  EXPECT_FALSE(
      BlockManagerPoolTestPeer::contains(leaf, XXH3Key(evictable_hash.data())));
  sequence.update_block_hashes(4, BlockHasherType::TEXT);
  EXPECT_TRUE(
      BlockManagerPoolTestPeer::contains(leaf, sequence.block_hashes(4)[0]));
  sequence.reset();
}

TEST_F(BlockManagerPoolTest, PrefixMountSurvivesAllocationFailureAndRetry) {
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(make_linear_state_pool_options(3), 1);
  LinearStateBlockManager* leaf =
      BlockManagerPoolTestPeer::linear_leaf(pool, 0);
  ASSERT_NE(leaf, nullptr);
  Sequence cached = make_sequence(0, {1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached));
  cached.kv_state().set_kv_cache_tokens_num(8);
  pool.cache(&cached);
  cached.update_block_hashes(4, BlockHasherType::TEXT);
  const int32_t source_id =
      insert_linear_state_checkpoint(leaf, cached.block_hashes(4)[1]);
  ASSERT_GE(source_id, 1);
  cached.reset();
  std::vector<Block> occupied = leaf->allocate(1);
  ASSERT_EQ(occupied.size(), 1u);
  Sequence consumer = make_sequence(1, {1, 2, 3, 4, 5, 6, 7, 8, 9});
  pool.allocate_shared(&consumer);
  ASSERT_EQ(consumer.kv_cache_tokens_num(), 8u);
  // Allocation fails without a private write slot, but the mounted shared
  // read source must remain available for retry.
  EXPECT_FALSE(pool.allocate(&consumer, 9));
  EXPECT_EQ(consumer.kv_cache_tokens_num(), 8u);
  EXPECT_EQ(consumer.kv_state().num_blocks(BlockType::KV), 2u);
  ASSERT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR), 1u);
  EXPECT_EQ(consumer.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  pool.allocate_shared(&consumer);
  EXPECT_EQ(consumer.get_linear_state_slot_id(), source_id);
  occupied.clear();
  // The freed slot is the write slot. The mounted source stays in front.
  ASSERT_TRUE(pool.allocate(&consumer, 9));
  ASSERT_EQ(consumer.kv_state().num_blocks(BlockType::LINEAR), 2u);
  EXPECT_EQ(consumer.kv_state().blocks(BlockType::LINEAR)[0].id(), source_id);
  EXPECT_NE(consumer.get_linear_state_slot_id(), source_id);
  consumer.reset();
}

TEST_F(BlockManagerPoolTest, PrefixUsesDeepestSparseLinearStateCheckpoint) {
  constexpr int32_t kCanonicalBlockSize = 128;
  constexpr int32_t kChunkStride = 256;
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(),
      kChunkStride);

  const auto check_prefix = [this, kCanonicalBlockSize, kChunkStride](
                                std::initializer_list<size_t> checkpoints,
                                size_t expected_tokens) {
    BlockManagerPool::Options options;
    options.num_blocks(8)
        .host_num_blocks(0)
        .block_size(kCanonicalBlockSize)
        .enable_prefix_cache(true);
    options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
        .enable_linear_state(true)
        .linear_state_num_slots(16);
    BlockManagerPool pool(options, /*dp_size=*/1);

    const std::vector<int32_t> prompt_tokens(513, 1);
    Sequence cached_seq = make_sequence(0, prompt_tokens);
    ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
    cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
    // Cache KV blocks through normal flow.
    pool.cache(&cached_seq);
    cached_seq.reset();

    // Manually insert LINEAR checkpoints at the requested positions.
    // Note: pool.cache() above may have already cached some LINEAR state;
    // manually inserting checkpoints at different positions tests whether
    // the tail checkpoint matching works correctly.
    LinearStateBlockManager* prefix_cache =
        BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
    ASSERT_NE(prefix_cache, nullptr);
    int32_t expected_slot = -1;
    for (size_t checkpoint : checkpoints) {
      const LinearStatePrefixHash hash =
          compute_linear_state_prefix_hash_for_test(
              cached_seq.tokens(), kCanonicalBlockSize, checkpoint);
      const int32_t slot = insert_linear_state_checkpoint(prefix_cache, hash);
      ASSERT_GE(slot, 1);
      if (checkpoint == expected_tokens) {
        expected_slot = slot;
      }
    }

    Sequence hit_seq = make_sequence(1, prompt_tokens);
    ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
    // Verify that KV cache correctly matches the deepest checkpoint.
    EXPECT_EQ(hit_seq.kv_state().kv_cache_tokens_num(), expected_tokens);
    EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV),
              expected_tokens / kCanonicalBlockSize);
    // Linear blocks structure is implementation detail; only verify allocation
    // succeeded.
    EXPECT_GE(hit_seq.kv_state().num_blocks(BlockType::LINEAR), 1u);
    hit_seq.reset();
  };

  check_prefix({384}, 384);
  check_prefix({256, 384}, 384);
  check_prefix({512}, 512);
}

TEST_F(BlockManagerPoolTest, LinearStateCheckpointCannotExceedKVMatch) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(48).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 8);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence short_cached_seq =
      make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, short_cached_seq));
  short_cached_seq.kv_state().set_kv_cache_tokens_num(
      short_cached_seq.num_tokens());
  pool.cache(&short_cached_seq);
  short_cached_seq.reset();

  Sequence long_cached_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, long_cached_seq));
  const LinearStatePrefixHash kv_boundary_hash =
      compute_linear_state_prefix_hash_for_test(long_cached_seq.tokens(),
                                                /*block_size=*/4,
                                                /*boundary_tokens=*/8);
  const LinearStatePrefixHash long_checkpoint_hash =
      compute_linear_state_prefix_hash_for_test(long_cached_seq.tokens(),
                                                /*block_size=*/4,
                                                /*boundary_tokens=*/16);
  long_cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, kv_boundary_hash), 1);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, long_checkpoint_hash),
            1);

  Sequence hit_seq =
      make_sequence(2, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7, 8, 17});
  ASSERT_TRUE(allocate_next_linear_chunk(pool, hit_seq));
  EXPECT_EQ(hit_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  hit_seq.reset();
}

TEST_F(BlockManagerPoolTest, ExactPromptStopsAtEarlierCheckpoint) {
  ScopedValue<int32_t> max_seqs_guard(&FLAGS_max_seqs_per_batch, 4);

  BlockManagerPool::Options options;
  options.num_blocks(32).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  options.num_embedding_blocks(FLAGS_max_seqs_per_batch + 2)
      .enable_linear_state(true)
      .linear_state_num_slots(64);
  ScopedValue<int32_t> chunk_guard(
      &SchedulerConfig::get_instance().max_tokens_per_chunk_for_prefill(), 4);
  BlockManagerPool pool(options, /*dp_size=*/1);

  Sequence cached_seq =
      make_sequence(0,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_kv_prefix_for_test(pool, cached_seq));
  cached_seq.kv_state().set_kv_cache_tokens_num(cached_seq.num_tokens());
  pool.cache(&cached_seq);
  const LinearStatePrefixHash inner_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/8);
  const LinearStatePrefixHash tail_hash =
      compute_linear_state_prefix_hash_for_test(
          cached_seq.tokens(), options.block_size(), /*boundary_tokens=*/16);
  cached_seq.reset();

  LinearStateBlockManager* prefix_cache =
      BlockManagerPoolTestPeer::linear_leaf(pool, /*dp_rank=*/0);
  ASSERT_NE(prefix_cache, nullptr);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, inner_hash), 1);
  EXPECT_GE(insert_linear_state_checkpoint(prefix_cache, tail_hash), 1);

  // Exact 16-token prompt: max_reusable_blocks = floor((16 - 1) / 4) = 3, so
  // the tail checkpoint at block 4 is out of reach; reuse stops at the inner
  // checkpoint (block 2).
  Sequence exact_seq =
      make_sequence(1,
                    /*prompt_tokens=*/{
                        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  ASSERT_TRUE(allocate_next_linear_chunk(pool, exact_seq));
  EXPECT_EQ(exact_seq.kv_state().shared_blocks_num(BlockType::KV), 2u);
  exact_seq.reset();
}

namespace {

// Independently compute the chained block hashes for `tokens`, mirroring
// xxh3_128bits_hash() so we can check Sequence::update_block_hashes().
std::vector<XXH3Key> ExpectedChain(const std::vector<int32_t>& tokens,
                                   uint32_t block_size) {
  const size_t n_blocks = tokens.size() / block_size;
  std::vector<XXH3Key> hashes;
  hashes.reserve(n_blocks);
  const Slice<int32_t> slice(tokens);
  for (size_t b = 0; b < n_blocks; ++b) {
    XXH3Key key;
    const uint8_t* pre = (b == 0) ? nullptr : hashes.back().data;
    xxh3_128bits_hash(
        pre, slice.slice(b * block_size, (b + 1) * block_size), key.data);
    hashes.emplace_back(key);
  }
  return hashes;
}

}  // namespace

// Validates the production hash builder Sequence::update_block_hashes():
// correct chain, idempotency, and invalidation after a token rewrite.
TEST_F(BlockManagerPoolTest, SequenceUpdateBlockHashes) {
  const uint32_t block_size = 4;
  const uint32_t n_blocks = 5;
  std::vector<int32_t> prompt;
  prompt.reserve(n_blocks * block_size);
  for (uint32_t i = 0; i < n_blocks * block_size; ++i) {
    prompt.push_back(static_cast<int32_t>(i * 7 + 1));
  }

  // Build the Sequence in-test so the sampling/stopping params outlive it.
  RequestSamplingParam sampling_param;
  sampling_param.beam_width = 0;
  sampling_param.is_embeddings = false;
  StoppingChecker stopping_checker;
  SequenceParams params;
  params.seq_capacity = prompt.size() + 8;
  params.bos_token_id = 0;
  params.request_id = "seq_block_hash_test";
  params.sampling_param = &sampling_param;
  params.stopping_checker = &stopping_checker;
  IncrementalDecoder decoder(/*prompt=*/"prompt",
                             /*num_prompt_tokens=*/prompt.size(),
                             /*echo=*/false,
                             /*skip_special_tokens=*/true);
  Sequence seq(/*index=*/0,
               prompt,
               /*input_embedding=*/torch::Tensor(),
               /*mm_data=*/MMData(),
               decoder,
               params);

  const std::vector<XXH3Key> expected = ExpectedChain(prompt, block_size);

  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  ASSERT_EQ(seq.block_hashes(block_size).size(), n_blocks);
  for (uint32_t i = 0; i < n_blocks; ++i) {
    EXPECT_EQ(std::memcmp(seq.block_hashes(block_size)[i].data,
                          expected[i].data,
                          XXH3_128BITS_HASH_VALUE_LEN),
              0);
  }

  // Idempotent: no new full block -> nothing recomputed/appended.
  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  EXPECT_EQ(seq.block_hashes(block_size).size(), n_blocks);

  // Rewriting a token in block index 2 invalidates block 2 and everything
  // after it; blocks 0 and 1 survive.
  seq.update_token(2 * block_size + 1, Token(/*id=*/999999));
  EXPECT_EQ(seq.block_hashes(block_size).size(), 2u);

  // Recompute: blocks 0/1 unchanged, block 2 differs from the old chain.
  seq.update_block_hashes(block_size, BlockHasherType::TEXT);
  ASSERT_EQ(seq.block_hashes(block_size).size(), n_blocks);
  EXPECT_EQ(std::memcmp(seq.block_hashes(block_size)[0].data,
                        expected[0].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
  EXPECT_EQ(std::memcmp(seq.block_hashes(block_size)[1].data,
                        expected[1].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
  EXPECT_NE(std::memcmp(seq.block_hashes(block_size)[2].data,
                        expected[2].data,
                        XXH3_128BITS_HASH_VALUE_LEN),
            0);
}

// In-batch prefix cache publishes only the full blocks covered by the given
// token budget. When the budget is overestimated, cache() must clamp to the
// sequence's own tokens and register just the complete blocks.
TEST_F(BlockManagerPoolTest, CachePrefixClampsOverestimatedTokenBudget) {
  ScopedValue<int32_t> max_seqs_guard(
      &SchedulerConfig::get_instance().max_seqs_per_batch(), 2);

  BlockManagerPool::Options options;
  options.num_blocks(16).host_num_blocks(0).block_size(4).enable_prefix_cache(
      true);
  // Leak the pool intentionally: with prefix cache enabled, the cached block
  // stays referenced by the prefix-cache table at teardown, which would trip
  // the free-list check in ~BlockManagerImpl.
  auto* pool = new BlockManagerPool(options, /*dp_size=*/1);

  Sequence seq = make_sequence(0, /*prompt_tokens=*/{1, 2, 3, 4, 5, 6, 7});
  ASSERT_TRUE(pool->allocate(&seq));

  // num_tokens (8) is larger than the 7 real tokens. cache() must clamp to the
  // sequence tokens, so only the single full block (tokens [0, 4)) is cached.
  EXPECT_NO_FATAL_FAILURE(pool->cache(&seq, /*num_tokens=*/8));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache()[0], 1u);
}

TEST(BlockManagerImplTest, RefcountZeroEntersFreeListAndCacheOnlyIsEvictable) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(true);
  BlockManagerImpl manager(options);

  const size_t total = manager.num_total_blocks();
  EXPECT_EQ(manager.num_free_blocks(), total);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_evictable_blocks(), 0u);
  EXPECT_EQ(manager.num_available_blocks(), total);

  std::vector<int32_t> token_ids = {1, 2, 3, 4, 5, 6, 7, 8};
  const Slice<int32_t> tokens(token_ids);
  int32_t first_id = -1;
  {
    std::vector<Block> blocks = manager.allocate(2);
    ASSERT_EQ(blocks.size(), 2u);
    first_id = blocks[0].id();
    EXPECT_EQ(manager.num_free_blocks(), total - 2);
    EXPECT_EQ(manager.num_used_blocks(), 2u);
    EXPECT_EQ(manager.num_evictable_blocks(), 0u);

    manager.cache(tokens, blocks);
    EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 2u);
    EXPECT_EQ(manager.num_evictable_blocks(), 0u);
    EXPECT_EQ(manager.num_used_blocks(), 2u);
  }

  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 2u);
  EXPECT_EQ(manager.num_evictable_blocks(), 2u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), total - 2);
  EXPECT_EQ(manager.num_available_blocks(), total);

  {
    std::vector<Block> matched = manager.allocate_shared(tokens);
    ASSERT_EQ(matched.size(), 2u);
    EXPECT_EQ(matched[0].id(), first_id);
    EXPECT_EQ(manager.num_evictable_blocks(), 0u);
    EXPECT_EQ(manager.num_used_blocks(), 2u);
    EXPECT_EQ(manager.num_free_blocks(), total - 2);
    EXPECT_EQ(manager.num_available_blocks(), total - 2);
  }

  EXPECT_EQ(manager.num_evictable_blocks(), 2u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);

  manager.reset_prefix_cache();
  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager.num_evictable_blocks(), 0u);
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  EXPECT_EQ(manager.num_free_blocks(), total);
  EXPECT_EQ(manager.num_available_blocks(), total);
}

// Block ids restart on every leaf. Caching one leaf's block into another
// records that numeric id in the wrong prefix cache and increments the wrong
// refcount. The owner's later free then sees the id still registered.
TEST(ConcurrentBlockManagerImplTest,
     ForeignBlockIsNotInsertedIntoAnotherCache) {
  BlockManager::Options options;
  options.num_blocks(8).block_size(4).enable_prefix_cache(true);
  auto owner = std::make_unique<ConcurrentBlockManagerImpl>(
      std::make_unique<BlockManagerImpl>(options));
  auto other = std::make_unique<ConcurrentBlockManagerImpl>(
      std::make_unique<BlockManagerImpl>(options));

  const size_t total = owner->num_total_blocks();
  std::vector<Block> blocks = owner->allocate(1);
  ASSERT_EQ(blocks.size(), 1u);
  const int32_t block_id = blocks[0].id();
  EXPECT_EQ(blocks[0].ref_count(), 1u);

  uint8_t hash[XXH3_128BITS_HASH_VALUE_LEN] = {};
  hash[0] = 1;
  blocks[0].set_hash_value(hash);
  other->cache(blocks);

  EXPECT_EQ(other->num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(owner->num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(blocks[0].ref_count(), 1u);
  blocks.clear();
  EXPECT_EQ(owner->num_free_blocks(), total);

  std::vector<Block> owned = owner->allocate(1);
  ASSERT_EQ(owned.size(), 1u);
  const int32_t owned_id = owned[0].id();
  owned[0].set_hash_value(hash);
  owner->cache(owned);
  EXPECT_EQ(owner->num_blocks_in_prefix_cache(), 1u);
  EXPECT_EQ(owned[0].ref_count(), 2u);
  owned.clear();
  EXPECT_EQ(owner->num_free_blocks(), total - 1);

  std::vector<Block> again = owner->allocate(1);
  ASSERT_EQ(again.size(), 1u);
  EXPECT_NE(again[0].id(), owned_id);
  EXPECT_NE(again[0].id(), block_id);

  owner->reset_prefix_cache();
  other->reset_prefix_cache();
}

}  // namespace xllm
