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

#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "block_manager_impl.h"
#include "composite_block_manager.h"
#include "concurrent_block_manager_impl.h"
#include "framework/prefix_cache/prefix_cache.h"

namespace xllm {
namespace {

void release_prefix_cache(ConcurrentBlockManagerImpl* manager) {
  CHECK(manager != nullptr);
  std::vector<Block> blocks = manager->allocate(manager->num_total_blocks());
  EXPECT_EQ(blocks.size(), manager->num_total_blocks());
  blocks.clear();
}

}  // namespace

TEST(ConcurrentBlockManagerTest,
     SharedPrefixStaysUsedUntilLastSequenceRelease) {
  BlockManager::Options options;
  options.num_blocks(3).block_size(2).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));
  const std::vector<int32_t> token_ids = {11, 12};
  std::vector<Block> first = manager.allocate(/*num_blocks=*/1);
  ASSERT_EQ(first.size(), 1u);
  PrefixCache::compute_hash_keys(token_ids, first);
  manager.cache(first);
  std::vector<Block> second = manager.allocate_shared(token_ids);
  ASSERT_EQ(second.size(), 1u);
  EXPECT_EQ(first.front().id(), second.front().id());
  ASSERT_EQ(second.front().ref_count(), 3u);
  EXPECT_EQ(manager.num_used_blocks(), 1u);

  first.clear();
  EXPECT_TRUE(first.empty());
  EXPECT_EQ(second.front().ref_count(), 2u);
  EXPECT_EQ(manager.num_used_blocks(), 1u);
  EXPECT_DOUBLE_EQ(manager.kv_cache_utilization(), 0.5);
  EXPECT_TRUE(manager.allocate(manager.num_total_blocks()).empty());

  second.clear();
  EXPECT_TRUE(second.empty());
  EXPECT_EQ(manager.num_used_blocks(), 0u);
  release_prefix_cache(&manager);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
}

TEST(ConcurrentBlockManagerTest, ContinuesPrefixCacheFromExistingBlocks) {
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(5).block_size(block_size).enable_prefix_cache(true);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));

  std::vector<int32_t> token_ids = {11, 12, 13, 14};
  std::vector<Block> seed_blocks = manager.allocate(/*num_blocks=*/2);
  ASSERT_EQ(seed_blocks.size(), 2);
  PrefixCache::compute_hash_keys(token_ids, seed_blocks);

  const int32_t first_block_id = seed_blocks[0].id();
  const int32_t second_block_id = seed_blocks[1].id();

  std::vector<Block> existed_blocks;
  existed_blocks.reserve(1);
  existed_blocks.emplace_back(std::move(seed_blocks[0]));

  std::vector<Block> tail_blocks;
  tail_blocks.reserve(1);
  tail_blocks.emplace_back(std::move(seed_blocks[1]));
  manager.cache(tail_blocks);
  tail_blocks.clear();
  tail_blocks.clear();
  seed_blocks.clear();

  ASSERT_EQ(manager.num_blocks_in_prefix_cache(), 1);

  std::vector<Block> matched_blocks =
      manager.allocate_shared(token_ids, existed_blocks);

  EXPECT_EQ(matched_blocks.size(), 2);
  if (matched_blocks.size() == 2) {
    EXPECT_EQ(matched_blocks[0].id(), first_block_id);
    EXPECT_EQ(matched_blocks[1].id(), second_block_id);
  }

  matched_blocks.clear();
  existed_blocks.clear();
  matched_blocks.clear();
  existed_blocks.clear();
  release_prefix_cache(&manager);

  EXPECT_EQ(manager.num_blocks_in_prefix_cache(), 0);
  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
}

TEST(ConcurrentBlockManagerTest, AllocatesWhileBlocksReleaseConcurrently) {
  BlockManager::Options options;
  options.num_blocks(65).block_size(2).enable_prefix_cache(false);
  ConcurrentBlockManagerImpl manager(
      std::make_unique<BlockManagerImpl>(options));

  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 10000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));

  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([&manager, &start, kNumIterations]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        std::vector<Block> blocks = manager.allocate(/*num_blocks=*/1);
        if (blocks.empty()) {
          std::this_thread::yield();
          continue;
        }

        blocks.clear();
        blocks.clear();
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  EXPECT_EQ(manager.num_free_blocks(), manager.num_total_blocks());
  EXPECT_EQ(manager.num_used_blocks(), 0);
}

class LinearConcurrentAllocationTest
    : public ::testing::TestWithParam<int32_t> {};

TEST_P(LinearConcurrentAllocationTest, KeepsLiveSlotsUniqueUntilFinalRelease) {
  constexpr int32_t kNumSlots = 64;
  constexpr int32_t kNumThreads = 8;
  constexpr int32_t kNumIterations = 10000;
  BlockManager::Options options;
  options.num_blocks(128)
      .block_size(128)
      .enable_linear_state(true)
      .linear_state_num_slots(kNumSlots)
      .enable_prefix_cache(false)
      .instance_is_decode(true)
      .enable_disagg_pd(GetParam() == 0)
      .enable_kvcache_store(GetParam() == 1)
      .enable_host_offload(GetParam() == 2);
  auto leaves = build_composite_leaves(options);
  BlockManager* manager = leaves.at(BlockType::LINEAR).leaf.get();
  std::vector<std::atomic<int32_t>> live_ids(kNumSlots + 1);
  for (auto& live : live_ids) {
    live.store(0, std::memory_order_relaxed);
  }
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(kNumThreads);
  for (int32_t thread_index = 0; thread_index < kNumThreads; ++thread_index) {
    workers.emplace_back([&]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int32_t iteration = 0; iteration < kNumIterations; ++iteration) {
        std::vector<Block> blocks = manager->allocate(1);
        if (blocks.empty()) {
          continue;
        }
        const int32_t block_id = blocks.front().id();
        ASSERT_GT(block_id, 0);
        ASSERT_LE(block_id, kNumSlots);
        EXPECT_EQ(live_ids[block_id].fetch_add(1), 0);
        Block inflight = blocks.front();
        blocks.clear();
        blocks.clear();
        EXPECT_EQ(live_ids[block_id].fetch_sub(1), 1);
        inflight = Block();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks());
  EXPECT_EQ(manager->num_used_blocks(), 0u);
}

INSTANTIATE_TEST_SUITE_P(ConcurrentModes,
                         LinearConcurrentAllocationTest,
                         ::testing::Values(0, 1, 2));

// End-to-end repro of the production signature through the real
// ConcurrentBlockManagerImpl + PrefixCache: blocks that live only in the prefix
// cache get pinned as is_shared() and can no longer be reclaimed.
//
// Each thread emulates a prefill request in steady state, faithfully mirroring
// CompositeBlockManager: it matches the cached prefix (scheduler-side
// allocate_shared, inc under lock; the matched blocks are the sequence's single
// alias set -- add_shared_blocks() MOVES them into KVCacheState), deallocates
// exactly that alias set on finish
// (CompositeBlockManager::deallocate_for_sequence calls
// leaf->deallocate(kv_state.blocks(...))), then destroys the alias OUTSIDE the
// manager lock (BlockManagerPool::deallocate -> sequence->reset()). The racing
// dec on a cached block's ref_count is the root-cause window: a lost decrement
// leaves the block is_shared() with no logical owner.
//
// On the buggy (non-atomic) code a lost decrement pins a cached block
// is_shared() forever, so evict() skips it permanently -- allocate() can no
// longer reclaim it and the pool starves (num_free < total while num_used ==
// 0), exactly the production deadlock. Reclaiming every block below (allocate
// the full pool, which forces eviction) is therefore the deadlock detector: it
// cannot obtain `total` blocks while any block is pinned. It passes once
// ref_count_ is atomic.
TEST(ConcurrentBlockManagerTest, PrefixCacheSharedBlockLifecycleLeak) {
  const uint32_t block_size = 2;
  BlockManager::Options options;
  options.num_blocks(64).block_size(block_size).enable_prefix_cache(true);
  // Heap-allocated and intentionally leaked: on the buggy code blocks stay
  // pinned, so ~BlockManagerImpl()'s "all blocks freed" CHECK would abort and
  // mask the assertions below. Leaking the manager keeps the gtest signal
  // clean.
  auto* manager = new ConcurrentBlockManagerImpl(
      std::make_unique<BlockManagerImpl>(options));

  // Seed a 4-block shared prefix, then drop the seeding aliases so the blocks
  // live ONLY in the prefix cache (ref_count == 1 each, evictable) -- the
  // steady state a finished prefill request leaves behind.
  std::vector<int32_t> token_ids = {11, 12, 13, 14, 15, 16, 17, 18};
  std::vector<Block> seed = manager->allocate(/*num_blocks=*/4);
  ASSERT_EQ(seed.size(), 4u);
  PrefixCache::compute_hash_keys(token_ids, seed);
  manager->cache(seed);
  seed.clear();
  seed.clear();
  ASSERT_EQ(manager->num_blocks_in_prefix_cache(), 4u);

  constexpr int32_t kNumThreads = 4;
  constexpr int32_t kNumIterations = 50000;
  std::atomic<bool> start{false};
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(kNumThreads));
  for (int32_t i = 0; i < kNumThreads; ++i) {
    workers.emplace_back([manager, &token_ids, &start]() {
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (int32_t iter = 0; iter < kNumIterations; ++iter) {
        // Scheduler-side match: copies the cached blocks (inc ref, under lock).
        // These ARE the sequence's alias set (production MOVEs them into
        // KVCacheState), so ref_count == cache(1) + sequence(1) == 2.
        std::vector<Block> seq_blocks = manager->allocate_shared(token_ids);
        if (seq_blocks.empty()) {
          continue;
        }
        // Finish: deallocate exactly the sequence's blocks (under lock) ...
        seq_blocks.clear();
        // ... then sequence->reset() drops the alias OUTSIDE the manager lock
        // -- the dec that races other threads' matches on the same cached
        // block.
        seq_blocks.clear();
      }
    });
  }
  start.store(true, std::memory_order_release);
  for (std::thread& worker : workers) {
    worker.join();
  }

  // No sequence holds anything now. A correct implementation leaves every
  // cached block evictable (ref_count == 1); a lost decrement pins one
  // is_shared(). Allocating the whole pool forces eviction of the cache and
  // only succeeds if nothing is pinned -- this is the deadlock probe. (We do
  // NOT assert num_used before this point: the deallocate-side ref_count<=2
  // usage gate is intentionally approximate and can transiently over-count
  // while peer threads sit in their own deallocate->reset window; free()
  // reconciles it on eviction, so num_used is exact only once the cache has
  // been fully drained below.)
  release_prefix_cache(manager);
  EXPECT_EQ(manager->num_blocks_in_prefix_cache(), 0u);
  EXPECT_EQ(manager->num_free_blocks(), manager->num_total_blocks())
      << "leaked blocks: " << manager->num_blocks_in_prefix_cache()
      << " still pinned in prefix cache; num_free="
      << manager->num_free_blocks()
      << " num_used=" << manager->num_used_blocks()
      << " total=" << manager->num_total_blocks();
  EXPECT_EQ(manager->num_used_blocks(), 0u);
}

}  // namespace xllm
