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

#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>

#include "block_manager.h"

namespace xllm {

// Concurrency adapter: wraps an inner leaf BlockManager and serializes every
// entry point under one recursive mutex. Used for modes whose sequence-level
// calls run off the scheduler thread (disagg PD / kvcache store prefill
// threadpools).
//
// Composition, not inheritance: the inner manager is always a leaf
// (BlockManagerImpl / EmbeddingBlockManager / ...), never a
// CompositeBlockManager. The composite constructs this wrapper around the
// leaves that need it.
class ConcurrentBlockManagerImpl : public BlockManager {
 public:
  explicit ConcurrentBlockManagerImpl(std::unique_ptr<BlockManager> inner);
  ~ConcurrentBlockManagerImpl() override = default;

  // Block-level primitives.
  std::vector<Block> allocate(size_t num_blocks) override;
  void incr_ref_count(std::atomic<uint32_t>* ref_count) override;
  uint32_t dec_ref_count(std::atomic<uint32_t>* ref_count) override;
  void free(Block* block) override;
  size_t num_evictable_blocks() const override;

  std::vector<Block> allocate_shared(
      const Slice<int32_t>& token_ids,
      const Slice<Block>& existed_shared_blocks = {},
      const MMData& mm_data = MMData(),
      const Slice<XXH3Key>& block_hashes = {}) override;

  void cache(const Slice<int32_t>& token_ids,
             std::vector<Block>& blocks,
             size_t existed_shared_blocks_num = 0,
             const MMData& mm_data = MMData(),
             const Slice<XXH3Key>& block_hashes = {}) override;
  void cache(const std::vector<Block>& blocks) override;

  // Sequence-level growth forwards under the lock (touches the inner leaf's
  // free list / pool).
  std::optional<std::vector<Block>> allocate_for_sequence(
      Sequence* seq,
      KVCacheState& kv_state,
      size_t num_tokens) override;
  bool allocate_for_prefetch(Sequence* seq, size_t num_tokens) override;
  void trim_prefetch_blocks(Sequence* seq, size_t max_hit_tokens) override;

  void reset_prefix_cache() override;

  size_t num_blocks_in_prefix_cache() const override;
  size_t num_free_blocks() const override;
  size_t num_used_blocks() const override;
  size_t num_total_blocks() const override;
  double kv_cache_utilization() const override;

 private:
  std::unique_ptr<BlockManager> inner_;  // always a leaf, never a composite
  mutable std::recursive_mutex mutex_;
};

}  // namespace xllm
