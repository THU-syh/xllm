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

#include "block_manager_impl.h"

#include <typeinfo>
#include <unordered_set>

#include "framework/prefix_cache/prefix_cache_factory.h"
namespace xllm {
namespace {

// 1 means allocate() took this id off the free list. clear_used runs only when
// refcount hits 0, so a prefix-cache resident stays marked. allocate_shared
// aliases that id and must not mark it again.
bool mark_used(std::vector<uint8_t>* usage_ids, int32_t block_id) {
  CHECK(usage_ids != nullptr);
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids->size());
  if ((*usage_ids)[block_id] != 0) {
    return false;
  }
  (*usage_ids)[block_id] = 1;
  return true;
}

bool clear_used(std::vector<uint8_t>* usage_ids, int32_t block_id) {
  CHECK(usage_ids != nullptr);
  CHECK_GE(block_id, 0);
  CHECK_LT(static_cast<size_t>(block_id), usage_ids->size());
  if ((*usage_ids)[block_id] == 0) {
    return false;
  }
  (*usage_ids)[block_id] = 0;
  return true;
}

}  // namespace

BlockManagerImpl::BlockManagerImpl(const Options& options)
    : BlockManager(options) {
  CHECK_GT(options.num_blocks(), 0) << "No blocks to allocate";
  CHECK_GT(options.block_size(), 0) << "Block size must be positive";
  if (options_.enable_prefix_cache()) {
    PrefixCache::Options prefix_cache_options;
    prefix_cache_options.block_size(options.block_size())
        .hasher_type(options.hasher_type())
        .block_type(options.block_type());
    prefix_cache_ = create_prefix_cache(prefix_cache_options);
    CHECK(prefix_cache_) << "Failed to create prefix cache!";
  }

  size_t total_blocks = options_.num_blocks();
  block_size_ = options_.block_size();
  num_free_blocks_.store(total_blocks, std::memory_order_relaxed);
  usage_accounted_ids_.assign(total_blocks, 0);
  free_blocks_.reserve(total_blocks);
  for (int32_t i = 0; i < total_blocks; ++i) {
    // push smaller block ids to the back of the vector
    free_blocks_.push_back(total_blocks - i - 1);
  }

  // reserve block 0 for padding. It stays off the free list for the manager's
  // lifetime, and free() rejects id 0, so this block must not call free.
  // Padding is taken before a wrapper is attached. The qualified call stays
  // on this free list and does not enter block_allocator_.
  std::vector<Block> padding = BlockManagerImpl::allocate(1);
  CHECK_EQ(padding.size(), 1u);
  padding_block_ = std::move(padding[0]);
  CHECK_EQ(padding_block_.id(), 0) << "Padding block id should be 0";
  padding_block_.set_manager(nullptr);
}

std::vector<Block> BlockManagerImpl::allocate(size_t num_blocks) {
  if (!has_enough_blocks(num_blocks)) {
    return {};
  }

  CHECK(num_blocks <= num_free_blocks_) << "Not enough blocks available";
  std::vector<Block> blocks;
  blocks.reserve(num_blocks);
  for (uint32_t i = 0; i < num_blocks; ++i) {
    size_t prev_count =
        num_free_blocks_.fetch_sub(1, std::memory_order_relaxed);
    const int32_t block_id = free_blocks_[prev_count - 1];
    CHECK(mark_used(&usage_accounted_ids_, block_id))
        << "block " << block_id << " usage accounted repeatedly";
    blocks.emplace_back(block_id, this);
  }

  return blocks;
}

size_t BlockManagerImpl::num_used_blocks() const {
  const size_t total = num_total_blocks();
  const size_t not_held = num_free_blocks() + num_evictable_blocks();
  return total > not_held ? total - not_held : 0;
}

size_t BlockManagerImpl::num_evictable_blocks() const {
  if (prefix_cache_ == nullptr) {
    return 0;
  }
  return prefix_cache_->num_evictable();
}

double BlockManagerImpl::kv_cache_utilization() const {
  const size_t total = num_total_blocks();
  if (total == 0) {
    return 0.0;
  }
  return static_cast<double>(num_used_blocks()) / static_cast<double>(total);
}

bool BlockManagerImpl::has_enough_blocks(uint32_t num_blocks) {
  if (num_blocks <= num_free_blocks_) {
    return true;
  }

  // prefix cache is disabled, no way to evict blocks
  if (!options_.enable_prefix_cache()) {
    return false;
  }

  // try to evict some blocks from the prefix cache
  const uint32_t n_blocks_to_evict = num_blocks - num_free_blocks_;

  AUTO_COUNTER(prefix_cache_latency_seconds_evict);
  const uint32_t n_blocks_evicted = prefix_cache_->evict(n_blocks_to_evict);
  if (n_blocks_evicted < n_blocks_to_evict) {
    return false;
  }

  if (num_free_blocks_ >= num_blocks) {
    return true;
  }

  LOG(WARNING) << "Potential block leak, free blocks in allocator: "
               << num_free_blocks_
               << " blocks in prefix cache: " << prefix_cache_->num_blocks();
  return false;
}

std::vector<Block> BlockManagerImpl::allocate_shared(
    const Slice<int32_t>& token_ids,
    const Slice<Block>& existed_shared_blocks,
    const MMData& mm_data,
    const Slice<XXH3Key>& block_hashes) {
  // KV, SWA and LINEAR all probe through this path. SWA and LINEAR use a
  // gap-tolerant cache, so a partial shared prefix is not a resume point.
  // LINEAR also drops the last token: that token is the live write, not a
  // finished checkpoint.
  if (!options_.enable_prefix_cache() || prefix_cache_ == nullptr ||
      block_size_ == 0) {
    return {};
  }
  if (block_type() == BlockType::LINEAR && token_ids.empty()) {
    return {};
  }

  const bool linear = block_type() == BlockType::LINEAR;
  const bool ignore_existed = linear || block_type() == BlockType::SWA;
  const Slice<int32_t> match_tokens =
      linear ? token_ids.slice(0, token_ids.size() - 1) : token_ids;
  const Slice<Block> empty_blocks;
  const Slice<Block>& existed =
      ignore_existed ? empty_blocks : existed_shared_blocks;

  AUTO_COUNTER(prefix_cache_latency_seconds_match);
  std::vector<Block> shared_blocks =
      prefix_cache_->match(match_tokens, existed, mm_data, block_hashes);

  VLOG(1) << "Prefix cache matched manager=" << typeid(*this).name()
          << " this=" << this << " decode=" << options_.instance_is_decode()
          << " blocks=" << shared_blocks.size()
          << " prefix_length=" << shared_blocks.size() * block_size_;

  return shared_blocks;
}

void BlockManagerImpl::cache(const Slice<int32_t>& token_ids,
                             std::vector<Block>& blocks,
                             size_t existed_shared_blocks_num,
                             const MMData& mm_data,
                             const Slice<XXH3Key>& block_hashes) {
  if (options_.enable_prefix_cache()) {
    check_owned(blocks);
    AUTO_COUNTER(prefix_cache_latency_seconds_insert);
    // Add the kv cache to the prefix cache
    prefix_cache_->insert(
        token_ids, blocks, existed_shared_blocks_num, mm_data, block_hashes);
  }
}

void BlockManagerImpl::check_owned(const std::vector<Block>& blocks) const {
  for (const Block& block : blocks) {
    if (!block.is_valid()) {
      continue;
    }
    const BlockManager* manager = block.manager();
    // Block ids are per leaf. Another leaf's block would record that id in
    // this prefix cache and increment the other leaf's refcount.
    CHECK(manager == this || manager == block_allocator_)
        << "Prefix cache insert of block " << block.id()
        << " manager=" << manager << " leaf=" << this;
  }
}

void BlockManagerImpl::cache(const std::vector<Block>& blocks) {
  if (!options_.enable_prefix_cache()) {
    return;
  }
  check_owned(blocks);
  AUTO_COUNTER(prefix_cache_latency_seconds_insert);
  prefix_cache_->insert(blocks);
}

void BlockManagerImpl::incr_ref_count(std::atomic<uint32_t>* ref_count) {
  CHECK(ref_count != nullptr);
  ref_count->fetch_add(1, std::memory_order_relaxed);
}

uint32_t BlockManagerImpl::dec_ref_count(std::atomic<uint32_t>* ref_count) {
  CHECK(ref_count != nullptr);
  return ref_count->fetch_sub(1, std::memory_order_acq_rel);
}

// caller should make sure the block_id is valid
void BlockManagerImpl::free(Block* block) {
  CHECK(block != nullptr && block->id() != 0);
  release_block_ref(block);

  if (prefix_cache_ != nullptr) {
    const bool in_prefix_cache = prefix_cache_->update_evictable(block);
    // A prefix-cache node shares this block's refcount. After the last
    // external alias drops, the cache copy must still hold one. Refcount 0
    // while the id is registered means the node did not own the counter.
    CHECK(!in_prefix_cache || block->ref_count() != 0)
        << "Block " << block->id() << " is in the prefix cache with refcount 0";
  }
  if (block->ref_count() == 0) {
    const int32_t block_id = block->id();
    CHECK(clear_used(&usage_accounted_ids_, block_id))
        << "Block " << block_id << " freed twice";
    size_t prev_count =
        num_free_blocks_.fetch_add(1, std::memory_order_relaxed);
    CHECK(prev_count < free_blocks_.size());
    free_blocks_[prev_count] = block_id;
  }
}

// Flat incremental growth (the default sequence-level policy). Reads how many
// blocks the sequence already holds for this manager's block_type(), allocates
// the delta to cover num_tokens, and returns it (does not insert into the
// sequence -- the CompositeBlockManager commits the returned blocks). Returns
// std::nullopt on post-eviction shortage. SlidingWindow / Single override.
std::optional<std::vector<Block>> BlockManagerImpl::allocate_for_sequence(
    Sequence* seq,
    KVCacheState& kv_state,
    size_t num_tokens) {
  if (seq == nullptr) {
    return std::nullopt;
  }
  if (block_size_ == 0) {
    return std::vector<Block>{};
  }
  const size_t held = kv_state.num_blocks(block_type());
  const size_t num_blocks_needed = (num_tokens + block_size_ - 1) / block_size_;
  if (num_blocks_needed <= held) {
    return std::vector<Block>{};
  }
  const size_t num_additional = num_blocks_needed - held;
  std::vector<Block> blocks = block_allocator()->allocate(num_additional);
  if (blocks.size() != num_additional) {
    return std::nullopt;
  }
  return blocks;
}

bool BlockManagerImpl::allocate_for_prefetch(Sequence* seq, size_t num_tokens) {
  if (seq == nullptr || block_size_ == 0) {
    return false;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t target_blocks = num_tokens / block_size_;
  const size_t cached_blocks = host_state.kv_cache_tokens_num() / block_size_;
  // prefetch_from_storage already returns when the host hit covers the target.
  CHECK_LT(cached_blocks, target_blocks);
  // The existing vector is the matched prefix. Extend only past its length.
  std::vector<Block> blocks = host_state.take_blocks(block_type());
  CHECK_LE(blocks.size(), cached_blocks);
  blocks.resize(target_blocks);

  seq->update_block_hashes(static_cast<uint32_t>(block_size_),
                           options_.hasher_type());
  const Slice<XXH3Key> hashes =
      seq->block_hashes(static_cast<uint32_t>(block_size_));
  CHECK_GE(hashes.size(), target_blocks);

  const size_t needed = target_blocks - cached_blocks;
  std::vector<Block> allocated = block_allocator()->allocate(needed);
  if (allocated.empty()) {
    allocated =
        block_allocator()->allocate(std::min(needed, num_free_blocks()));
  }

  for (size_t index = 0; index < allocated.size(); ++index) {
    const size_t block_index = cached_blocks + index;
    allocated[index].set_hash_value(hashes[block_index].data);
    blocks[block_index] = std::move(allocated[index]);
  }

  if (!blocks.empty()) {
    host_state.replace_composite_blocks(
        block_type(), std::move(blocks), cached_blocks, cached_blocks);
  }
  return allocated.size() == needed;
}

void BlockManagerImpl::trim_prefetch_blocks(Sequence* seq,
                                            size_t max_hit_tokens) {
  if (seq == nullptr || block_size_ == 0) {
    return;
  }

  KVCacheState& host_state = seq->host_kv_state();
  const size_t keep = std::min(max_hit_tokens / block_size_,
                               host_state.num_blocks(block_type()));
  std::vector<Block> blocks = host_state.take_blocks(block_type());
  if (blocks.size() > keep) {
    blocks.resize(keep);
  }

  if (blocks.empty()) {
    host_state.erase_blocks(block_type());
    return;
  }

  // Store prefetch owns newly materialized Host blocks until this point. Once
  // the gate boundary is known, publish exactly the retained range and make
  // both state cursors describe that published logical range.
  seq->update_block_hashes(static_cast<uint32_t>(block_size_),
                           options_.hasher_type());
  cache(seq->hash_tokens(options_.hasher_type()),
        blocks,
        /*existed_shared_blocks_num=*/0,
        seq->mm_data(),
        seq->block_hashes(static_cast<uint32_t>(block_size_)));
  host_state.replace_composite_blocks(
      block_type(), std::move(blocks), keep, keep);
}

}  // namespace xllm
