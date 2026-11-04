/* Copyright 2025-2026 The xLLM Authors.

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

#include <torch/library.h>

#include <string>
#include <tuple>
#include <vector>

#include "core/kernels/npu/aclnn/pytorch_npu_helper.hpp"
#include "xllm_ops_api.h"

namespace xllm::kernel::npu {

namespace {

constexpr int64_t kDim0 = 0;
constexpr int64_t kDim1 = 1;

// True when every axis except axis 0 has a densely packed stride.  The fused
// pool key cache is allowed to be non-contiguous only on its page axis
// (PA_BBND), with the page stride passed to the kernel as ``key_stride0``.
bool is_contiguous_except_axis0(const torch::Tensor& tensor) {
  const auto sizes = tensor.sizes();
  const auto strides = tensor.strides();
  const int64_t ndim = sizes.size();
  if (ndim <= 1) {
    return true;
  }
  int64_t contiguous_stride = 1;
  for (int64_t i = ndim - 1; i >= 1; --i) {
    if (strides[i] != contiguous_stride) {
      return false;
    }
    contiguous_stride *= sizes[i];
  }
  return true;
}

std::tuple<torch::Tensor, torch::Tensor> construct_output(
    const torch::Tensor& query,
    int64_t topk,
    int64_t pool_size,
    const std::string& layout_q,
    bool return_value) {
  TORCH_CHECK(topk > 0, "topk must be greater than 0, got ", topk);
  TORCH_CHECK(
      pool_size > 0, "pool_size must be greater than 0, got ", pool_size);
  TORCH_CHECK(topk % pool_size == 0,
              "topk(",
              topk,
              ") must be divisible by pool_size(",
              pool_size,
              ")");

  const int64_t indices_len = topk + pool_size - 1;
  const int64_t values_len = topk / pool_size;
  std::vector<int64_t> indices_shape;
  std::vector<int64_t> values_shape;
  if (layout_q == "BSND") {
    indices_shape = {query.size(kDim0), query.size(kDim1), indices_len};
    values_shape = {query.size(kDim0), query.size(kDim1), values_len};
  } else {
    indices_shape = {query.size(kDim0), indices_len};
    values_shape = {query.size(kDim0), values_len};
  }

  torch::Tensor indices =
      torch::empty(indices_shape, query.options().dtype(torch::kInt));
  torch::Tensor values =
      return_value
          ? torch::empty(values_shape, query.options().dtype(torch::kFloat))
          : torch::empty({0}, query.options().dtype(torch::kFloat));
  return std::make_tuple(indices, values);
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> pool_key_indexer(
    const torch::Tensor& query,
    const torch::Tensor& pool_key,
    const torch::Tensor& weights,
    const torch::Tensor& pool_tail_k,
    int64_t topk,
    int64_t pool_size,
    bool return_value,
    const c10::optional<torch::Tensor>& q_descale,
    const c10::optional<torch::Tensor>& k_descale,
    const c10::optional<torch::Tensor>& actual_seq_q,
    const c10::optional<torch::Tensor>& actual_seq_k,
    const c10::optional<torch::Tensor>& block_table,
    c10::string_view layout_q,
    c10::string_view layout_k,
    int64_t mask_mode,
    int64_t quant_mode,
    int64_t sparse_indices_mode) {
  TORCH_CHECK(query.numel() > 0, "Tensor query is empty.");
  TORCH_CHECK(weights.numel() > 0, "Tensor weights is empty.");
  TORCH_CHECK(pool_tail_k.numel() > 0, "Tensor pool_tail_k is empty.");
  TORCH_CHECK(sparse_indices_mode == 0 || sparse_indices_mode == 1,
              "sparse_indices_mode must be 0 (legacy) or 1 (compact), got ",
              sparse_indices_mode);
  // The Python wrapper only exposes the non-quantized kPool path today.  Keep
  // the check explicit rather than silently misinterpreting descale tensors.
  TORCH_CHECK(
      quant_mode == -1,
      "quantized pool_key_indexer is not supported by the xLLM wrapper, "
      "got quant_mode=",
      quant_mode);
  TORCH_CHECK(!(q_descale.has_value() && q_descale->defined()) &&
                  !(k_descale.has_value() && k_descale->defined()),
              "quant_mode=-1 requires q_descale and k_descale to be null");

  const std::string layout_q_str(layout_q);
  const std::string layout_k_str(layout_k);

  // pool_key may be an empty "zero-pool" tensor: all visible tokens are then
  // carried by pool_tail_k.  Any other empty shape is rejected.
  if (pool_key.numel() == 0) {
    const bool zero_pool = (layout_k_str == "BSND" && pool_key.dim() == 4 &&
                            pool_key.size(1) == 0) ||
                           (layout_k_str == "TND" && pool_key.dim() == 3 &&
                            pool_key.size(0) == 0) ||
                           (layout_k_str == "PA_BBND" && pool_key.dim() == 4 &&
                            pool_key.size(0) == 0);
    TORCH_CHECK(zero_pool,
                "pool_key is empty; only zero-pool encodings are allowed, got "
                "shape ",
                pool_key.sizes());
  }

  auto [sparse_indices, sparse_values] =
      construct_output(query, topk, pool_size, layout_q_str, return_value);

  // A PA_BBND pool cache may be non-contiguous on the page axis; forward its
  // stride(0) so tiling does not assume a densely packed cache.
  int64_t key_stride0 = -1;
  if (!pool_key.is_contiguous()) {
    TORCH_CHECK(layout_k_str == "PA_BBND",
                "pool_key non-contiguous tensor is only supported with "
                "layout_k='PA_BBND', got '",
                layout_k_str,
                "'");
    TORCH_CHECK(is_contiguous_except_axis0(pool_key),
                "pool_key only supports a non-contiguous axis 0; all other "
                "axes must be densely packed");
    key_stride0 = pool_key.stride(0);
    const int64_t contiguous_stride0 =
        pool_key.size(1) * pool_key.size(2) * pool_key.size(3);
    TORCH_CHECK(key_stride0 >= contiguous_stride0,
                "pool_key stride0(",
                key_stride0,
                ") must be >= contiguous stride(",
                contiguous_stride0,
                ") in PA_BBND layout");
  }

  // The Tensor value-input variants require int64 tensors on the compute
  // device; converting here keeps the op graph-capturable.
  const at::Device compute_device = query.device();
  torch::Tensor pool_tail_k_t =
      pool_tail_k.to(torch::kLong).to(compute_device).contiguous();
  c10::optional<torch::Tensor> actual_seq_q_t;
  if (actual_seq_q.has_value() && actual_seq_q->defined()) {
    actual_seq_q_t =
        actual_seq_q->to(torch::kLong).to(compute_device).contiguous();
  }
  c10::optional<torch::Tensor> actual_seq_k_t;
  if (actual_seq_k.has_value() && actual_seq_k->defined()) {
    actual_seq_k_t =
        actual_seq_k->to(torch::kLong).to(compute_device).contiguous();
  }

  char* layout_q_ptr = const_cast<char*>(layout_q_str.c_str());
  char* layout_k_ptr = const_cast<char*>(layout_k_str.c_str());
  int64_t k_descale_stride0 = -1;

  if (sparse_indices_mode == 1) {
    EXEC_NPU_CMD_NAMED(aclnnPoolKeyIndexerV2TensorGetWorkspaceSize,
                       aclnnPoolKeyIndexerV2,
                       query,
                       pool_key,
                       weights,
                       pool_tail_k_t,
                       actual_seq_q_t,
                       actual_seq_k_t,
                       block_table,
                       q_descale,
                       k_descale,
                       topk,
                       pool_size,
                       layout_q_ptr,
                       layout_k_ptr,
                       mask_mode,
                       quant_mode,
                       return_value,
                       key_stride0,
                       k_descale_stride0,
                       sparse_indices_mode,
                       sparse_indices,
                       sparse_values);
  } else {
    EXEC_NPU_CMD_NAMED(aclnnPoolKeyIndexerTensorGetWorkspaceSize,
                       aclnnPoolKeyIndexer,
                       query,
                       pool_key,
                       weights,
                       pool_tail_k_t,
                       actual_seq_q_t,
                       actual_seq_k_t,
                       block_table,
                       q_descale,
                       k_descale,
                       topk,
                       pool_size,
                       layout_q_ptr,
                       layout_k_ptr,
                       mask_mode,
                       quant_mode,
                       return_value,
                       key_stride0,
                       k_descale_stride0,
                       sparse_indices,
                       sparse_values);
  }

  return std::make_tuple(sparse_indices, sparse_values);
}

}  // namespace xllm::kernel::npu
