#include <ATen/ATen.h>
#include <ATen/cuda/AccumulateType.cuh>
#include <ATen/TensorUtils.h>
#include <THC/THCDeviceUtils.cuh>

#include <cuda.h>
#include <cuda_runtime.h>

#include <assert.h>


namespace {
template <typename scalar_t, bool multi_embedding>
__global__ void fast_embedding_backward_kernel(
  scalar_t* __restrict__ grads,
  int64_t* __restrict__ indices,
  scalar_t* __restrict__ grad_weights,
  int64_t size,
  int64_t embedding_dim) {

  using accscalar_t = at::cuda::acc_type<scalar_t>;

  int64_t ix = threadIdx.x;
  int64_t iy = blockIdx.x;

  if (ix < embedding_dim) {
    int64_t idx = iy * embedding_dim + ix;
    auto ref = indices[multi_embedding ? idx : iy];
    auto destAddress = ref * embedding_dim + ix;
    atomicAdd(&grad_weights[destAddress], static_cast<accscalar_t>(grads[idx]));
  }
}
}

template <bool multi_embedding>
at::Tensor fast_embedding_backward_cuda_impl(
    at::Tensor grads,
    at::Tensor indices,
    int64_t num_embeddings,
    cudaStream_t stream) {

  auto grads_arg = at::TensorArg(grads, "grad", 1);
  auto indices_arg = at::TensorArg(indices, "indices", 1);

  if (!multi_embedding) {
    auto num_indices = indices.numel();
    grads = grads.view({num_indices, grads.size(-1)}).contiguous();
  }

  at::checkScalarType("embedding_backward", indices_arg, at::kLong);
  at::checkSameGPU("embedding_backward", grads_arg, indices_arg);

  auto d_weights = at::zeros(grads.type(), {num_embeddings, grads.size(-1)});
  int64_t size = grads.size(0);
  int64_t embedding_dim = grads.stride(0);

  dim3 block(32 * THCCeilDiv(embedding_dim, (int64_t) 32));
  dim3 grid(size);

  AT_DISPATCH_FLOATING_TYPES(grads.type(), "fast_embedding_backward_cuda", ([&] {
    fast_embedding_backward_kernel<scalar_t, multi_embedding><<<grid, block, 0, stream>>>(
      grads.data<scalar_t>(),
      indices.data<int64_t>(),
      d_weights.data<scalar_t>(),
      size,
      embedding_dim);
  }));

  return d_weights;
}

template at::Tensor fast_embedding_backward_cuda_impl<true>(at::Tensor, at::Tensor, int64_t, cudaStream_t);
template at::Tensor fast_embedding_backward_cuda_impl<false>(at::Tensor, at::Tensor, int64_t, cudaStream_t);


