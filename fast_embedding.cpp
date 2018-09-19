#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <torch/torch.h>
#include <THC/THC.h>

#include <stdio.h>
#include <vector>

// CUDA forward declarations

template<bool multi_embedding>
at::Tensor fast_embedding_backward_cuda_impl(
    at::Tensor grad_output,
    at::Tensor inputs,
    int64_t num_embeddings,
    cudaStream_t stream);

// C++ interface

extern THCState *state;

template<bool multi_embedding>
at::Tensor fast_embedding_backward_cuda(
    at::Tensor grad_output,
    at::Tensor inputs,
    int64_t num_embeddings) {

  cudaStream_t stream = THCState_getCurrentStream(state);
  return fast_embedding_backward_cuda_impl<multi_embedding>(
      grad_output,
      inputs,
      num_embeddings,
      stream);
}

template <typename scalar_t, bool multi_embedding>
void fast_embedding_backward_cpu_impl(
    scalar_t* grads,
    int64_t* indices,
    scalar_t* d_weights,
    int64_t size,
    int64_t embedding_dim) {
#ifdef _OPENMP
  #pragma omp parallel for if (size > 1000)
  for (int64_t ix = 0; ix < size; ++ix) {
    for (int64_t iy = 0; iy < embedding_dim; ++iy) {
      unsigned int idx = ix * embedding_dim + iy;
      auto ref = indices[multi_embedding ? idx : ix];
      const auto destAddress = ref * embedding_dim + iy;
      const auto grad = grads[idx];
      auto & mem = d_weights[destAddress];

      #pragma omp atomic
      mem += grad;
    }
  }
#else
  for (int64_t ix = 0; ix < size; ++ix) {
    for (int64_t iy = 0; iy < embedding_dim; ++iy) {
      unsigned int idx = ix * embedding_dim + iy;
      auto ref = indices[multi_embedding ? idx : ix];
      auto destAddress = ref * embedding_dim + iy;

      d_weights[destAddress] += grads[idx];
    }
  }
#endif
}

template<bool multi_embedding>
at::Tensor fast_embedding_backward_cpu(
    at::Tensor grads,
    at::Tensor indices,
    int64_t num_embeddings) {


  auto grads_arg = at::TensorArg(grads, "grad", 1);
  auto indices_arg = at::TensorArg(indices, "indices", 1);

  at::Tensor grads_cont;
  if (!multi_embedding) {
    auto num_indices = indices.numel();
    grads = grads.view({num_indices, grads.size(-1)}).contiguous();
  }

  at::checkScalarType("embedding_backward", indices_arg, at::kLong);

  auto d_weights = at::zeros(grads.type(), {num_embeddings, grads.size(-1)});
  int64_t size = grads.size(0);
  int64_t embedding_dim = grads.stride(0);
  assert(indices.size(0) == size);

  AT_DISPATCH_FLOATING_TYPES(grads.type(), "fast_embedding_backward_cpu", [&] {
    fast_embedding_backward_cpu_impl<scalar_t, multi_embedding>(
      grads.data<scalar_t>(),
      indices.data<int64_t>(),
      d_weights.data<scalar_t>(),
      size,
      embedding_dim);
  });
  return d_weights;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward_cuda", &fast_embedding_backward_cuda<false>, "fast_embedding backward (CUDA)");
  m.def("backward_cpu", &fast_embedding_backward_cpu<false>, "fast_embedding backward (CPU)");
  m.def("multi_backward_cuda", &fast_embedding_backward_cuda<true>, "fast_multi_embedding backward (CUDA)");
  m.def("multi_backward_cpu", &fast_embedding_backward_cpu<true>, "fast_multi_embedding backward (CPU)");
}
