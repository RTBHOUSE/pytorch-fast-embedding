from typing import List

import torch
from torch import nn
from torch.autograd import Function

import fast_embedding_native


class FastEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weights):
        inputs = inputs.contiguous()
        ctx.save_for_backward(inputs)
        ctx.num_weights = weights.size(0)

        return torch.embedding(weights, inputs)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        indices, = ctx.saved_tensors
        if grad_output.is_cuda:
            output = fast_embedding_native.backward_cuda(grad_output, indices, ctx.num_weights)
        else:
            output = fast_embedding_native.backward_cpu(grad_output, indices, ctx.num_weights)
        return None, output


class FastEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, _weight: torch.Tensor = None):
        super(FastEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        if _weight is None:
            self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

    def forward(self, x):
        return FastEmbeddingFunction.apply(x, self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def extra_repr(self):
        return f"{self.weight.size(0)}, {self.weight.size(1)}"


class FastMultiEmbeddingFunction(Function):
    @staticmethod
    def forward(ctx, inputs, weight):
        inputs = inputs.contiguous()
        ctx.save_for_backward(inputs)
        ctx.num_weights = weight.size(0)

        return torch.gather(weight, 0, inputs)

    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        if grad_output.is_cuda:
            output = fast_embedding_native.multi_backward_cuda(grad_output, indices, ctx.num_weights)
        else:
            output = fast_embedding_native.multi_backward_cpu(grad_output, indices, ctx.num_weights)
        return None, output


class FastMultiEmbedding(nn.Module):
    def __init__(self, embedding_sizes: List[int], embedding_dims: List[int], _weight: torch.Tensor = None):
        super(FastMultiEmbedding, self).__init__()
        assert len(embedding_sizes) == len(embedding_dims)
        self.embedding_sizes = embedding_sizes
        self.embedding_dims = embedding_dims

        membership_index = torch.tensor([i for i, v in enumerate(embedding_dims) for _ in range(v)])
        self.register_buffer("membership_index", membership_index)

        if _weight is None:
            self.weight = nn.Parameter(torch.zeros(max(embedding_sizes), sum(embedding_dims)))
            self.reset_parameters()
        else:
            assert list(_weight.shape) == [max(embedding_sizes), sum(embedding_dims)], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

    def forward(self, x):
        #  "Only two dimensional inputs are allowed"
        if x.dim() == 1:
            x.unsqueze_(1)
        index = x.long().index_select(1, self._buffers["membership_index"])
        return FastMultiEmbeddingFunction.apply(index, self.weight)

    def reset_parameters(self):
        self.weight.data.normal_(0, 1)

    def extra_repr(self):
        return f"{self.embedding_sizes}, {self.embedding_dims}"
