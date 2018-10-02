import copy
from itertools import product
from typing import List

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from fast_embedding import FastEmbedding, FastMultiEmbedding

TEST_SIZE = 8192


def test_initialization():
    model = FastEmbedding(4, 3)
    assert not model.weight.data.eq(0).all()
    model = FastMultiEmbedding([4], [3])
    assert not model.weight.data.eq(0).all()


def test_repr():
    assert repr(FastEmbedding(4, 3)) == "FastEmbedding(4, 3)"
    assert repr(FastMultiEmbedding([4, 5], [3, 7])) == "FastMultiEmbedding([4, 5], [3, 7])"


@pytest.mark.parametrize("cuda", [False, True])
def test_simple_forward(cuda: bool):
    weight = torch.tensor(np.arange(0, 12) * 10, dtype=torch.float32).view(4, 3)
    model = FastEmbedding(4, 3, _weight=weight)
    if cuda:
        model = model.cuda()

    inputs = [
        torch.tensor([1, 2, 1], dtype=torch.int64),
        torch.tensor([], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([2], dtype=torch.int64).view(1, -1)
    ]
    expected_outputs = [
        torch.tensor([[30., 40., 50.], [60., 70., 80.], [30., 40., 50.]]),
        torch.tensor([], dtype=torch.float32),
        torch.tensor([[60., 70., 80.]]),
        torch.tensor([[[60., 70., 80.]]])
    ]

    for i, t in zip(inputs, expected_outputs):
        if cuda:
            assert model(i.cuda()).eq(t.cuda()).all()

    model = FastMultiEmbedding([4], [3], _weight=weight)
    if cuda:
        model = model.cuda()
    inputs = [
        torch.tensor([[1], [2], [1]], dtype=torch.int64),
        torch.tensor([[2]], dtype=torch.int64)
    ]
    expected_outputs = [
        torch.tensor([[30., 40., 50.], [60., 70., 80.], [30., 40., 50.]]),
        torch.tensor([[60., 70., 80.]])
    ]
    for i, t in zip(inputs, expected_outputs):
        if cuda:
            assert model(i.cuda()).eq(t.cuda()).all()


@pytest.mark.parametrize("cuda", [False, True])
def test_simple_backward(cuda: bool):
    weight = torch.tensor(np.arange(0, 12) * 10, dtype=torch.float32).view(4, 3)
    model = FastEmbedding(4, 3, _weight=weight)
    if cuda:
        model = model.cuda()
    indices = [
        torch.tensor([1, 2, 1, 1, 3], dtype=torch.int64),
        torch.tensor([1, 2, 1, 1, 3], dtype=torch.int64).view(-1, 1),
        torch.tensor([[1, 2], [2, 3], [1, 2], [1, 2], [3, 0]], dtype=torch.int64)
    ]
    grad_outputs = [
        torch.cat([torch.arange(1, 6, dtype=torch.float32).view(-1, 1),
                   torch.arange(1, 6, dtype=torch.float32).view(-1, 1) * 10,
                   torch.arange(1, 6, dtype=torch.float32).view(-1, 1) * 100
                   ], 1),
        torch.cat([torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1),
                   torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 10,
                   torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 100
                   ], 2),
        torch.cat([
            torch.cat([
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1),
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 10,
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 100], 2),
            torch.cat([
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1),
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 10,
                torch.arange(1, 6, dtype=torch.float32).view(-1, 1, 1) * 100], 2)
        ], 1)
    ]
    expected_grads = [
        torch.tensor([[0., 0., 0.], [8., 80., 800.], [2., 20., 200.], [5., 50., 500.]], dtype=torch.float32),
        torch.tensor([[0., 0., 0.], [8., 80., 800.], [2., 20., 200.], [5., 50., 500.]], dtype=torch.float32),
        torch.tensor([[5., 50., 500.], [8., 80., 800.], [10., 100., 1000.], [7., 70., 700.]], dtype=torch.float32)
    ]

    for i, go, t in zip(indices, grad_outputs, expected_grads):
        m = copy.deepcopy(model)
        if cuda:
            m = m.cuda()
            i = i.cuda()
            go = go.cuda()
            t = t.cuda()
        y = m(i)
        y.backward(go)
        assert m.weight.grad.data.eq(t).all()


@pytest.mark.parametrize("multi,cuda,double", product([False, True], repeat=3))
def test_forward(multi, cuda, double):
    sizes = [10, 100, 1000, 10000]
    dims = [16, 9, 5, 3]
    offsets = get_offsets(dims)
    model = create_embedding_module(sizes, dims, multi=multi, cuda=cuda, double=double)
    x, x_data = create_index(TEST_SIZE, sizes, cuda=cuda)
    o = model(x)
    o_data = get_output(o, offsets)
    weight = get_weight(model, multi, offsets)
    for col in range(len(dims)):
        for row in range(8192):
            assert_almost_equals(o_data[col][row], weight[col][x_data[row, col]], delta=1e-4)


@pytest.mark.parametrize("multi,cuda,double,dim", product([False, True], [False, True], [False, True], [3, 4, 9]))
def test_backward_single_embedding(multi, cuda, double, dim):
    sizes = [100]
    dims = [dim]
    offsets = get_offsets(dims)
    model = create_embedding_module(sizes, dims, multi=multi, cuda=cuda, double=double)
    x, x_data = create_index(TEST_SIZE, sizes, cuda=cuda)
    t_data = (x_data % 3).astype(np.float32)
    t, t_data = create_target(t_data, double=double, cuda=cuda)
    o = model(x).sum(1, keepdim=True)
    loss = F.mse_loss(o, t)
    loss.backward()
    weight = get_weight(model, multi, offsets)
    grad = get_grad(model, multi, offsets)
    for idx in range(100):
        mask = x_data == idx
        value = weight[0][idx, :].sum()
        cur_t = np.tile(np.expand_dims(t_data[mask], 1), dims[0])
        expected_grad = np.sum(2 * (value - cur_t), axis=0) / TEST_SIZE
        assert_almost_equals(grad[0][idx, :], expected_grad, delta=1e-4)


@pytest.mark.parametrize("multi,cuda,double", product([False, True], repeat=3))
def test_backward_multiple_embeddings(multi, cuda, double):
    import warnings
    sizes = [10, 100, 1000, 10000]
    dims = [16, 9, 5, 3]
    offsets = get_offsets(dims)
    model = create_embedding_module(sizes, dims, multi=multi, cuda=cuda, double=double)
    x, x_data = create_index(TEST_SIZE, sizes, cuda=cuda)
    t_data = np.sum(x_data % 3, axis=1, keepdims=True).astype(np.float32)
    t, t_data = create_target(t_data, double=double, cuda=cuda)
    o = model(x).sum(1, keepdim=True)
    loss = F.mse_loss(o, t)
    loss.backward()
    grad = get_grad(model, multi, offsets)
    output = o.detach().cpu().numpy()
    for col, size in enumerate(sizes):
        for idx in range(size):
            mask = x_data[:, col] == idx
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                value = output[mask].mean()
                cur_t = np.tile(t_data[mask], dims[col])
            expected_grad = np.sum(2 * (value - cur_t), axis=0) / TEST_SIZE
            assert_almost_equals(grad[col][idx, :], expected_grad, delta=1e-4)


def create_embedding_module(dict_sizes: List[int], embedding_sizes: List[int], multi=False, cuda=False, double=False):
    if multi:
        model = FastMultiEmbedding(dict_sizes, embedding_sizes)
        model.weight.data.normal_()
    else:
        class Model(nn.Module):
            def __init__(self, dict_sizes, embedding_sizes):
                super(Model, self).__init__()
                self.submodules = nn.ModuleList([FastEmbedding(size, dim) for size, dim in zip(dict_sizes, embedding_sizes)])

            def forward(self, x):
                return torch.cat([m(i.squeeze(1)) for i, m in zip(x.long().split(1, 1), self.submodules)], dim=1)

        model = Model(dict_sizes, embedding_sizes)
        for item in model.submodules:
            item.weight.data.normal_()
    if double:
        model.double()
    if cuda:
        return model.cuda()
    else:
        return model


def create_index(size: int, dict_sizes: List[int], cuda: bool = True):
    x_data = np.zeros((size, len(dict_sizes)), dtype=np.int64)
    for i, item in enumerate(dict_sizes):
        x_data[:, i] = np.random.randint(0, item, size)

    x = torch.from_numpy(x_data)
    if cuda:
        x = x.cuda()

    return x, x_data


def create_target(t_data, double, cuda):
    t = torch.from_numpy(t_data)
    if double:
        t_data = t_data.astype(np.float64)
        t = t.double()
    if cuda:
        t = t.cuda()
    return t, t_data


def get_offsets(dims):
    offsets = [0]
    sum = 0
    for item in dims:
        sum += item
        offsets.append(sum)
    return offsets


def get_output(o, offsets=None):
    return [o.detach()[:, offsets[i]:offsets[i + 1]].cpu().numpy() for i in range(len(offsets) - 1)]


def get_weight(m, multi, offsets=None):
    if multi:
        return [m.weight.data[:, offsets[i]:offsets[i + 1]].cpu().numpy() for i in range(len(offsets) - 1)]
    else:
        return [x.weight.data.cpu().numpy() for x in m.submodules]


def get_grad(m, multi, offsets=None):
    if multi:
        return [m.weight.grad[:, offsets[i]:offsets[i + 1]].cpu().numpy() for i in range(len(offsets) - 1)]
    else:
        return [x.weight.grad.cpu().numpy() for x in m.submodules]


def assert_almost_equals(x, y, delta):
    if isinstance(x, float) or isinstance(x, np.float32):
        assert isinstance(y, float) or isinstance(y, np.float32), f"{type(x)}, {type(y)}"
        assert abs(x - y) < delta, f"observed: {x}, expected: {y}"
    elif isinstance(x, np.ndarray):
        assert isinstance(y, np.ndarray)
        assert x.shape == y.shape
        assert np.all(np.abs(x - y) < delta), f"observed: {x}, expected: {y}"
    elif isinstance(x, torch.Tensor):
        assert isinstance(y, torch.Tensor)
        assert x.size() == y.size(), f"{x.size()} != {y.size()}"
        assert (x - y).abs().sum() < delta
    else:
        raise NotImplementedError

