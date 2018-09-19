import time
from enum import Enum

import numpy as np
import pandas as pd
import torch
import torch.legacy.nn
import torch.nn as nn
from fast_embedding import FastEmbedding, FastMultiEmbedding


class EmbScenario(Enum):
    UNIFORM = "uniform"
    POW_1_2_K = "0.5^k"  # e.g. for batch_size=32 - shuffle([16 x 1, 8 x 2, 4 x 3, 2 x 4, 1 x 5, 1 x 6])
    SAME_INDEX = "same index - 100% collisions"
    UNI_AND_BINOMIAL = "50% uniform; 50% binomial (p=0.99)"
    UNI_AND_20P_SAME_INDEX = "80% UNIFORM, 20% SAME_INDEX"


class MultiEmbScenario(Enum):
    ALL_UNIFORM = ("all: uniform", lambda idx, dim: EmbScenario.UNIFORM)
    ALL_MIXED = ("all: 50% uniform; 50% binomial (p=0.99)", lambda idx, dim: EmbScenario.UNI_AND_BINOMIAL)
    ALL_POW_1_2_K = ("all: 0.5^k", lambda idx, dim: EmbScenario.POW_1_2_K)
    MIXED = ("50% MIXED; 50% UNIFORM", lambda idx, dim: [EmbScenario.UNIFORM, EmbScenario.UNI_AND_BINOMIAL][idx % 2])
    MIXED_POW_1_2_K = ("50 UNIFORM; 50%: 0.5^k", lambda idx, dim: [EmbScenario.UNIFORM, EmbScenario.POW_1_2_K][idx % 2])
    SMALL_UNI_LARGE_MIX = ("small embeddings - UNIFORM; large embeddings - MIXED",
                           lambda idx, dim: [EmbScenario.UNIFORM, EmbScenario.UNI_AND_BINOMIAL][dim > 32])
    SMALL_MIX_LARGE_UNI = ("small embeddings - MIXED; large embeddings - UNIFORM",
                           lambda idx, dim: [EmbScenario.UNIFORM, EmbScenario.UNI_AND_BINOMIAL][dim <= 32])


class MultiEmbConfig(Enum):
    MT_10XS = [1, 2, 4, 8, 1, 2, 4, 8, 1, 2]
    MT_20XS = [2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4, 8, 2, 4]
    MT_5S_5L = [4, 8, 16, 24, 32, 64, 96, 128, 192, 256]
    MT_5L_5S = [256, 192, 128, 96, 64, 32, 24, 16, 8, 4]
    MT_5S_5L_RND = [8, 96, 16, 32, 128, 64, 24, 256, 192, 4]
    MT_10S_10L_RND = [256, 4, 24, 96, 192, 64, 4, 128, 16, 64, 32, 128, 96, 24, 32, 256, 192, 8, 16, 8]
    MT_10S = [16, 8, 4, 24, 16, 24, 32, 8, 32, 4]
    MT_10S_UNEVEN = [17, 9, 5, 25, 17, 25, 33, 9, 33, 5]
    MT_20S = [16, 8, 4, 24, 16, 24, 32, 8, 32, 4] * 2
    MT_10L = [128, 256, 64, 192, 256, 128, 96, 192, 64, 96]
    MT_10S_1L = [16, 8, 256, 4, 24, 16, 24, 32, 8, 32, 4]


class MultiEmbModuleType(Enum):
    FAST_MULTI = 'fast_multi'
    FAST = 'fast'
    ORIGINAL = 'original'


def measure_execution_time(num_iters, body):
    body()  # warmup run
    body()  # warmup run

    run_times = []
    for i in range(num_iters):
        start_time = time.time()
        body()
        end_time = time.time()
        run_times.append(end_time - start_time)

    run_times = np.array(run_times)

    return run_times.mean(), run_times.std(), run_times


def create_indices(batch_size, num_embeddings, scenario):
    if scenario == EmbScenario.UNIFORM:
        indices = np.random.randint(0, num_embeddings, size=batch_size)
    elif scenario == EmbScenario.POW_1_2_K:
        assert batch_size & (batch_size - 1) == 0  # batch size must be power of 2
        indices = np.zeros(batch_size)
        offset = 0
        length = batch_size
        idx = 0

        while length:
            indices[offset:offset + length // 2] = idx
            length = length // 2
            offset += length
            idx += 1

        np.random.shuffle(indices)
        indices = np.minimum(indices, num_embeddings - 1)
    elif scenario == EmbScenario.SAME_INDEX:
        indices = np.ones(batch_size) * int(np.random.randint(0, num_embeddings))
    elif scenario == EmbScenario.UNI_AND_BINOMIAL:
        indices = np.random.binomial(num_embeddings - 1, p=0.99, size=batch_size)
        indices[:batch_size // 2] = np.random.randint(0, num_embeddings, size=batch_size // 2)
        np.random.shuffle(indices)
    elif scenario == EmbScenario.UNI_AND_20P_SAME_INDEX:
        indices = np.random.randint(0, num_embeddings, size=batch_size)
        indices[:batch_size // 5] = 0
    else:
        raise ValueError("Invalid scenario")

    return torch.Tensor(indices).long()


def run_embedding_test(emb_module, num_embeddings, embedding_dim, batch_size, scenario: EmbScenario, device, num_iters=10):
    emb = emb_module(num_embeddings, embedding_dim).to(device)
    linear = nn.Linear(embedding_dim, 1).to(device)
    indices = create_indices(batch_size, num_embeddings, scenario).to(device)

    on_cuda = device.type == 'cuda'

    def run():
        x = emb.forward(indices)
        linear.forward(x).sum().backward()
        if on_cuda:
            torch.cuda.synchronize()

    return measure_execution_time(num_iters, run)


def run_multi_embedding_test(module_type: MultiEmbModuleType, conf: MultiEmbConfig, batch_size, scenario: MultiEmbScenario, device, num_iters=10):
    dims = conf.value
    sizes = ([1000, 5000, 10000, 50000] * 10)[:len(conf.value)]

    if module_type == MultiEmbModuleType.FAST_MULTI:
        emb = FastMultiEmbedding(sizes, dims).to(device)
    elif module_type == MultiEmbModuleType.FAST:
        embs = [FastEmbedding(size, dim).to(device) for size, dim in zip(sizes, dims)]

        def emb(multi_indices):
            return torch.cat([e(multi_indices[:, idx]) for (idx, e) in enumerate(embs)], 1)
    else:
        embs = [nn.Embedding(size, dim).to(device) for size, dim in zip(sizes, dims)]

        def emb(multi_indices):
            return torch.cat([e(multi_indices[:, idx]) for (idx, e) in enumerate(embs)], 1)

    linear = nn.Linear(sum(dims), 1).to(device)
    indices = torch.cat([create_indices(batch_size, num_embs, scenario.value[1](idx, dim)).view(-1, 1)
                         for idx, (dim, num_embs) in enumerate(zip(dims, sizes))], 1).to(device)

    on_cuda = device.type == 'cuda'

    def run():
        x = emb(indices)
        linear.forward(x).sum().backward()
        if on_cuda:
            torch.cuda.synchronize()

    return measure_execution_time(num_iters, run)


def run_all_embedding_tests(num_iters):
    batch_sizes = (2 ** np.arange(5, 15)).astype(np.int32)
    results = []

    for device in [torch.device('cuda'), torch.device('cpu')]:
        for emb_module in [nn.Embedding, FastEmbedding]:
            for num_embeddings in [10, 100, 1000, 10_000, 100_000, 1_000_000]:
                for embedding_dim in [8, 32, 128, 256]:
                    for batch_size in batch_sizes:
                        for scenario in EmbScenario:
                            time_mean, time_std, _ = run_embedding_test(
                                emb_module, num_embeddings, embedding_dim, batch_size, scenario, device, num_iters)
                            result = [emb_module.__name__, num_embeddings, embedding_dim, batch_size, scenario.name, device.type, time_mean, time_std]
                            results.append(result)
                            print(result)

    df = pd.DataFrame(results)
    df.columns = ["emb_module", "num_embeddings", "embedding_dim", "batch_size", "scenario", "device", "time_mean", "time_std"]
    return df


def run_all_multi_embedding_tests(num_iters):
    batch_sizes = (2 ** np.arange(5, 15)).astype(np.int32)
    results = []

    for device in [torch.device('cuda'), torch.device('cpu')]:
        for emb_module in MultiEmbModuleType:
            for config in MultiEmbConfig:
                for batch_size in batch_sizes:
                    for scenario in MultiEmbScenario:
                        time_mean, time_std, _ = run_multi_embedding_test(emb_module, config, batch_size, scenario, device, num_iters)
                        result = [emb_module.name, config.name, batch_size, scenario.name, device.type, time_mean, time_std]
                        results.append(result)
                        print(result)

    df = pd.DataFrame(results)
    df.columns = ["emb_module", "config", "batch_size", "scenario", "device", "time_mean", "time_std"]
    return df


def run_all_tests():
    with open("emb_perf_results.csv", "w") as f_out:
        for i in range(20):
            result = run_all_embedding_tests(10)
            result.insert(0, "run", i)
            if i == 0:
                result.to_csv(f_out)
            else:
                result.to_csv(f_out, header=False)

    with open("multi_emb_perf_results.csv", "w") as f_out:
        for i in range(20):
            result = run_all_multi_embedding_tests(10)
            result.insert(0, "run", i)
            if i == 0:
                result.to_csv(f_out)
            else:
                result.to_csv(f_out, header=False)


if __name__ == '__main__':
    run_all_tests()
