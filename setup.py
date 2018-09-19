from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fast_embedding_native',
    ext_modules=[
        CUDAExtension('fast_embedding_native', [
            'fast_embedding.cpp',
            'fast_embedding_kernel.cu',
        ], extra_compile_args={"nvcc": ["-gencode", "arch=compute_61,code=compute_61", "-gencode", "arch=compute_60,code=compute_60"],
                               "cxx": ["-fopenmp"]}),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
