#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# ROCm/HIP build support author: Jeff Daily <jeff.daily@amd.com>
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import shutil
import torch

GLM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "third_party", "glm")


def _patch_hipify_ignore_glm():
    """Keep the bundled third_party/glm out of torch's hipify on ROCm.

    torch's hipify (via CUDAExtension) walks every .hpp under the build dir and
    the extension include dirs into its file set, then content-rewrites any GLM
    header a source pulls in -- which drops GLM's .inl files (hipify only copies
    .hpp/.h) and mangles GLM's __CUDACC__/__HIP__ compiler detection, breaking
    the build. The bundled GLM already detects __HIP__ (glm/simd/platform.h ->
    GLM_COMPILER_HIP -> __device__ __host__) and compiles verbatim under the
    -x hip pass, so the fix is to leave it untouched: add the glm dir to
    hipify's ``ignores`` and drop it from ``header_include_dirs``. The source
    keeps including <glm/...> via -I, resolved against the pristine bundled tree.
    """
    if not torch.version.hip:
        return
    from torch.utils.hipify import hipify_python

    glm_patterns = [os.path.join(GLM_DIR, "*"), GLM_DIR + "*"]
    orig_hipify = hipify_python.hipify

    def hipify_no_glm(*args, **kwargs):
        kwargs["ignores"] = list(kwargs.get("ignores", ())) + glm_patterns
        kwargs["header_include_dirs"] = [
            d for d in kwargs.get("header_include_dirs", [])
            if os.path.abspath(d) != os.path.abspath(GLM_DIR)
        ]
        return orig_hipify(*args, **kwargs)

    hipify_python.hipify = hipify_no_glm


_patch_hipify_ignore_glm()

# On Windows with ROCm/HIP, MSVC cl.exe compiles .cpp files but cannot link
# c10::ValueError(SourceLocation, string) from the Clang-built c10.dll (the
# inherited ctor is absent from the MSVC import lib). Fix: copy ext.cpp to
# ext_winhip.cu so torch's BuildExtension routes it through hipcc/amdclang++,
# which shares the same ABI as c10.dll. The copy has identical contents; it is
# guarded so Linux and CUDA-Windows builds are unaffected.
_ext_sources = [
    "cuda_rasterizer/rasterizer_impl.cu",
    "cuda_rasterizer/forward.cu",
    "cuda_rasterizer/backward.cu",
    "rasterize_points.cu",
    "ext.cpp",
]
if os.name == 'nt' and torch.version.hip:
    shutil.copy("ext.cpp", "ext_winhip.cu")
    _ext_sources = [s if s != "ext.cpp" else "ext_winhip.cu" for s in _ext_sources]

setup(
    name="diff_surfel_rasterization_wet",
    packages=['diff_surfel_rasterization_wet'],
    version='0.0.1',
    ext_modules=[
        CUDAExtension(
            name="diff_surfel_rasterization_wet._C",
            sources=_ext_sources,
            extra_compile_args={"nvcc": ["-I" + GLM_DIR]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
