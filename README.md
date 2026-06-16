# diff-surfel-rasterizations
Collection of differentiable surfel rasterizers with different feature channels.

## AMD GPU support

These rasterizers build and run on AMD GPUs through ROCm. Each variant is a
PyTorch CUDA extension, so when a ROCm build of PyTorch is installed, torch's
`hipify` translates the CUDA sources to HIP at build time and defines
`USE_ROCM` automatically -- the usual `pip install -e .` (or `pip install .`)
in a variant directory just works, with no separate ROCm-specific command. The
NVIDIA CUDA build path is unchanged. Set `PYTORCH_ROCM_ARCH` to your GPU
architecture (for example `gfx90a`, `gfx1100`, or `gfx1201`) if torch does not
detect it.
