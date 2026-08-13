# diff-surfel-rasterizations
Collection of differentiable surfel rasterizers with different feature channels.

## AMD GPU support

All fourteen rasterizer variants build and run on AMD GPUs through ROCm. Each variant is a
PyTorch CUDA extension, so when a ROCm build of PyTorch is installed, torch's
`hipify` translates the CUDA sources to HIP at build time and defines
`USE_ROCM` automatically -- the usual `pip install -e .` (or `pip install .`)
in a variant directory just works, with no separate ROCm-specific command. The
NVIDIA CUDA build path is unchanged. Set `PYTORCH_ROCM_ARCH` to your GPU
architecture (for example `gfx942`, `gfx90a`, `gfx1100`, or `gfx1201`) if torch
does not detect it.

The bundled GLM headers under `third_party/glm` are a submodule, so clone with
`--recurse-submodules` (or run `git submodule update --init --recursive`) before
building. Without them the compiler falls back to whatever GLM is installed
system-wide, which is unlikely to recognise HIP as a device compiler and fails
with GLM functions reported as host-only.

`tests/` renders one scene through every variant and cross-checks them; see
`tests/README.md`.
