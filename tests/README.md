# Tests

GPU tests covering every rasterizer variant in this repository. They run on
NVIDIA (CUDA) and AMD (ROCm) alike; nothing in them is vendor specific.

The variants have no reference implementation to be checked against, so they are
used as each other's oracle. All fourteen are three code bodies compiled at ten
different channel counts, and the first three colour channels composite
identically in all of them, so one scene rendered through every variant and
compared on channels 0-2 checks far more than a per-variant smoke test would.
The rendered depth, alpha, normals and distortion are a stricter oracle still,
because they depend only on the geometry and so are identical at every channel
count, as are the compositing weights the `wet` variants accumulate.

On top of that the suite checks `markVisible` against a CPU reimplementation of
the frustum test, forward determinism, the colour-from-spherical-harmonics path
a caller reaches by not precomputing colours, gradient finiteness, an opacity
finite difference against the analytic gradient, and the four-column
`dL_dmeans2D` the two `abs` variants return.

The `tile1` variant is the same algorithm with 1x1 tiles, and it is checked both
in bulk against the 16x16 tiling and on the one pixel where the two provably
disagree: tile bounds are computed by integer truncation, so a 1x1 tiling never
bins a surfel into the pixel column at `floor(centre + radius)` while a 16x16
tiling rounds that up to a whole tile and keeps it. The difference therefore
follows the sub-pixel position of the centre rather than the size of the surfel,
and the test sweeps that offset.

## Running

The bundled GLM headers are a submodule, so a clone without them cannot build
any variant:

```
git submodule update --init --recursive
```

Install every variant, then run the suite from the repository root:

```
for v in diff-surfel-rasterization*/; do
  ( cd "$v" && pip install -e . --no-build-isolation --no-deps ) || break
done
python -m pytest tests/test_variants.py -v
```

On ROCm, set `PYTORCH_ROCM_ARCH` to your GPU architecture (for example
`gfx942`, `gfx90a`, `gfx1100`) if torch does not detect it, and re-hipify from
scratch when you edit a source file, because torch leaves translated `.hip`
mirrors next to the originals:

```
rm -rf build *.egg-info hip_rasterizer *.hip
```

A variant that is not installed is skipped rather than failed, so the suite is
useful even when only some of them are built.
