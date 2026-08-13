#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""GPU tests for every rasterizer variant in this repository.

There is no reference implementation to compare against, so the variants act as
each other's oracle: they are three code bodies compiled at ten channel counts,
and the alpha-compositing arithmetic of the first three channels is the same in
all of them. Rendering one scene through all of them and comparing channels 0-2
turns coverage into correctness -- a variant that compiles but computes garbage
fails here, which a per-variant smoke test would miss. The auxiliary buffer is a
stronger oracle still: depth, alpha, normals and distortion depend only on the
geometry, so they are identical at every channel count.

Install every variant first (see tests/README.md), then::

    python -m pytest tests/test_variants.py -v
"""

import collections
import importlib
import math

import pytest
import torch

# variant directory -> (python package, NUM_CHANNELS, BLOCK_X, family)
VARIANTS = {
    "diff-surfel-rasterization": ("diff_surfel_rasterization", 3, 16, "base"),
    "diff-surfel-rasterization-ch05": ("diff_surfel_rasterization_ch05", 5, 16, "base"),
    "diff-surfel-rasterization-ch11": ("diff_surfel_rasterization_ch11", 11, 16, "base"),
    "diff-surfel-rasterization-ch18": ("diff_surfel_rasterization_ch18", 18, 16, "base"),
    "diff-surfel-rasterization-ch26": ("diff_surfel_rasterization_ch26", 26, 16, "base"),
    "diff-surfel-rasterization-tile1": ("diff_surfel_rasterization_tile1", 3, 1, "base"),
    "diff-surfel-rasterization-wet": ("diff_surfel_rasterization_wet", 3, 16, "wet"),
    "diff-surfel-rasterization-wet-ch05": ("diff_surfel_rasterization_wet_ch05", 5, 16, "wet"),
    "diff-surfel-rasterization-wet-ch07": ("diff_surfel_rasterization_wet_ch07", 7, 16, "wet"),
    "diff-surfel-rasterization-wet-ch11": ("diff_surfel_rasterization_wet_ch11", 11, 16, "wet"),
    "diff-surfel-rasterization-wet-ch18": ("diff_surfel_rasterization_wet_ch18", 18, 16, "wet"),
    "diff-surfel-rasterization-wet-ch26": ("diff_surfel_rasterization_wet_ch26", 26, 16, "wet"),
    "diff-surfel-rasterization-wet-abs": ("diff_surfel_rasterization_wet_abs", 3, 16, "wet-abs"),
    "diff-surfel-rasterization-wet-abs-ch05": ("diff_surfel_rasterization_wet_abs_ch05", 5, 16, "wet-abs"),
}

BASE = "diff-surfel-rasterization"
TILE1 = "diff-surfel-rasterization-tile1"
WET = "diff-surfel-rasterization-wet"
DEVICE = "cuda"
RTOL, ATOL = 1e-4, 1e-5

# One block per pixel is only tractable at a small resolution, so the 1x1-tile
# variant runs the per-variant checks on a smaller scene with larger surfels.
# Larger is deliberate: below about three times the default scale every surfel
# lands on the filter-size floor of the projected radius, and dL_dmean2D is
# written only where the 2D filter is not the binding constraint, so the
# returned means2D gradient is identically zero -- for the 16x16 tiling too.
SMALL_SCENE = dict(width=32, height=24, num_points=1500, scale=3.0)

# The two abs variants return dL_dmeans2D with four columns (2-3 carry the
# homodirectional absolute gradient), and autograd matches gradient shape to
# input shape, so means2D has to be handed in with four columns for them.
MEANS2D_COLS = {"base": 3, "wet": 3, "wet-abs": 4}

# Depth range the test scene puts its surfels in, in world units.
SCENE_Z = (2.0, 6.0)

# How many surfels the culling check moves behind the camera, on its own copy of
# the scene. Small enough that the rest of the render is unchanged.
CULLED_SURFELS = 8

# out_others channel layout, from cuda_rasterizer/auxiliary.h.
DEPTH, ALPHA, NORMAL, MIDDEPTH, DISTORTION = 0, 1, slice(2, 5), 5, 6

Rendered = collections.namedtuple("Rendered", "color radii others weight args")


def load(variant):
    pkg = VARIANTS[variant][0]
    try:
        return importlib.import_module(pkg)
    except ImportError as exc:  # pragma: no cover - reported as a skip
        pytest.skip(f"{pkg} is not installed: {exc}")


def projection(znear, zfar, tanfovx, tanfovy):
    """3DGS-convention projection matrix, already transposed for the kernel."""
    p = torch.zeros(4, 4)
    p[0, 0] = 1.0 / tanfovx
    p[1, 1] = 1.0 / tanfovy
    p[2, 2] = zfar / (zfar - znear)
    p[3, 2] = 1.0
    p[2, 3] = -(zfar * znear) / (zfar - znear)
    return p.transpose(0, 1)


def field_of_view(width, height):
    tanfovy = math.tan(0.5 * math.radians(60.0))
    return tanfovy * width / height, tanfovy


def settings_for(channels, width, height, sh_degree=0):
    tanfovx, tanfovy = field_of_view(width, height)
    viewmatrix = torch.eye(4)
    return {
        "image_height": height,
        "image_width": width,
        "tanfovx": tanfovx,
        "tanfovy": tanfovy,
        "bg": torch.zeros(channels, device=DEVICE),
        "scale_modifier": 1.0,
        "viewmatrix": viewmatrix.to(DEVICE),
        "projmatrix": (viewmatrix @ projection(0.01, 100.0, tanfovx, tanfovy)).to(DEVICE),
        "sh_degree": sh_degree,
        "campos": torch.zeros(3, device=DEVICE),
        "prefiltered": False,
        "debug": False,
    }


def make_scene(channels, means2d_cols, num_points=4000, width=200, height=150,
               seed=0, scale=1.0):
    """A fixed camera at the origin looking down +z, with surfels in front of it.

    The first three colour columns are the same values for every channel count,
    which is what makes the cross-variant comparison meaningful.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rand(*shape):
        return torch.rand(*shape, generator=gen)

    near, far = SCENE_Z
    means3d = torch.stack([
        rand(num_points) * 2.0 - 1.0,
        rand(num_points) * 1.5 - 0.75,
        rand(num_points) * (far - near) + near,
    ], dim=1)
    scales = (rand(num_points, 2) * 0.04 + 0.02) * scale
    rotations = rand(num_points, 4) * 2.0 - 1.0
    rotations = rotations / rotations.norm(dim=1, keepdim=True)
    opacities = rand(num_points, 1) * 0.7 + 0.2

    # Same first three columns at every channel count; the rest is padding that
    # only the wider variants ever read.
    colors = torch.empty(num_points, channels)
    base_gen = torch.Generator(device="cpu").manual_seed(seed + 1)
    colors[:, :3] = torch.rand(num_points, 3, generator=base_gen)
    if channels > 3:
        pad_gen = torch.Generator(device="cpu").manual_seed(seed + 2)
        colors[:, 3:] = torch.rand(num_points, channels - 3, generator=pad_gen)

    return {
        "means3D": means3d.to(DEVICE),
        "means2D": torch.zeros(num_points, means2d_cols, device=DEVICE),
        "opacities": opacities.to(DEVICE),
        "scales": scales.to(DEVICE),
        "rotations": rotations.to(DEVICE),
        "colors_precomp": colors.to(DEVICE),
        "settings": settings_for(channels, width, height),
    }


def add_spherical_harmonics(scene, degree=3, seed=5):
    """Give a three-channel scene an SH colour field of the requested degree.

    The band-0 coefficient reproduces the precomputed colours the scene already
    carries (computeColorFromSH adds the 0.5 offset itself), so the bands above
    it are the only difference between this render and the precomputed one.
    """
    num_points = scene["means3D"].shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    coefficients = torch.zeros(num_points, (degree + 1) ** 2, 3)
    sh_c0 = 0.28209479177387814
    coefficients[:, 0, :] = (scene["colors_precomp"][:, :3].cpu() - 0.5) / sh_c0
    coefficients[:, 1:, :] = (torch.rand(num_points, (degree + 1) ** 2 - 1, 3,
                                         generator=gen) - 0.5) * 0.3
    scene = dict(scene)
    scene["shs"] = coefficients.to(DEVICE)
    scene["settings"] = dict(scene["settings"], sh_degree=degree)
    return scene


def render(variant, scene, requires_grad=False, use_sh=False):
    """Run one forward pass and return every tensor the rasterizer produces."""
    mod = load(variant)
    settings = mod.GaussianRasterizationSettings(**scene["settings"])
    rasterizer = mod.GaussianRasterizer(raster_settings=settings)

    names = ["means3D", "means2D", "opacities", "scales", "rotations",
             "shs" if use_sh else "colors_precomp"]
    args = {}
    for name in names:
        tensor = scene[name].clone()
        if requires_grad:
            tensor.requires_grad_(True)
            tensor.retain_grad()
        args[name] = tensor

    colour = {"shs": args["shs"]} if use_sh else {"colors_precomp": args["colors_precomp"]}
    out = rasterizer(
        means3D=args["means3D"],
        means2D=args["means2D"],
        opacities=args["opacities"],
        scales=args["scales"],
        rotations=args["rotations"],
        **colour,
    )
    # Every variant returns (colour, radii, auxiliary buffer); the wet families
    # append the per-surfel accumulated weight.
    return Rendered(out[0], out[1], out[2], out[3] if len(out) > 3 else None, args)


def scene_for(variant, **kwargs):
    _, channels, _, family = VARIANTS[variant]
    return make_scene(channels, MEANS2D_COLS[family], **kwargs)


def scene_kwargs(variant):
    """1x1 tiles need the small scene; everything else uses the default one."""
    return dict(SMALL_SCENE) if VARIANTS[variant][2] == 1 else {}


def default_scene(variant):
    return scene_for(variant, **scene_kwargs(variant))


def weighted_loss(color):
    weights = torch.linspace(0.5, 1.5, color.numel(), device=DEVICE).reshape(color.shape)
    return weights, (color * weights).sum()


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no GPU available")


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_import_and_symbols(variant):
    mod = load(variant)
    for symbol in ("rasterize_gaussians", "rasterize_gaussians_backward", "mark_visible"):
        assert hasattr(mod._C, symbol), f"{variant} does not export {symbol}"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_mark_visible_matches_cpu_reference(variant):
    """checkFrustum keeps a point when its view-space z exceeds 0.2."""
    mod = load(variant)
    scene = default_scene(variant)
    settings = mod.GaussianRasterizationSettings(**scene["settings"])

    gen = torch.Generator(device="cpu").manual_seed(7)
    points = torch.stack([
        torch.rand(2048, generator=gen) * 2.0 - 1.0,
        torch.rand(2048, generator=gen) * 2.0 - 1.0,
        torch.rand(2048, generator=gen) * 7.0 - 1.0,  # straddles the near plane
    ], dim=1).to(DEVICE)

    got = mod.GaussianRasterizer(raster_settings=settings).markVisible(points)

    view = scene["settings"]["viewmatrix"].flatten()
    z = view[2] * points[:, 0] + view[6] * points[:, 1] + view[10] * points[:, 2] + view[14]
    expected = z > 0.2

    assert torch.equal(got, expected)
    assert expected.any() and not expected.all(), "test scene must straddle the near plane"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_forward_is_finite_and_non_trivial(variant):
    out = render(variant, default_scene(variant))

    assert torch.isfinite(out.color).all()
    covered = (out.color > 0).any(dim=0).float().mean().item()
    assert covered > 0.05, f"only {covered:.3%} of pixels covered"
    assert out.color.max().item() > 0.05
    assert 0.0 <= out.color.min().item()
    assert (out.radii > 0).sum().item() > 100, "too few visible surfels"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_auxiliary_outputs_are_plausible(variant):
    """The depth/alpha/normal/distortion buffer every variant renders alongside."""
    scene = default_scene(variant)
    out = render(variant, scene)
    height = scene["settings"]["image_height"]
    width = scene["settings"]["image_width"]

    assert out.others.shape == (7, height, width)
    assert torch.isfinite(out.others).all()

    alpha = out.others[ALPHA]
    assert alpha.min().item() >= 0.0 and alpha.max().item() <= 1.0
    assert alpha.max().item() > 0.5, "nothing is opaquely covered"

    # Depth is accumulated against alpha rather than normalised, so it is the
    # ratio that has to land inside the depth range the scene occupies.
    depth = out.others[DEPTH]
    assert depth.min().item() >= 0.0
    opaque = alpha > 0.9
    assert opaque.any()
    ratio = depth[opaque] / alpha[opaque]
    assert ratio.min().item() > 0.9 * SCENE_Z[0], f"nearest depth {ratio.min().item():.3f}"
    assert ratio.max().item() < 1.1 * SCENE_Z[1], f"farthest depth {ratio.max().item():.3f}"

    median = out.others[MIDDEPTH]
    assert median.min().item() >= 0.0 and median.max().item() < 1.1 * SCENE_Z[1]

    lengths = out.others[NORMAL].norm(dim=0)
    assert lengths.max().item() < 1.01, f"normal of length {lengths.max().item():.4f}"
    assert lengths.max().item() > 0.1, "no surfel normal was accumulated"

    distortion = out.others[DISTORTION]
    assert distortion.min().item() > -1e-6, "distortion is a sum of squares"
    assert distortion.max().item() > 0.0


@pytest.mark.parametrize("variant", [v for v in VARIANTS if VARIANTS[v][3] != "base"])
def test_out_weight_is_plausible(variant):
    """The wet variants accumulate each surfel's compositing weight over pixels.

    The scene every other test uses puts all of its surfels inside the frustum,
    so a copy of it moves a few of them behind the camera to reach the culling
    path: preprocess then leaves radii at 0 and never bins them, and since
    out_weight is allocated zero-filled and only ever written by atomicAdd from
    the compositing loop, their rows are the one place where a stale or
    uninitialised entry would surface.
    """
    scene = dict(default_scene(variant))
    means3d = scene["means3D"].clone()
    means3d[:CULLED_SURFELS, 2] = -1.0
    scene["means3D"] = means3d
    out = render(variant, scene)

    assert out.weight is not None, f"{variant} returned no weight tensor"
    assert out.weight.shape == (scene["means3D"].shape[0], 1)
    assert torch.isfinite(out.weight).all()
    assert out.weight.min().item() >= 0.0
    assert out.weight.sum().item() > 0.0
    visible = (out.radii > 0)
    assert visible.any()
    contributing = (out.weight[:, 0][visible] > 0).float().mean().item()
    assert contributing > 0.5, f"only {contributing:.3%} of visible surfels contributed"

    culled = ~visible
    culled_count = int(culled.sum().item())
    assert culled_count > 0, "nothing was culled, so the weight check below is vacuous"
    assert culled_count >= CULLED_SURFELS, f"only {culled_count} surfels were culled"
    assert (out.weight[:, 0][culled] == 0).all(), "a culled surfel accumulated weight"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_forward_is_deterministic(variant):
    scene = default_scene(variant)
    first = render(variant, scene)
    second = render(variant, scene)
    assert torch.equal(first.color, second.color)
    assert torch.equal(first.others, second.others)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_backward_gradients_are_finite(variant):
    out = render(variant, default_scene(variant), requires_grad=True)
    _, loss = weighted_loss(out.color)
    loss.backward()

    for name in ("means3D", "means2D", "opacities", "scales", "rotations", "colors_precomp"):
        grad = out.args[name].grad
        assert grad is not None, f"{name} has no gradient"
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum().item() > 0.0, f"{name} gradient is identically zero"


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_opacity_finite_difference(variant):
    """The decisive gradient check for an alpha-compositing kernel.

    The perturbation direction is non-negative on purpose. A sign-balanced one
    makes the directional derivative of thousands of surfels cancel to a small
    difference of large numbers, which buys a check an order of magnitude looser
    for the same two renders. With this direction the central difference tracks
    the analytic gradient to about a percent from a step of 1e-5 upwards; the
    step here is 1e-3, far enough above the noise floor and still small enough
    that the second-order term stays inside the band (measured 1.006-1.010 over
    all fourteen variants on an AMD Instinct MI300X).
    """
    scene = default_scene(variant)
    out = render(variant, scene, requires_grad=True)
    weights, loss = weighted_loss(out.color)
    loss.backward()
    analytic = out.args["opacities"].grad.clone()

    gen = torch.Generator(device="cpu").manual_seed(11)
    direction = torch.rand(scene["opacities"].shape, generator=gen).to(DEVICE)
    eps = 1e-3
    losses = []
    for sign in (+1.0, -1.0):
        perturbed = dict(scene)
        perturbed["opacities"] = scene["opacities"] + sign * eps * direction
        losses.append((render(variant, perturbed).color * weights).sum().item())

    finite_difference = (losses[0] - losses[1]) / (2.0 * eps)
    predicted = (analytic * direction).sum().item()

    assert abs(predicted) > 1e-3, "test direction is orthogonal to the gradient"
    slope = finite_difference / predicted
    assert 0.97 < slope < 1.03, f"finite difference/analytic slope {slope:.4f}"


@pytest.mark.parametrize("variant", [v for v in VARIANTS
                                     if VARIANTS[v][1] == 3 and VARIANTS[v][2] != 1])
def test_spherical_harmonics_colours(variant):
    """The colour-from-SH path, which a caller reaches by not precomputing.

    Only the three-channel variants can take it: the kernel sizes its colour
    scratch buffer for three channels whatever NUM_CHANNELS is.
    """
    scene = add_spherical_harmonics(scene_for(variant))
    out = render(variant, scene, requires_grad=True, use_sh=True)

    assert torch.isfinite(out.color).all()
    covered = (out.color > 0).any(dim=0).float().mean().item()
    assert covered > 0.05, f"only {covered:.3%} of pixels covered"
    assert out.color.min().item() >= 0.0

    # Band 0 reproduces the precomputed colours, so anything but a flat image
    # means the higher bands were evaluated.
    precomputed = render(variant, scene).color
    assert (out.color - precomputed).abs().max().item() > 1e-3, "higher SH bands did nothing"

    _, loss = weighted_loss(out.color)
    loss.backward()
    for name in ("shs", "means3D", "opacities", "scales", "rotations"):
        grad = out.args[name].grad
        assert torch.isfinite(grad).all(), f"{name} gradient is not finite"
        assert grad.abs().sum().item() > 0.0, f"{name} gradient is identically zero"


@pytest.mark.parametrize("variant", [v for v in VARIANTS if VARIANTS[v][2] != 1])
def test_first_three_channels_match_base(variant):
    """Every variant composites channels 0-2 identically to the base variant.

    The auxiliary buffer is compared as well, and it is the stricter half of the
    check: depth, alpha, normals and distortion are functions of the geometry
    alone, so the channel count cannot legitimately change them.

    Compared with a tolerance rather than exactly, because clang(HIP) contracts
    FMAs more aggressively than nvcc and the channel count changes the order of
    the per-pixel loads.
    """
    if variant == BASE:
        pytest.skip("this variant is the reference")
    reference = render(BASE, scene_for(BASE))
    out = render(variant, scene_for(variant))

    assert out.color.shape[1:] == reference.color.shape[1:]
    torch.testing.assert_close(out.color[:3], reference.color[:3], rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out.others, reference.others, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("variant", [v for v in VARIANTS
                                     if VARIANTS[v][3] != "base" and v != WET])
def test_out_weight_matches_wet_reference(variant):
    """Compositing weights are colour-independent, so every wet variant agrees.

    Not bit-identical: the weights are accumulated with atomicAdd across pixels,
    so the summation order is not reproducible between two grid shapes.
    """
    reference = render(WET, scene_for(WET))
    out = render(variant, scene_for(variant))
    torch.testing.assert_close(out.weight, reference.weight, rtol=1e-4, atol=1e-5)


def pixel_to_world(px, py, z, width, height):
    """World-space point that projects onto pixel centre (px, py) at depth z.

    Inverts ndc2Pix (auxiliary.h) against the projection built above, so a test
    can put a surfel at a chosen sub-pixel offset.
    """
    tanfovx, tanfovy = field_of_view(width, height)
    ndc_x = (2.0 * px + 1.0) / width - 1.0
    ndc_y = (2.0 * py + 1.0) / height - 1.0
    return [ndc_x * tanfovx * z, ndc_y * tanfovy * z, z]


def single_surfel_scene(px, py, z=3.0, scale=0.01, opacity=0.99, width=32, height=24):
    return {
        "means3D": torch.tensor([pixel_to_world(px, py, z, width, height)], device=DEVICE),
        "means2D": torch.zeros(1, 3, device=DEVICE),
        "opacities": torch.full((1, 1), opacity, device=DEVICE),
        "scales": torch.full((1, 2), scale, device=DEVICE),
        "rotations": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
        "colors_precomp": torch.ones(1, 3, device=DEVICE),
        "settings": settings_for(3, width, height),
    }


@pytest.mark.parametrize("offset,differs", [(0.0, False), (0.25, False), (0.5, False),
                                            (0.8, True), (0.95, True)])
def test_tile1_drops_the_truncated_boundary_pixel(offset, differs):
    """Where the 1x1 tiling differs from the 16x16 one, and why.

    Binning truncates: getRect (auxiliary.h) computes the exclusive upper bound
    as (p + radius + BLOCK - 1) / BLOCK in integer arithmetic, which with
    BLOCK_X = 1 is floor(p.x + radius), and the emit loop in rasterizer_impl.cu
    is half-open. So the 1x1 tiling never bins a surfel into the pixel column at
    floor(p.x + radius), which sits only radius - frac(p.x) away from the centre
    -- close enough to be well above the 1/255 alpha cutoff once frac(p.x) is
    large. A 16x16 tiling rounds the same expression up to a whole tile and
    keeps the column. The same holds per row.

    So the difference tracks the sub-pixel placement of the centre, not the size
    of the surfel: this scene's surfel spans the same three pixels either way.
    Both are the same algorithm binned differently, and a CUDA build truncates
    exactly as this one does.
    """
    scene = single_surfel_scene(16.0 + offset, 12.0 + offset)
    reference = render(BASE, scene)
    out = render(TILE1, scene)

    radius = reference.radii.item()
    assert radius > 0, "the test surfel is not visible"
    difference = out.color - reference.color
    if not differs:
        assert difference.abs().max().item() == 0.0, "expected an identical image"
        return

    assert difference.abs().max().item() > 1e-3, "expected the boundary column to differ"
    assert (difference <= 0).all(), "1x1 tiling can only drop contributions"
    dropped_column = math.floor(16.0 + offset + radius)
    dropped_row = math.floor(12.0 + offset + radius)
    for row, column in (difference.abs() > 0).any(dim=0).nonzero().tolist():
        assert row == dropped_row or column == dropped_column, \
            f"pixel ({row}, {column}) differs but is not on the truncated boundary"


def test_tile1_matches_base():
    """Over a whole scene the two tilings agree in bulk.

    They are not bit-identical, for the truncation reason above, so the check is
    agreement in bulk rather than element-wise equality. Small image because one
    pixel per block is pathologically slow.
    """
    small = dict(width=32, height=24, num_points=1500)
    reference = render(BASE, scene_for(BASE, **small))
    out = render(TILE1, scene_for(TILE1, **small))

    difference = (out.color - reference.color).abs()
    correlation = torch.corrcoef(
        torch.stack([out.color.flatten(), reference.color.flatten()]))[0, 1]
    assert difference.mean().item() < 1e-3, f"mean |difference| {difference.mean().item():.2e}"
    assert difference.max().item() < 0.02, f"max |difference| {difference.max().item():.2e}"
    assert correlation.item() > 0.9999, f"correlation {correlation.item():.6f}"


@pytest.mark.parametrize("variant,twin", [
    ("diff-surfel-rasterization-wet-abs", "diff-surfel-rasterization-wet"),
    ("diff-surfel-rasterization-wet-abs-ch05", "diff-surfel-rasterization-wet-ch05"),
])
def test_wet_abs_gradient_columns(variant, twin):
    """Columns 2-3 of dL_dmeans2D carry the homodirectional absolute gradient."""
    def means2d_grad(name):
        out = render(name, scene_for(name), requires_grad=True)
        _, loss = weighted_loss(out.color)
        loss.backward()
        return out.args["means2D"].grad

    abs_grad = means2d_grad(variant)
    ref_grad = means2d_grad(twin)

    assert abs_grad.shape[1] == 4 and ref_grad.shape[1] == 3
    assert (abs_grad[:, 2:4] >= 0).all(), "absolute gradient must be non-negative"
    assert abs_grad[:, 2:4].abs().sum().item() > 0.0
    torch.testing.assert_close(abs_grad[:, 0:2], ref_grad[:, 0:2], rtol=RTOL, atol=ATOL)
