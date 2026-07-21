"""
generate_natural_frames.py
----------------------------
Generates a MUCH more realistic-looking rocky slope demo sequence than
generate_demo_frames.py -- uses layered fractal noise + natural rock
coloring + directional lighting instead of a flat gray background with
a plain circle "boulder". Still synthetic (no internet access available
to download real photos in this environment), but looks like an actual
rock face instead of a test pattern.

For genuinely real photos, download these public-domain USGS images
yourself and use them as your "risk" example frames:
  https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/s3fs-public/landslide.jpg
  https://www.usgs.gov/programs/landslide-hazards/multimedia/images

Run:  python generate_natural_frames.py
"""

import os
import numpy as np
import cv2

OUT_DIR = "data/natural_sequence"
NUM_FRAMES = 18
SIZE = 512


def fractal_noise(size, octaves=5, persistence=0.55, seed=0):
    """Layered Perlin-like noise built by summing upscaled random grids --
    gives natural-looking rock texture instead of flat random pixels."""
    rng = np.random.RandomState(seed)
    noise = np.zeros((size, size), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    grid_size = 4
    for _ in range(octaves):
        small = rng.rand(grid_size, grid_size).astype(np.float32)
        layer = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
        noise += layer * amplitude
        total_amplitude += amplitude
        amplitude *= persistence
        grid_size *= 2
    noise /= total_amplitude
    return noise


def make_rock_base(seed=11):
    """Builds a natural-looking rocky cliff face: layered noise, brown/grey
    rock coloring, directional lighting (darker toward one edge)."""
    texture = fractal_noise(SIZE, octaves=6, seed=seed)

    # Rock color palette: greyish-brown, not pure grey
    base_color = np.array([80, 95, 110], dtype=np.float64)   # BGR: cool grey-brown
    highlight = np.array([120, 130, 140], dtype=np.float64)  # lighter patches

    texture_norm = (texture - texture.min()) / (texture.max() - texture.min())
    img = base_color[None, None, :] + texture_norm[:, :, None] * (highlight - base_color)[None, None, :]

    # Directional lighting: simulate sunlight from top-left
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    light_gradient = 1.0 - 0.35 * ((xx + yy) / (2 * SIZE))
    img = img * light_gradient[:, :, None]

    # Add fine grain noise on top for realism (rock grain / sensor noise)
    grain = np.random.RandomState(seed + 1).normal(0, 6, (SIZE, SIZE, 1))
    img = img + grain

    # A few darker fracture lines (cracks) -- static, part of the base rock
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    rng = np.random.RandomState(seed + 2)
    for _ in range(4):
        pt1 = (rng.randint(0, SIZE), rng.randint(0, SIZE))
        pt2 = (pt1[0] + rng.randint(-150, 150), pt1[1] + rng.randint(-150, 150))
        cv2.line(img_u8, pt1, pt2, (50, 55, 60), thickness=rng.randint(1, 3))

    return img_u8


def make_boulder_patch(radius=38, seed=100):
    """
    Precompute the boulder's own texture ONCE, so across frames only its
    POSITION changes -- not its appearance. (Earlier version regenerated
    random texture every frame even for a "stationary" boulder, which
    made every frame look different regardless of real movement -- that
    washed out the stable/creep/shift story we want to demonstrate.)
    """
    patch_size = radius * 2 + 20
    texture = fractal_noise(patch_size, octaves=4, seed=seed)
    # Freshly-exposed/displaced rock tends to look noticeably lighter and
    # less weathered than the surrounding slope -- a real, documented visual
    # cue geologists look for. We lean into that here, which also gives the
    # detector a genuine signal to pick up on (not just an arbitrary color).
    boulder_color = np.array([120, 140, 158], dtype=np.float64)
    boulder_highlight = np.array([165, 180, 195], dtype=np.float64)
    tex_norm = (texture - texture.min()) / (texture.max() - texture.min())
    patch = boulder_color[None, None, :] + tex_norm[:, :, None] * (boulder_highlight - boulder_color)[None, None, :]

    # top-left highlight for a 3D look, baked into the fixed patch
    yy, xx = np.mgrid[0:patch_size, 0:patch_size]
    c = patch_size / 2
    dist = np.sqrt((xx - c * 0.7) ** 2 + (yy - c * 0.7) ** 2)
    highlight_strength = np.clip(1.0 - dist / (radius * 1.2), 0, 1) ** 2
    patch = patch + highlight_strength[:, :, None] * 35

    circle_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    cv2.circle(circle_mask, (int(c), int(c)), radius, 255, -1)

    return np.clip(patch, 0, 255).astype(np.uint8), circle_mask


def draw_boulder(base, center, boulder_patch, boulder_mask, radius=38):
    """Pastes the FIXED boulder texture patch onto the base at `center`."""
    out = base.copy()
    patch_size = boulder_patch.shape[0]
    half = patch_size // 2

    x0, y0 = center[0] - half, center[1] - half
    x1, y1 = x0 + patch_size, y0 + patch_size

    # clip to image bounds
    bx0, by0 = max(x0, 0), max(y0, 0)
    bx1, by1 = min(x1, SIZE), min(y1, SIZE)
    px0, py0 = bx0 - x0, by0 - y0
    px1, py1 = px0 + (bx1 - bx0), py0 + (by1 - by0)

    if bx1 <= bx0 or by1 <= by0:
        return out  # boulder fully off-frame

    mask_region = boulder_mask[py0:py1, px0:px1]
    patch_region = boulder_patch[py0:py1, px0:px1]
    mask_3ch = cv2.merge([mask_region, mask_region, mask_region]) / 255.0

    out_region = out[by0:by1, bx0:bx1]
    blended = (out_region * (1 - mask_3ch) + patch_region * mask_3ch).astype(np.uint8)
    out[by0:by1, bx0:bx1] = blended

    # soft shadow, grounding the boulder
    shadow_mask_full = np.zeros((SIZE, SIZE), dtype=np.uint8)
    cv2.circle(shadow_mask_full, (center[0] + 6, center[1] + 8), radius, 255, -1)
    full_boulder_mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    full_boulder_mask[by0:by1, bx0:bx1] = mask_region
    shadow_mask_full = cv2.subtract(shadow_mask_full, full_boulder_mask)
    shadow_mask_full = cv2.GaussianBlur(shadow_mask_full, (9, 9), 0)
    shadow_3ch = cv2.merge([shadow_mask_full, shadow_mask_full, shadow_mask_full]) / 255.0 * 0.3
    out = (out * (1 - shadow_3ch)).astype(np.uint8)

    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_rock_base()
    boulder_patch, boulder_mask = make_boulder_patch(radius=48, seed=100)

    start_pos = np.array([150, 260])
    boulder_pos = start_pos.astype(float)
    rng = np.random.RandomState(42)

    for i in range(1, NUM_FRAMES + 1):
        if i <= 6:
            drift = rng.uniform(-0.5, 0.5, size=2)          # Phase 1: stable
        elif i <= 12:
            drift = np.array([5.0, 1.5]) + rng.uniform(-0.6, 0.6, size=2)   # Phase 2: creep
        else:
            drift = np.array([45.0, 13.0]) + rng.uniform(-3.0, 3.0, size=2)  # Phase 3: shift

        boulder_pos += drift
        center = (int(boulder_pos[0]), int(boulder_pos[1]))

        frame = draw_boulder(base, center, boulder_patch, boulder_mask, radius=48)

        # subtle per-frame lighting variation (clouds passing) -- smoothed,
        # not raw per-pixel noise, so it doesn't fool optical flow/SSIM into
        # reading it as movement (real sensor noise is spatially correlated,
        # not independent per pixel)
        variation = rng.randint(-3, 3, (SIZE // 8, SIZE // 8, 3)).astype(np.float32)
        variation = cv2.resize(variation, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
        frame = np.clip(frame.astype(np.float32) + variation, 0, 255).astype(np.uint8)

        filename = os.path.join(OUT_DIR, f"frame_{i:02d}.jpg")
        cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"Generated {NUM_FRAMES} natural-looking rock texture frames in {OUT_DIR}/")
    print("These use layered fractal noise + rock coloring + lighting -- ")
    print("much closer to a real rock face than flat synthetic circles.")
    print("Upload frame_01.jpg through frame_18.jpg in order via the dashboard.")


if __name__ == "__main__":
    main()
