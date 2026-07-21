"""
generate_demo_frames.py
------------------------
Generates a realistic 15-20 frame DEMO SEQUENCE for presenting the project,
instead of just 3 static test frames.

The story it tells:
  Frames 1-6:   slope is stable -- boulder barely moves (sensor noise only)
  Frames 7-12:  slope starts drifting -- boulder creeps a few pixels per frame
                (small, gradual movement -- realistic pre-failure creep)
  Frames 13-16: movement accelerates sharply -- boulder shifts a lot per frame
                (this is where your alert system should start firing)

Upload these to the dashboard IN ORDER (frame_01.jpg, frame_02.jpg, ...) and
you'll see the movement-score chart stay flat, then climb, then spike --
which is a much stronger demo than a single before/after pair.

Run:  python generate_demo_frames.py
"""

import os
import numpy as np
import cv2

OUT_DIR = "data/demo_sequence"
NUM_FRAMES = 18


def make_base_slope(seed=7, size=512):
    """Rocky-textured background, shared by every frame."""
    rng = np.random.RandomState(seed)
    base = (rng.rand(size, size) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (3, 3), 0)
    # add a few static "fixed" rocks so the frame isn't just one boulder
    for _ in range(6):
        cx, cy = rng.randint(50, size - 50, size=2)
        r = rng.randint(15, 30)
        cv2.circle(base, (cx, cy), r, int(rng.randint(50, 90)), -1)
    return base


def draw_moving_boulder(base, center, radius=35, color=25):
    out = base.copy()
    cv2.circle(out, center, radius, color, -1)
    # small highlight so it reads as a rounded rock, not a flat circle
    cv2.circle(out, (center[0] - 10, center[1] - 10), radius // 3, color + 40, -1)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = make_base_slope()

    start_pos = np.array([180, 260])
    boulder_pos = start_pos.astype(float)

    rng = np.random.RandomState(42)

    for i in range(1, NUM_FRAMES + 1):
        if i <= 6:
            # Phase 1: stable -- negligible drift, just noise
            drift = rng.uniform(-0.5, 0.5, size=2)
        elif i <= 12:
            # Phase 2: gradual creep -- consistent small movement in one direction
            drift = np.array([4.5, 1.2]) + rng.uniform(-0.6, 0.6, size=2)
        else:
            # Phase 3: accelerating shift -- clear pre-failure movement,
            # large enough to cross ANOMALY_THRESHOLD (0.15) and fire alerts
            drift = np.array([38.0, 12.0]) + rng.uniform(-3.0, 3.0, size=2)

        boulder_pos += drift
        center = (int(boulder_pos[0]), int(boulder_pos[1]))

        frame = draw_moving_boulder(base, center)

        # per-frame sensor noise, applied to every frame including stable ones
        noise = rng.randint(-3, 3, base.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filename = os.path.join(OUT_DIR, f"frame_{i:02d}.jpg")
        cv2.imwrite(filename, frame)

    print(f"Generated {NUM_FRAMES} frames in {OUT_DIR}/")
    print("Upload them in order (frame_01.jpg through frame_18.jpg) via the dashboard.")
    print("Expected story: flat scores for frames 1-6, a gentle rise for 7-12, "
          "then a sharp spike (alerts firing) for 13-18.")


if __name__ == "__main__":
    main()
