
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

    for _ in range(6):
        cx, cy = rng.randint(50, size - 50, size=2)
        r = rng.randint(15, 30)
        cv2.circle(base, (cx, cy), r, int(rng.randint(50, 90)), -1)
    return base


def draw_moving_boulder(base, center, radius=35, color=25):
    out = base.copy()
    cv2.circle(out, center, radius, color, -1)

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
          
            drift = rng.uniform(-0.5, 0.5, size=2)
        elif i <= 12:
         
            drift = np.array([4.5, 1.2]) + rng.uniform(-0.6, 0.6, size=2)
        else:
       
            drift = np.array([38.0, 12.0]) + rng.uniform(-3.0, 3.0, size=2)

        boulder_pos += drift
        center = (int(boulder_pos[0]), int(boulder_pos[1]))

        frame = draw_moving_boulder(base, center)

        
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
