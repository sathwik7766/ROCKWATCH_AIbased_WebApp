"""
make_test_frames.py
--------------------
Regenerates the 3 synthetic test frames locally. Use this if data/frames/
is empty or missing on your machine (e.g. after copying the project via
OneDrive, which sometimes doesn't sync file contents properly).

Run:  python make_test_frames.py
"""

import numpy as np
import cv2
import os

os.makedirs("data/frames", exist_ok=True)
np.random.seed(1)

base = (np.random.rand(512, 512) * 255).astype(np.uint8)
base = cv2.GaussianBlur(base, (3, 3), 0)


def add_boulder(img, center, radius=40, color=30):
    out = img.copy()
    cv2.circle(out, center, radius, color, -1)
    return out


# Frame 1: boulder at (200, 250)
frame1 = add_boulder(base, (200, 250))
cv2.imwrite("data/frames/frame1.jpg", frame1)

# Frame 2: same position + tiny sensor noise (should be "stable")
noise = np.random.randint(-2, 2, base.shape).astype(np.int16)
frame2 = np.clip(frame1.astype(np.int16) + noise, 0, 255).astype(np.uint8)
cv2.imwrite("data/frames/frame2.jpg", frame2)

# Frame 3: boulder shifted 60px (should be flagged as movement)
frame3 = add_boulder(base, (260, 250))
cv2.imwrite("data/frames/frame3.jpg", frame3)

print("Test frames created in data/frames/")
print("Try: python movement_detector.py data/frames/frame1.jpg data/frames/frame2.jpg")
print("Try: python movement_detector.py data/frames/frame1.jpg data/frames/frame3.jpg")
