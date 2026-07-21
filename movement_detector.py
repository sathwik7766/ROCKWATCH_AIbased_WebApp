"""
Three techniques, from simplest to most informative:
  1. Frame differencing  -> fast, catches any pixel-level change
  2. SSIM (structural similarity) -> catches structural changes, ignores
     minor lighting noise better than raw differencing
  3. Optical flow (Farneback) -> estimates actual motion/displacement,
     which is the closest thing to "how much did the rock move"
"""

import cv2
import numpy as np


def load_gray(image_path, resize_to=(512, 512)):
    """Load an image as grayscale and resize for consistent comparison."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, resize_to)
    return gray


def frame_diff_score(prev_gray, curr_gray, blur_ksize=5):
    """
    Simple absolute difference between two frames.
    Blurring first reduces false positives from camera sensor noise.
    Returns: (score, diff_image)
    """
    prev_blur = cv2.GaussianBlur(prev_gray, (blur_ksize, blur_ksize), 0)
    curr_blur = cv2.GaussianBlur(curr_gray, (blur_ksize, blur_ksize), 0)

    diff = cv2.absdiff(prev_blur, curr_blur)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    score = float(np.count_nonzero(thresh)) / thresh.size
    return score, thresh


def ssim_score(prev_gray, curr_gray):
    """
    Structural Similarity Index between two frames.
    Returns dissimilarity score: 0 = identical, 1 = completely different.
    (We return 1 - SSIM so that, like the other functions, HIGHER = more change.)
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = prev_gray.astype(np.float64)
    img2 = curr_gray.astype(np.float64)

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    ssim_value = ssim_map.mean()
    return 1.0 - ssim_value  # higher = more change


def optical_flow_score(prev_gray, curr_gray):
    """
    Dense optical flow (Farneback method).
    Returns the average magnitude of motion vectors across the frame --
    this is the closest proxy to "how much did things actually move".
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(magnitude.mean())


def combined_movement_score(prev_path, curr_path, weights=(0.3, 0.4, 0.3)):
    """
    Combine all three signals into one normalized movement score.
    weights = (frame_diff_weight, ssim_weight, optical_flow_weight)
    """
    prev_gray = load_gray(prev_path)
    curr_gray = load_gray(curr_path)

    fd_score, _ = frame_diff_score(prev_gray, curr_gray)
    ss_score = ssim_score(prev_gray, curr_gray)
    of_score = optical_flow_score(prev_gray, curr_gray)


    of_score_norm = min(of_score / 5.0, 1.0)

    combined = (
        weights[0] * fd_score +
        weights[1] * ss_score +
        weights[2] * of_score_norm
    )
    return {
        "frame_diff": round(fd_score, 4),
        "ssim_dissimilarity": round(ss_score, 4),
        "optical_flow": round(of_score, 4),
        "combined_score": round(combined, 4),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python movement_detector.py <prev_frame.jpg> <curr_frame.jpg>")
        sys.exit(1)
    result = combined_movement_score(sys.argv[1], sys.argv[2])
    print(result)
