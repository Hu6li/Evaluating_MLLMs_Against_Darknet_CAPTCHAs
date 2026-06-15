#!/usr/bin/env python3
"""
Shared helpers for polar gradient cross-correlation puzzle solvers.

Imported by both polarGradientValidator (grayscale).
"""

from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np


def create_clean_composite(
        inner_path: str,
        outer_path: str,
        output_path: Optional[str] = None,
) -> np.ndarray:
    """
    Composite inner disc onto outer ring using a hard alpha threshold.

    Each source image carries a smooth anti-aliased alpha channel.  A soft
    blend produces a transition zone at the seam that introduces errors in
    the ring-signal extraction.  Using a hard threshold (inner_alpha >= 128
    → inner BGR, else outer BGR) creates a clean one-pixel boundary with no
    blending, so the grayscale ring signals on either side of the seam are
    uncontaminated.

    If output_path is given the result is saved as a lossless PNG.
    Returns the composited BGR image as a numpy array.
    """
    inner_bgra = _load_bgra(inner_path)
    outer_bgra = _load_bgra(outer_path)

    mask = inner_bgra[:, :, 3] >= 128          # True where inner disc lives
    composite = outer_bgra[:, :, :3].copy()
    composite[mask] = inner_bgra[:, :, :3][mask]

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), composite)

    return composite


def _load_bgra(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.ndim == 2:
        b = img
        img = cv2.merge([b, b, b, np.full_like(b, 255)])
    elif img.shape[2] == 3:
        b, g, r = cv2.split(img)
        img = cv2.merge([b, g, r, np.full_like(b, 255)])
    return img


def _estimate_seam_radius(
        inner_alpha: np.ndarray, outer_alpha: np.ndarray) -> float:
    """Estimate the seam radius from alpha masks."""
    h, w = inner_alpha.shape
    cy, cx = h // 2, w // 2
    ys, xs = np.indices((h, w))
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)

    inner_mask = inner_alpha > 128
    outer_mask = outer_alpha > 128

    if not np.any(inner_mask) or not np.any(outer_mask):
        return h // 4

    r_inner = dist[inner_mask].max()
    r_outer = dist[outer_mask].min()

    return (r_inner + r_outer) / 2.0


def _extract_ring_signal(channel: np.ndarray, cx: int, cy: int,
                         radius: float, band_width: int = 2,
                         n_samples: int = 360) -> np.ndarray:
    """
    Extract pixel values along a circular ring as a 1D signal.
    Samples at n_samples points around the circle.
    Averages over band_width radii for robustness.
    Uses bilinear interpolation so sub-pixel radius positions are smooth.
    """
    angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    signal = np.zeros(n_samples, dtype=np.float64)
    h, w = channel.shape

    for r_off in range(-band_width, band_width + 1):
        r = radius + r_off
        xs = cx + r * np.cos(angles)
        ys = cy + r * np.sin(angles)

        x0 = np.clip(np.floor(xs).astype(int), 0, w - 1)
        x1 = np.clip(x0 + 1,                   0, w - 1)
        y0 = np.clip(np.floor(ys).astype(int), 0, h - 1)
        y1 = np.clip(y0 + 1,                   0, h - 1)

        dx = xs - np.floor(xs)
        dy = ys - np.floor(ys)

        signal += (channel[y0, x0] * (1 - dx) * (1 - dy) +
                   channel[y0, x1] * dx * (1 - dy) +
                   channel[y1, x0] * (1 - dx) * dy +
                   channel[y1, x1] * dx * dy)

    return signal / (2 * band_width + 1)


def _gradient(signal: np.ndarray) -> np.ndarray:
    """Circular central-difference gradient magnitude."""
    n = len(signal)
    grad = np.empty(n, dtype=np.float64)
    for i in range(n):
        grad[i] = abs(signal[(i + 1) % n] - signal[(i - 1) % n]) / 2.0
    return grad


def _cross_correlate(sig1: np.ndarray, sig2: np.ndarray) -> np.ndarray:
    """
    Normalised circular cross-correlation via FFT.
    Returns a curve of length N; index k = correlation when sig2 is
    shifted by k samples relative to sig1.
    Returns all-zeros if either signal is flat (zero variance).
    """
    s1 = sig1 - sig1.mean()
    s2 = sig2 - sig2.mean()
    std1, std2 = s1.std(), s2.std()
    if std1 < 1e-6 or std2 < 1e-6:
        return np.zeros_like(sig1)
    s1 /= std1
    s2 /= std2
    return np.real(np.fft.ifft(np.fft.fft(s1) * np.conj(np.fft.fft(s2))))


def solve_rotation_single_image(
        image_path: str,
        seam_frac: float = 0.35,
        round_to_45: bool = False,
        n_samples: int = 360,
) -> Tuple[int, float, bool]:
    """
    Solve rotation from a single combined puzzle image using grayscale
    polar gradient cross-correlation.

    Identical algorithm to solve_rotation_polar_gradient.  The only
    difference is how inner/outer channels are obtained: instead of
    loading two separate alpha-masked images and estimating the seam
    from their alpha channels, a single combined image is loaded and
    the seam radius is derived from seam_frac * image_width
    (default 0.35 ≈ 70 px for 200×200 images).  Pixels at radius
    seam-2 fall within the inner disc; pixels at seam+2 fall within
    the outer ring.

    Returns:
        predicted_angle: Best rotation angle in degrees (0-359)
        confidence:      Correlation peak value (higher is better)
        is_ambiguous:    True if a competing peak exists
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2

    # ── 2. Convert to grayscale ───────────────────────────────────────────────
    channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── 3. Seam ───────────────────────────────────────────────────────────────
    seam_radius = seam_frac * w

    # ── 4. Sampling resolution ────────────────────────────────────────────────
    deg_per_sample = 360.0 / n_samples

    # ── 5. Extract ring signals at seam ──────────────────────────────────────
    # ── 5. Extract ring signals at seam ──────────────────────────────────────
    inner_ring = _extract_ring_signal(
        channel, cx, cy, seam_radius - 3, band_width=2, n_samples=n_samples)
    outer_ring = _extract_ring_signal(
        channel, cx, cy, seam_radius + 4, band_width=2, n_samples=n_samples)

    # ── 6. Gradient magnitude ─────────────────────────────────────────────────
    inner_grad = _gradient(inner_ring)
    outer_grad = _gradient(outer_ring)

    # ── 7. Circular cross-correlation via FFT ─────────────────────────────────
    correlation = _cross_correlate(inner_grad, outer_grad)

    # ── 8. Pick highest peak and detect ambiguity ─────────────────────────────
    best_shift, confidence, is_ambiguous = _find_peak(
        correlation, threshold_ratio=0.8)

    # ── 9. Convert shift to angle ─────────────────────────────────────────────
    predicted_angle = int(best_shift * deg_per_sample % 360)

    if round_to_45:
        predicted_angle = int(round(predicted_angle / 45.0) * 45) % 360

    predicted_angle = (360 - predicted_angle) % 360
    return predicted_angle, confidence, is_ambiguous


def _find_peak(correlation: np.ndarray,
               threshold_ratio: float = 0.8) -> Tuple[int, float, bool]:
    """
    Locate the strongest peak and flag ambiguity if a second peak is nearly
    as strong and well-separated (> n/36 samples ≈ 10°) from the best.
    """
    n = len(correlation)
    best_idx = int(np.argmax(correlation))
    best_val = float(correlation[best_idx])

    if best_val < 1e-6:
        return 0, 0.0, True

    threshold = best_val * threshold_ratio
    significant = []
    for i in range(n):
        if (correlation[i] >= threshold and
                correlation[i] >= correlation[(i - 1) % n] and
                correlation[i] >= correlation[(i + 1) % n]):
            sep = min(abs(i - best_idx), n - abs(i - best_idx))
            if sep > n // 36:
                significant.append(i)

    return best_idx, best_val, len(significant) > 0
