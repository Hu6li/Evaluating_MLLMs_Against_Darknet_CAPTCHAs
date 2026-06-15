#!/usr/bin/env python3
"""
Grayscale polar gradient cross-correlation puzzle rotation solver.

Converts both images to grayscale, extracts a ring signal at the seam,
computes the angular gradient, and uses FFT cross-correlation to find
the rotation that best aligns inner with outer.
"""

from typing import Tuple
import cv2
import numpy as np

from polarGradientCore import (
    _load_bgra, _estimate_seam_radius, _extract_ring_signal,
    _gradient, _cross_correlate, _find_peak,
)


def solve_rotation_polar_gradient(inner_path: str, outer_path: str,
                                  round_to_45: bool = False,
                                  n_samples: int = 360) -> Tuple[int, float, bool]:
    """
    Solve rotation using grayscale polar gradient cross-correlation.

    Returns:
        predicted_angle: Best rotation angle in degrees (0-359)
        confidence:      Correlation peak value (higher is better)
        is_ambiguous:    True if a competing peak exists
    """
    # ── 1. Load ───────────────────────────────────────────────────────────────
    inner_bgra = _load_bgra(inner_path)
    outer_bgra = _load_bgra(outer_path)

    h, w = inner_bgra.shape[:2]
    cx, cy = w // 2, h // 2

    # ── 2. Convert to grayscale ───────────────────────────────────────────────
    inner_channel = cv2.cvtColor(inner_bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)
    outer_channel = cv2.cvtColor(outer_bgra[:, :, :3], cv2.COLOR_BGR2GRAY).astype(np.float64)

    # ── 3. Seam ───────────────────────────────────────────────────────────────
    seam_radius = _estimate_seam_radius(inner_bgra[:, :, 3], outer_bgra[:, :, 3])

    # ── 4. Sampling resolution ────────────────────────────────────────────────
    deg_per_sample = 360.0 / n_samples

    # ── 5. Extract ring signals at seam ──────────────────────────────────────
    inner_ring = _extract_ring_signal(inner_channel, cx, cy, seam_radius - 2, band_width=2, n_samples=n_samples)
    outer_ring = _extract_ring_signal(outer_channel, cx, cy, seam_radius + 2, band_width=2, n_samples=n_samples)

    # ── 6. Gradient magnitude ─────────────────────────────────────────────────
    inner_grad = _gradient(inner_ring)
    outer_grad = _gradient(outer_ring)

    # ── 7. Circular cross-correlation via FFT ─────────────────────────────────
    correlation = _cross_correlate(inner_grad, outer_grad)

    # ── 8. Pick highest peak and detect ambiguity ─────────────────────────────
    best_shift, confidence, is_ambiguous = _find_peak(correlation, threshold_ratio=0.8)

    # ── 9. Convert shift to angle ─────────────────────────────────────────────
    # FFT peak lands at j = R / deg_per_sample, so R = best_shift * deg_per_sample
    predicted_angle = int(best_shift * deg_per_sample % 360)

    if round_to_45:
        predicted_angle = int(round(predicted_angle / 45.0) * 45) % 360

    predicted_angle = (360 - predicted_angle) % 360
    return predicted_angle, confidence, is_ambiguous
