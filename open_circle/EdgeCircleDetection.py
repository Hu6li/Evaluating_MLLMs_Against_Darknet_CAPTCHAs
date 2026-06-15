from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import cv2
from PIL import Image
from time import sleep


@dataclass
class OpenCircleResult:
    point: Tuple[int, int]
    circle: Tuple[float, float, float]  # (cx, cy, r)
    score: float
    debug: Optional[Dict[str, Any]] = None


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def _pil_to_gray_np(img: Image.Image) -> np.ndarray:
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    arr = np.array(img)
    if arr.ndim == 2:
        return arr.astype(np.uint8)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _make_panel(panels: list, target_h: int = 200) -> np.ndarray:
    """
    Build a horizontal comparison strip.

    panels  : list of (title: str, img: np.ndarray)  — img may be grayscale or BGR
    target_h: all cells are scaled to this height (aspect ratio preserved)
    Returns a single BGR image.
    """
    cells = []
    for title, img in panels:
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            bgr = img.copy()
        h, w = bgr.shape[:2]
        new_w = max(1, int(w * target_h / h))
        cell = cv2.resize(bgr, (new_w, target_h))
        bar = np.zeros((24, new_w, 3), dtype=np.uint8)
        cv2.putText(bar, title, (2, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cells.append(np.vstack([bar, cell]))
    return np.hstack(cells)


def _largest_structured_gap(edges: np.ndarray, cx: float, cy: float, r: float):
    """
    Returns:
        largest_gap_fraction
        total_perimeter_support
        thetas            — angular sample positions (np.ndarray)
        gap_start_idx     — index into thetas where the largest gap begins
        gap_length        — number of sample points in the largest gap
    """

    h, w = edges.shape[:2]

    # Angular sampling proportional to radius; remove the 1000-point floor
    # that oversampled small circles (r≈10 → 15× oversampling at 1000 pts).
    num_points = max(360, int(3 * np.pi * r))
    thetas = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

    flags = np.zeros(num_points, dtype=np.float32)

    # Sample at +-3 the center detection is not such accurate ...
    for offset in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
        rr = r + offset
        for i, theta in enumerate(thetas):
            x = int(round(cx + rr * np.cos(theta)))
            y = int(round(cy + rr * np.sin(theta)))
            if 0 <= x < w and 0 <= y < h and edges[y, x] > 0:
                flags[i] = 1.0

    total_support = flags.mean()

    # Smooth angular signal — kernel spans ~12° regardless of circle size
    kernel_size = max(5, int(num_points * 12 / 360))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = np.ones(kernel_size) / kernel_size
    smooth = np.convolve(np.concatenate([flags, flags]), kernel, mode="same")
    smooth = smooth[:num_points]

    binary = smooth > 0.3

    # Measure largest contiguous gap; track start position
    doubled = np.concatenate([binary, binary])
    max_gap = 0
    current = 0
    current_start = 0
    gap_start_idx = 0

    for idx, val in enumerate(doubled):
        if not val:
            if current == 0:
                current_start = idx
            current += 1
        else:
            if current > max_gap:
                max_gap = current
                gap_start_idx = current_start
            current = 0

    # Handle gap that runs to the end of doubled
    if current > max_gap:
        max_gap = current
        gap_start_idx = current_start

    max_gap = min(max_gap, num_points)
    gap_start_idx = gap_start_idx % num_points

    largest_gap_fraction = max_gap / num_points

    return largest_gap_fraction, total_support, thetas, gap_start_idx, max_gap, flags


def _gap_is_dark(
    gray: np.ndarray,
    cx: float, cy: float, r: float,
    thetas: np.ndarray,
    gap_start_idx: int,
    gap_length: int,
    bg_thresh: int = 40,
    bright_fraction_max: float = 0.30,
) -> bool:
    """
    Sample gray pixel values at the gap's angular region.

    Returns True  — mostly dark → real background gap (keep candidate).
    Returns False — mostly bright → ink present, Canny just missed edges (reject).
    """
    h, w = gray.shape[:2]
    n = len(thetas)
    bright = total = 0
    for offset in range(gap_start_idx, gap_start_idx + gap_length):
        i = offset % n
        x = int(round(cx + r * np.cos(thetas[i])))
        y = int(round(cy + r * np.sin(thetas[i])))
        if 0 <= x < w and 0 <= y < h:
            if gray[y, x] > bg_thresh:
                bright += 1
            total += 1
    if total == 0:
        return False
    return (bright / total) < bright_fraction_max


# ------------------------------------------------------------
# Main detection
# ------------------------------------------------------------

def detect_open_circle_point(
    pil_img: Image.Image,
    *,
    dp: float = 1.2,
    min_dist: Optional[float] = 8,
    canny_thresh1: int = 40,
    canny_thresh2: int = 30,
    hough_param2: float = 15.0,
    min_radius: Optional[int] = 10,
    max_radius: Optional[int] = 50,
    return_debug: bool = False,
) -> OpenCircleResult:


    gray = _pil_to_gray_np(pil_img)
    h, w = gray.shape[:2]

    if min_dist is None:
        min_dist = min(h, w) * 0.05

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2)

    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=float(canny_thresh2),
        param2=float(hough_param2),
        minRadius=0 if min_radius is None else int(min_radius),
        maxRadius=0 if max_radius is None else int(max_radius),
    )
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if circles is None:
        raise ValueError("No circles detected. Adjust Hough parameters.")

    circles = circles[0]

    best = None
    all_scores = []

    for (cx, cy, r) in circles:

        gap, support, thetas, gap_start, gap_len, flags = _largest_structured_gap(edges, cx, cy, r)

        # add centers
        cv2.circle(edges_color, (int(cx), int(cy)), 2, (0, 0, 255), 3)

        # Must still look like a circle
        if support < 0.75:
            continue

        # add edges
        for i, theta in enumerate(thetas):
            x = int(round(cx + r * np.cos(theta)))
            y = int(round(cy + r * np.sin(theta)))
            if flags[i]:
                cv2.circle(edges_color, (x,y), 1, (0, 255, 0), 3)
        # Reject candidates whose "gap" is actually bright ink (Canny failure)
        #if not _gap_is_dark(gray, cx, cy, r, thetas, gap_start, gap_len):
        #    continue

        # Weight gap by arc coverage squared: rewards circles with both a
        # clear gap AND strong evidence along the rest of the arc.
        score = gap * support ** 2

        all_scores.append((score, (cx, cy, r)))

        if best is None or score > best[0]:
            best = (score, cx, cy, r)


    if best is None:
        raise ValueError("No suitable open circle found.")

    best_score, best_cx, best_cy, best_r = best
    point = (int(round(best_cx)), int(round(best_cy)))

    debug = None
    if return_debug:
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for s, (cx, cy, r) in all_scores:
            cv2.circle(vis, (int(round(cx)), int(round(cy))), int(round(r)), (0, 255, 255), 1)
            cv2.putText(
                vis, f"{s:.2f}",
                (int(cx) - 10, int(cy) - int(r) - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, (160, 160, 160), 1,
            )
        # Winner: green circle + red centre dot
        cv2.circle(vis, point, int(round(best_r)), (0, 255, 0), 2)
        cv2.circle(vis, point, 4, (0, 0, 255), -1)

        debug = {
            "edges":  edges_color,
            "vis":    vis,
            "scores": all_scores,
        }

    return OpenCircleResult(
        point=point,
        circle=(float(best_cx), float(best_cy), float(best_r)),
        score=float(best_score),
        debug=debug,
    )


# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Open circle detection — Hough + gap (v2)")
    parser.add_argument("--batch", action="store_true",
                        help="Process all images in input directory")
    parser.add_argument("--input-dir",  default="../samples")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--image",      default=None,
                        help="Single image path")
    parser.add_argument("--samples",    nargs="+", type=int, default=None,
                        help="Image stem numbers for debug comparison (e.g. --samples 1 5 30)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)

    def _process_one(img_path: Path) -> None:
        out = Path("debug/ECD") / img_path.stem
        out.mkdir(parents=True, exist_ok=True)
        try:
            img = Image.open(img_path)
            res = detect_open_circle_point(img, return_debug=True)
        except Exception as e:
            print(f"  [{img_path.stem}] FAILED: {e}")
            return
        d = res.debug
        print(f"  [{img_path.stem}] centre={res.point}  r={res.circle[2]:.1f}  gap={res.score:.3f}")
        rgb_bgr = cv2.cvtColor(
            np.array(img.convert("RGB"), dtype=np.uint8), cv2.COLOR_RGB2BGR
        )
        cv2.imwrite(str(out / "01_original.png"), rgb_bgr)
        cv2.imwrite(str(out / "02_edges.png"),    d["edges"])
        cv2.imwrite(str(out / "03_vis.png"),      d["vis"])
        panel = _make_panel([
            ("Original", rgb_bgr),
            ("Edges",    d["edges"]),
            ("Result",   d["vis"]),
        ])
        cv2.imwrite(str(out / "panel.png"), panel)
        print(f"           → {out}/")

    if args.batch:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(exist_ok=True)
        image_files = sorted(
            [f for f in in_dir.iterdir()
             if f.suffix.lower() in (".jpg", ".jpeg", ".png")],
            key=lambda f: (0, int(f.stem)) if f.stem.isdigit() else (1, f.stem)
        )
        print(f"Processing {len(image_files)} images → {out_dir}")
        print("-" * 60)
        n_success = n_failed = 0
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                res = detect_open_circle_point(img)
                gray = _pil_to_gray_np(img)
                vis  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.circle(vis, res.point, int(round(res.circle[2])), (0, 255, 0), 2)
                cv2.circle(vis, res.point, 4, (0, 0, 255), -1)
                cv2.imwrite(str(out_dir / f"{img_path.stem}_detected.png"), vis)
                print(f"[+] {img_path.name}: center={res.point}, r={res.circle[2]:.1f}, gap={res.score:.3f}")
                n_success += 1
            except Exception as e:
                print(f"[-] {img_path.name}: {str(e)[:80]}")
                n_failed += 1
        print("-" * 60)
        print(f"OK={n_success}  FAIL={n_failed}  → {out_dir}/")

    elif args.samples:
        print(f"ECD debug — {len(args.samples)} sample(s):")
        for s in args.samples:
            candidates = [f for f in in_dir.iterdir() if f.stem == str(s)]
            if candidates:
                _process_one(candidates[0])
            else:
                print(f"  [{s}] not found in {in_dir}")

    else:
        _process_one(Path(args.image) if args.image else in_dir / "30.jpg")
