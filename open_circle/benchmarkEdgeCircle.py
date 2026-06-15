#!/usr/bin/env python3
"""

Usage
─────
    python3 benchmarkEdgeCircle.py           # all annotated images
    python3 benchmarkEdgeCircle.py 50        # first 50
    python3 benchmarkEdgeCircle.py --show benchmark_edge_circle_*.json
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from EdgeCircleDetection import detect_open_circle_point as detect


# ── Paths ─────────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent / "samples"
TRUTH_FILE  = Path(__file__).parent / "truth.json"

DETECTORS = [
    ("EdgeCircle", detect)
]


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_truth(path: Path = TRUTH_FILE) -> Tuple[Dict[str, Tuple[int, int]], int]:
    """Returns ({filename: (cx, cy)}, tolerance_px)."""
    with open(path) as f:
        data = json.load(f)
    tol = data.get("tolerance_px", 30)
    truth = {}
    for entry in data.get("images", []):
        if entry.get("center_x") is not None:
            truth[entry["file"]] = (entry["center_x"], entry["center_y"])
    return truth, tol


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dist(x1: int, y1: int, x2: int, y2: int) -> float:
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def _load_images(truth: Dict, n: int) -> List[Path]:
    files = sorted(
        [f for f in SAMPLES_DIR.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png") and f.name in truth],
        key=lambda f: (0, int(f.stem)) if f.stem.isdigit() else (1, f.stem)
    )
    return files[:n]


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(labels: List[str], stats: Dict, elapsed_times: Dict,
                 tol: int, W: int = 64) -> None:
    COL = 14

    def hdr(m):       return f"  {m:<22}" + "".join(f"{l:>{COL}}" for l in labels)
    def row(m, vals): return f"  {m:<22}" + "".join(f"{v:>{COL}}" for v in vals)
    def pct(v):       return f"{v * 100:.1f}%"
    def flt(v):       return f"{v:.1f} px"
    def itg(v):       return str(v)
    def ms(lst):      return f"{sum(lst)/len(lst):.0f} ms" if lst else "N/A"

    print("\n" + "=" * W)
    print("SUMMARY — EdgeCircleDetection CV benchmarks")
    print("=" * W)
    print(hdr("Metric"))
    print("  " + "-" * (22 + COL * len(labels)))
    print(row(f"Accuracy (≤{tol}px)",  [pct(stats[l]["accuracy"])          for l in labels]))
    print(row("Correct / total",       [f"{stats[l]['correct']}/{stats[l]['total']}" for l in labels]))
    print(row("Failed (exception)",    [itg(stats[l]["failed"])             for l in labels]))
    print()
    print(row("Avg dist correct",      [flt(stats[l]["avg_dist_correct"])   for l in labels]))
    print(row("Avg dist all",          [flt(stats[l]["avg_dist_all"])       for l in labels]))
    print()
    print(row("Avg time / image",      [ms(elapsed_times[l])                for l in labels]))
    print("=" * W)


# ── Main benchmark ────────────────────────────────────────────────────────────

def run_benchmark(n_samples: int) -> None:
    W = 64

    if not SAMPLES_DIR.exists():
        print(f"[!] samples not found: {SAMPLES_DIR}")
        sys.exit(1)
    if not TRUTH_FILE.exists():
        print(f"[!] truth.json not found: {TRUTH_FILE}")
        sys.exit(1)

    truth, tol = load_truth()
    image_files = _load_images(truth, n_samples)
    if not image_files:
        print("[!] No images found in truth set.")
        sys.exit(1)

    labels = [name for name, _ in DETECTORS]

    print("=" * W)
    print("Benchmark: EdgeCircleDetection")
    print("  Truth : truth.json (human-annotated)")
    print("=" * W)
    print(f"  Samples      : {len(image_files)} (of {n_samples} requested)")
    print(f"  Accept radius: {tol} px")
    print("=" * W)

    # {label: [(correct, dist), ...]}
    all_results:   Dict[str, List[Tuple[bool, float]]] = {l: [] for l in labels}
    elapsed_times: Dict[str, List[float]]              = {l: [] for l in labels}
    failed_cnt:    Dict[str, int]                      = {l: 0  for l in labels}

    for idx, img_path in enumerate(image_files, 1):
        ref_x, ref_y = truth[img_path.name]
        print(f"\n  [{idx:>3}] {img_path.name}  truth=({ref_x},{ref_y})")

        pil = Image.open(img_path)

        for label, detector in DETECTORS:
            try:
                t0 = time.perf_counter()
                result = detector(pil)
                elapsed_ms = (time.perf_counter() - t0) * 1000

                px, py = result.point
                dist    = _dist(px, py, ref_x, ref_y)
                correct = dist <= tol
                all_results[label].append((correct, dist))
                elapsed_times[label].append(elapsed_ms)

                mark = "✓" if correct else "✗"
                print(f"    {label:<14} {mark}  ({px:>4},{py:>4})  "
                      f"dist={dist:>6.1f}px  ({elapsed_ms:.0f}ms)")

            except Exception as e:
                print(f"    {label:<14} ✗  ERROR: {e}")
                all_results[label].append((False, float("inf")))
                elapsed_times[label].append(0.0)
                failed_cnt[label] += 1

    # ── Stats ─────────────────────────────────────────────────────────────────
    def compute(label: str) -> Dict:
        records = all_results[label]
        n       = len(records)
        correct = sum(1 for ok, _ in records if ok)
        dists   = [d for _, d in records if d != float("inf")]
        dists_c = [d for ok, d in records if ok]
        return {
            "total":            n,
            "correct":          correct,
            "accuracy":         correct / n if n else 0.0,
            "failed":           failed_cnt[label],
            "avg_dist_all":     sum(dists)   / len(dists)   if dists   else 0.0,
            "avg_dist_correct": sum(dists_c) / len(dists_c) if dists_c else 0.0,
        }

    stats = {l: compute(l) for l in labels}
    print_report(labels, stats, elapsed_times, tol, W)

    # ── Save ──────────────────────────────────────────────────────────────────
    output = {
        label: {
            "stats":   stats[label],
            "results": [
                {
                    "correct":    ok,
                    "dist":       d if d != float("inf") else None,
                    "elapsed_ms": elapsed_times[label][i],
                }
                for i, (ok, d) in enumerate(all_results[label])
            ],
            "elapsed_ms": elapsed_times[label],
        }
        for label in labels
    }
    out_file = f"benchmark_edge_circle_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[✓] Results saved to: {out_file}")


# ── --show: replay a saved result ────────────────────────────────────────────

def show_benchmark(path: str) -> None:
    print(f"[*] Loading: {path}")
    with open(path) as f:
        data = json.load(f)
    labels        = list(data.keys())
    stats         = {l: data[l]["stats"]      for l in labels}
    elapsed_times = {l: data[l]["elapsed_ms"] for l in labels}
    print_report(labels, stats, elapsed_times, tol=30)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = sys.argv[1:]

    if "--show" in args:
        idx = args.index("--show")
        if idx + 1 >= len(args):
            print("Usage: python3 benchmarkEdgeCircle.py --show <file.json>")
            sys.exit(1)
        show_benchmark(args[idx + 1])
        sys.exit(0)

    n = int(args[0]) if args and args[0].lstrip("-").isdigit() else 10_000
    run_benchmark(n)
