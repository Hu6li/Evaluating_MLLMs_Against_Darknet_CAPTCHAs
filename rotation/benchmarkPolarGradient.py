#!/usr/bin/env python3
"""
Benchmark the polar-gradient solver against all samples.

Runs solve_rotation_polar_gradient on every inner/outer pair in
samples/ and saves per-test results to results/.

Usage:
    python3 benchmarkPolarGradient.py              # all 149 pairs
    python3 benchmarkPolarGradient.py --round45    # snap predictions to 45°
    python3 benchmarkPolarGradient.py --tol 45     # change success tolerance (default 5°)
"""

import json
import sys
import time
from pathlib import Path

from polarGradientValidator import solve_rotation_polar_gradient

SAMPLE_DIR     = Path(__file__).parent / "samples"
RESULTS_DIR = Path(__file__).parent


def load_truth(path: Path):
    with open(path) as f:
        data = json.load(f)
    entries = []
    for img_entry in data["images"]:
        folder = img_entry["folder"]
        for k, pair in enumerate(img_entry.get("pairs", [])):
            entries.append({
                "key":        f"{folder.replace('_','')}_pair{k}",
                "folder":     folder,
                "outer":      pair["outer"],
                "inner":      pair["inner"],
                "true_rotation": int(pair["rotation_cw"]),
                "ambiguous":  pair.get("ambiguous", False),
            })
    return entries


def angular_error(pred: int, truth: int) -> int:
    diff = abs(pred - truth) % 360
    return min(diff, 360 - diff)


def run(round_to_45: bool, tolerance: int) -> None:
    truth_path = Path(__file__).parent / "truth.json"
    if not SAMPLE_DIR.exists():
        print(f"[!] samples/ not found: {SAMPLE_DIR}"); sys.exit(1)
    if not truth_path.exists():
        print(f"[!] truth.json not found: {truth_path}"); sys.exit(1)

    samples = load_truth(truth_path)
    n_total = len(samples)

    print("=" * 64)
    print("Polar Gradient — Samples Benchmark")
    print("=" * 64)
    print(f"  Pairs       : {n_total}")
    print(f"  Round to 45°: {'Yes' if round_to_45 else 'No'}")
    print(f"  Tolerance   : ±{tolerance}°")
    print("=" * 64)

    tests = []
    n_success = 0

    for idx, s in enumerate(samples, 1):
        outer_path = SAMPLE_DIR / s["folder"] / s["outer"]
        inner_path = SAMPLE_DIR / s["folder"] / s["inner"]

        if not outer_path.exists() or not inner_path.exists():
            print(f"  [{idx:>3}/{n_total}] {s['key']}  SKIP (missing image)")
            continue

        t0 = time.perf_counter()
        predicted, confidence, is_ambiguous = solve_rotation_polar_gradient(
            str(inner_path), str(outer_path), round_to_45=round_to_45
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        err = angular_error(predicted, s["true_rotation"])
        success = err <= tolerance

        if success:
            n_success += 1

        entry = {
            "key":           s["key"],
            "true_rotation": s["true_rotation"],
            "predicted":     predicted,
            "error_deg":     err,
            "success":       success,
            "confidence":    round(confidence, 6),
            "is_ambiguous":  is_ambiguous,
            "gt_ambiguous":  s["ambiguous"],
            "elapsed_ms":    round(elapsed_ms, 2),
        }
        tests.append(entry)

        mark = "✓" if success else "✗"
        print(
            f"  [{idx:>3}/{n_total}] {s['key']:<22}  truth={s['true_rotation']:>3}°"
            f"  pred={predicted:>3}°  err={err:>3}°  {mark}"
            f"  conf={confidence:.3f}  ({elapsed_ms:.0f}ms)"
        )

    n_evaluated = len(tests)
    errors = [t["error_deg"] for t in tests]
    mae = sum(errors) / len(errors) if errors else 0.0
    acc = n_success / n_evaluated * 100 if n_evaluated else 0.0

    summary = {
        "solver":       "polar_gradient",
        "round_to_45":  round_to_45,
        "tolerance":    tolerance,
        "count":        n_evaluated,
        "success":      n_success,
        "failed":       n_evaluated - n_success,
        "accuracy_pct": round(acc, 2),
        "mae_deg":      round(mae, 2),
        "tests":        tests,
    }

    print("\n" + "=" * 64)
    print(f"  Success (±{tolerance}°) : {n_success}/{n_evaluated}  ({acc:.1f}%)")
    print(f"  Mean angular error: {mae:.1f}°")
    print("=" * 64)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"polar_gradient_results_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[✓] Results saved → {out_file}")


if __name__ == "__main__":
    args = sys.argv[1:]
    round_to_45 = "--round45" in args or "-r" in args
    tol = int(args[args.index("--tol") + 1]) if "--tol" in args else 5
    run(round_to_45, tol)
