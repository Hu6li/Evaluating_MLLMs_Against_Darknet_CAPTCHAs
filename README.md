# Evaluating Multimodal Language Models Against Darknet CAPTCHAs — Reproduction Samples

This repository contains the sample data and solver code used to reproduce the experimental results from the paper **"Evaluating Multimodal Language Models Against Darknet CAPTCHAs"**.

Three CAPTCHA challenge types are covered, each in its own subdirectory with sample images, ground-truth annotations, benchmark scripts, and LLM prompts.

---

## Repository Layout

```
open_circle/      Open-circle CAPTCHA solver and benchmark
rotation/         Rotation-alignment CAPTCHA solver and benchmark
insects/          Insect-grid CAPTCHA prompts and samples
```

---

## Challenge Types

### 1. Open Circle (`open_circle/`)

The CAPTCHA presents several circular rings on a coloured background; exactly one ring has a visible gap. The task is to locate the centre of that incomplete ring.

**Approach — three strategies are provided:**

| File | Strategy |
|---|---|
| `EdgeCircleDetection.py` | Pure CV: Canny edges + Hough circles |
| `prompts.py` | LLM-only: vision model prompted to return the gap centre |
| `hybrid_prompts.py` | Hybrid: CV proposes a candidate; LLM verifies / corrects it |

**Benchmark:**

```bash
cd open_circle
pip install -r requirements.txt
python3 benchmarkEdgeCircle.py           # all annotated images
python3 benchmarkEdgeCircle.py N        # first <N > images
python3 benchmarkEdgeCircle.py --show benchmark_edge_circle_<ts>.json
```

Success criterion: predicted centre within 30 px of the ground-truth centre.

---

### 2. Rotation (`rotation/`)

The CAPTCHA shows a composite image: a fixed outer ring and a rotated inner disc. The task is to predict the clockwise rotation angle that would align the inner disc with the outer ring.

**Approach — three strategies are provided:**

| File | Strategy |
|---|---|
| `polarGradientCore.py` / `polarGradientValidator.py` | Pure CV: polar-coordinate gradient cross-correlation |
| `estimator_prompts.py` | LLM estimator: vision model predicts the rotation angle directly |
| `decider_prompts.py` | LLM decider: model picks the best-aligned image from 8 candidates |
| `hybrid_estimator_prmpts.py` | Hybrid: CV estimate refined by LLM |

**Benchmark:**

```bash
cd rotation
pip install -r requirements.txt
python3 benchmarkPolarGradient.py              # all pairs, ±5° tolerance
python3 benchmarkPolarGradient.py --round45    # snap predictions to nearest 45°
python3 benchmarkPolarGradient.py --tol 45     # change success tolerance
```

Success criterion: angular error within ±5° of the ground-truth rotation (configurable via `--tol`).

---

### 3. Insects (`insects/`)

The CAPTCHA presents a grid of photographs; some cells show insects (flies, bees, beetles, …) and some do not. The task is to identify all grid positions that contain an insect.

**Approach:** LLM-only. `prompts.py` contains a system prompt that frames the model as a biologist and a user prompt that instructs it to return all insect-containing grid indices.

No benchmark script is provided in this directory; the prompts are intended to be integrated into a vision-model inference pipeline.

---

## Ground Truth

Each benchmarked challenge ships a `truth.json` file:

- `open_circle/truth.json` — per-image centre coordinates `(center_x, center_y)` and a `tolerance_px` field.
- `rotation/truth.json` — per-folder image pairs with the ground-truth clockwise rotation angle and an `ambiguous` flag for near-symmetric patterns.
- `insects/truth.json` — per-image 3x3 matrix, array containing position of square showing an insect

---

## Dependencies

Install per-challenge:

```bash
pip install -r open_circle/requirements.txt   # numpy, opencv-python, pillow
pip install -r rotation/requirements.txt      # numpy, opencv-python
```

---

