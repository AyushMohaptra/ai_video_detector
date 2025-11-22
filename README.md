AI Video Detector

Lightweight heuristic detector to analyze videos for likely AI-generation
using body and temporal motion cues. The detector uses deterministic heuristics
(face dynamics, optical flow and temporal consistency) and does not require
pretrained machine-learning model files.

Repository layout
- `detector/` : main analysis script (`Detector.py`)
Prerequisites
- Python 3.8+ recommended
- A system with OpenCV and MediaPipe dependencies (see `requirements.txt`)

Quick start

1. Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the detector on a video file:

```bash
python detector/Detector.py path/to/video.mp4
```

Options
- `--quiet` (default): suppress native logs from MediaPipe/TensorFlow/OpenGL
- `--no-quiet`: show native logs

Example

```bash
python detector/Detector.py data/test/example.mp4

# Sample output (human-readable metrics)
# --- Face Dynamics ---
# Blink rate: 12.34 blinks/min
# Total blinks: 10
# Mouth variance: 0.000123
# Head motion std: 0.0321
# Face AI suspicion: 0.23

# --- Motion Dynamics ---
# Mean motion: 0.0123
# Motion AI suspicion: 0.18

# === Combined Body Dynamics Score ===
# ➡ Final body-based AI suspicion: 0.18
```

Notes & implementation
- The face analysis uses MediaPipe Face Mesh to compute eye aspect ratios
  (blink detection), mouth openness variance and approximate head motion.
- Motion analysis uses dense optical flow (Farneback) to summarize scene motion.
- Temporal inconsistency analysis inspects optical-flow based jitter, direction
  consistency, and spatial incoherence to produce a temporal suspicion score.

Known issues

- Subtle / low-motion videos: The detector can fail to produce reliable
  signals on videos where subjects exhibit only very small or infrequent
  movements (for example: long static closeups, minimal eye/mouth motion,
  or very stable camera setups). In such cases the optical-flow and face
  dynamics heuristics may not produce strong signals and the detector tends
  to perform better on clips with more pronounced body/head motion or
  facial activity.

Files to review
- `detector/Detector.py` : CLI entrypoint and heuristics implementation
- `requirements.txt`     : Python dependencies

Contributing

If you'd like to contribute to this project, thank you — contributions help improve
the detector. Please follow these guidelines to make reviews and CI easier.

- Report issues: open an issue with a clear title, reproduction steps,
  environment (OS, Python version), and example video if possible.
- Small, focused PRs: fork the repo, create a branch named like
  `fix/description` or `feat/description`, and keep changes small and scoped.
- Tests: add unit tests under `tests/` and keep them fast. Use `pytest` and
  add small sample videos under `data/` only when necessary (keep them tiny
  so CI remains fast). If a sample video is large, provide a download link
  instead of committing it.
- Style: follow existing code style (PEP8-ish). Run `flake8` or similar
  linters before opening a PR if adding formatting or refactors.
- Dependencies: if a change requires new dependencies, update `requirements.txt`
  and explain why in the PR description.
- Commit messages & PRs: write clear commit messages and a descriptive PR
  body that explains the problem, solution, and any trade-offs.


License
- This repository includes a `LICENSE` (MIT) file.
