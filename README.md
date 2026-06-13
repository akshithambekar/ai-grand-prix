# ai-grand-prix

Python autonomy and perception stack for the AI Grand Prix simulator.

## Windows GPU Setup

These instructions assume:

- Windows 11
- PowerShell
- NVIDIA GPU with working drivers
- The AI Grand Prix simulator is running on the same machine
- You want to run the new perception pipeline against the simulator's MAVLink and vision streams

### 1. Install Python 3.14

This repo is pinned to Python `3.14.x`.

Verify:

```powershell
py -3.14 --version
```

If that fails, install Python 3.14 first and make sure `py -3.14` works.

### 2. Install `uv`

If you do not already have `uv`:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then verify:

```powershell
uv --version
```

### 3. Create the project environment

From the repo root:

```powershell
uv sync
```

That installs the repo dependencies from `pyproject.toml` and `uv.lock`.

### 4. Replace CPU Torch with a CUDA-enabled Torch build

The repo depends on `torch`, but for live GPU inference you should install the CUDA build that matches your Windows box.

First check the currently installed torch build:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

If `torch.cuda.is_available()` prints `False`, install the correct CUDA wheel using the official PyTorch instructions for your machine:

[PyTorch Install Guide](https://pytorch.org/get-started/locally/)

Typical example for CUDA 12.6:

```powershell
uv run python -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Re-check:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

You want `True` there before expecting live YOLO inference to use the GPU.

### 5. Verify the perception environment

Run the built-in compatibility probe:

```powershell
uv run python -m perception.compat
```

Expected behavior:

- `ultralytics_import_ok` should be `true`
- `hf_download_ok` should be `true`
- `checkpoint_load_ok` should be `true`

Important detail:

- `openvision/yolo26-n-seg` is a Hugging Face repo id, not a direct Ultralytics model path
- the code already handles this by downloading `model.pt` from Hugging Face before loading the model

## Simulator Ports

The current code assumes the simulator exposes:

- MAVLink UDP on `127.0.0.1:14550`
- Vision UDP on `0.0.0.0:5600`

Those defaults match the existing example client and the current perception tooling.

If your simulator uses different ports, pass the alternate values when recording data or in the live runner script below.

## Perception Workflow

The perception stack currently has two modes:

1. Offline tooling for calibration, dataset generation, review, training, and replay evaluation
2. A live `PerceptionRunner` runtime API for simulator ingest and inference

## Live Perception Runner

Before running live inference you need:

- a camera calibration JSON produced by `perception-calibrate`
- optionally, trained YOLO weights from `perception-train-yolo`

If you do not pass `--weights`, the runner still performs ingest, synchronization, and projection, but it will report `detector_not_configured` instead of model detections.

### Start live perception from PowerShell

From the repo root:

```powershell
uv run perception-run-live `
  --calibration artifacts/perception/calibration/camera_calibration.json `
  --weights runs/perception/latest/best.pt `
  --device cuda:0
```

All flags are optional and default to the values shown above. Pass `--sim-host`, `--mavlink-port`, `--vision-host`, or `--vision-port` to override the simulator connection if needed.

## Offline Perception Commands

The repo exposes these CLI commands through `pyproject.toml`.

### Record a synchronized raw dataset

```powershell
uv run perception-record-dataset `
  --output-root artifacts/perception/raw `
  --session-id sim_session_01 `
  --sim-host 127.0.0.1 `
  --mavlink-port 14550 `
  --vision-host 0.0.0.0 `
  --vision-port 5600 `
  --calibration artifacts/perception/calibration/camera_calibration.json `
  --max-samples 2000
```

### Fit camera calibration from reviewed calibration samples

```powershell
uv run perception-calibrate `
  --manifest artifacts/perception/calibration/calibration_manifest.jsonl `
  --output artifacts/perception/calibration/camera_calibration.json
```

The calibration manifest must contain observed gate corners and matching telemetry/gate geometry. The code expects the JSONL format defined in `perception.calibration.load_calibration_samples()`.

### Generate weak labels from a recorded raw session

```powershell
uv run perception-generate-labels `
  --raw-session-root artifacts/perception/raw/sim_session_01 `
  --calibration artifacts/perception/calibration/camera_calibration.json `
  --output-root artifacts/perception/weak_labels
```

### Render review overlays

```powershell
uv run perception-review-labels render `
  --weak-session-root artifacts/perception/weak_labels/sim_session_01 `
  --output-dir artifacts/perception/review/sim_session_01
```

That writes:

- overlay images
- contact sheets
- `review_decisions_template.jsonl`

### Apply review decisions

```powershell
uv run perception-review-labels apply `
  --weak-session-root artifacts/perception/weak_labels/sim_session_01 `
  --decisions artifacts/perception/review/sim_session_01/review_decisions_template.jsonl `
  --output-root artifacts/perception/reviewed
```

### Train YOLO on the reviewed dataset

```powershell
uv run perception-train-yolo `
  --reviewed-root artifacts/perception/reviewed `
  --output-dir runs/perception `
  --device cuda:0 `
  --epochs 100 `
  --batch 4 `
  --imgsz 640
```

Outputs include:

- Ultralytics training artifacts
- best checkpoint
- split summary
- validation metrics

### Evaluate a recorded replay with calibration + trained weights

```powershell
uv run perception-evaluate-replay `
  --raw-session-root artifacts/perception/raw/sim_session_01 `
  --calibration artifacts/perception/calibration/camera_calibration.json `
  --weights runs/perception/20260610T000000Z/ultralytics/gate_seg/weights/best.pt `
  --output-dir artifacts/perception/replay_eval/sim_session_01 `
  --device cuda:0
```

## Recommended Bring-Up Order

On the Windows GPU machine, the least error-prone order is:

1. `uv sync`
2. install the correct CUDA-enabled Torch build
3. `uv run python -m perception.compat`
4. record a short raw session
5. generate or fit camera calibration
6. generate weak labels
7. review labels
8. train YOLO
9. evaluate replay
10. run the live `PerceptionRunner`

## Current Limitations

- The `perception-run-live` entrypoint prints raw result dicts; it does not yet write structured logs or publish MAVLink output.
- Calibration sample creation is not yet automated end-to-end. The calibration command expects a manifest with observed gate corners.
- The projection math assumes the current gate/body/camera frame conventions encoded in `src/perception/geometry.py`. Verify overlay alignment on real recordings before trusting training outputs.
