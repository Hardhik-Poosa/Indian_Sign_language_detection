# Sign Language Detector (Python)

Real-time hand sign classification (A–Z) using:
- OpenCV (camera + image handling)
- MediaPipe Hands (landmark extraction)
- scikit-learn RandomForest (classification)

This repository includes the full pipeline:
1) collect images, 2) build dataset features, 3) train model, 4) run webcam inference.

## Requirements

- OS: Windows (commands below use PowerShell)
- Python: **3.11** (recommended for MediaPipe compatibility)
- Webcam: required for `collect_imgs.py` and `inference_classifier.py`

## Quick Start

From the project root:

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\python.exe -m pip install --upgrade pip
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

## End-to-End Workflow

### 1) Collect images (optional)

If your `data/` folder is already populated, you can skip this step.

```powershell
.\.venv311\Scripts\python.exe collect_imgs.py
```

- Collects images into `data/<LETTER>/`
- Press `q` when prompted to start each class

### 2) Create dataset features

```powershell
.\.venv311\Scripts\python.exe create_dataset.py
```

Output:
- `data.pickle` containing:
	- `data` (shape: `N x 42`)
	- `labels`
	- `feature_size`

### 3) Train classifier

```powershell
.\.venv311\Scripts\python.exe train_classifier.py --data_path ./data.pickle --save_path ./model.p
```

Output:
- `model.p` containing trained model + metadata (`classes`, `feature_size`, `accuracy`)

### 4) Run real-time inference

```powershell
.\.venv311\Scripts\python.exe inference_classifier.py --model_path ./model.p --camera_index 0
```

- Press `q` to quit
- If camera `0` fails, try `--camera_index 1` or `2`

## Script Reference

- `collect_imgs.py`: Capture class images from webcam
- `create_dataset.py`: Extract hand landmark features and save `data.pickle`
- `train_classifier.py`: Train/test RandomForest and save `model.p`
- `inference_classifier.py`: Real-time prediction from webcam feed

## Project Structure

```text
.
├── collect_imgs.py
├── create_dataset.py
├── train_classifier.py
├── inference_classifier.py
├── requirements.txt
├── data/
│   ├── A/
│   ├── B/
│   └── ... Z/
├── data.pickle
└── model.p
```

## Troubleshooting

### `ModuleNotFoundError` (e.g., `mediapipe`)

Install dependencies in the same environment you are running:

```powershell
.\.venv311\Scripts\python.exe -m pip install -r requirements.txt
```

### MediaPipe issues on Python 3.13+

Use Python 3.11 for best compatibility:

```powershell
py -3.11 -m venv .venv311
```

### `No hand detected` for many images

- Improve lighting
- Keep hand fully visible in frame
- Avoid motion blur
- Re-collect low-quality classes and regenerate dataset

## Notes

- Model quality depends directly on data quality and class balance.
- Current model is static letter classification from hand shape landmarks.
