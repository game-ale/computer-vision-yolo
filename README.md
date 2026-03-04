# Computer Vision YOLO Repo

A lightweight computer vision project using **Ultralytics YOLO** for detection from webcam, images, videos, or folders.

## Features
- Webcam real-time detection
- Image/video/folder inference
- Configurable confidence and image size
- Simple save/predict workflow

## Project Structure
- `detect.py` – main inference script
- `train.py` – YOLO model training script
- `utils.py` – drawing helpers and class-name loader
- `test_detect.py` – unit tests (run with `pytest`)
- `requirements.txt` – dependencies
- `.gitignore` – ignores environments, runs, and model artifacts

## Setup
1. Create and activate your virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Place your model in the project root as `best.pt` (or pass custom path with `--weights`).

## Usage
Webcam:
- `python detect.py --source 0 --show`

Image:
- `python detect.py --source path/to/image.jpg --show`

Video:
- `python detect.py --source path/to/video.mp4 --save`

Custom weights:
- `python detect.py --weights path/to/model.pt --source 0 --show`

## Training

Train on a custom dataset (requires a `data.yaml` describing the dataset):

```
python train.py --data data.yaml --epochs 50
```

With a custom starting model:

```
python train.py --data data.yaml --weights best.pt --epochs 30
```

Resume from a checkpoint:

```
python train.py --data data.yaml --resume
```

## Notes
- By default, predictions are saved under `outputs/`.
- Training results are saved under `runs/train/`.
- If you want to keep model weights in Git, remove `*.pt` from `.gitignore`.
