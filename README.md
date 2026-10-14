# Virtual Mouse

An AI-powered virtual mouse controlled using hand gestures captured via your webcam, built with OpenCV, MediaPipe, and PyAutoGUI.

## Features
- Real-time hand landmark tracking using MediaPipe.
- Cursor movement mapped directly to hand gestures.
- Click and drag actions with customizable thresholds and smoothing.

## Requirements
- Python 3.8+
- Webcam

## Installation
```bash
pip install -r requirements.txt
```

Download the MediaPipe hand landmarker model:
- Download `hand_landmarker.task` from MediaPipe models and place it in the same directory as `virtual_mouse.py`.

## Usage
```bash
python virtual_mouse.py
```
