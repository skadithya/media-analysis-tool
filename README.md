# Media Analysis Tool

A local web application for analyzing images and videos using open-source computer vision, OCR, speech recognition, and evidence-grounded reporting.

## Features

### Image Analysis

- Extracts image EXIF metadata:
  - Camera make
  - Camera model
  - Date taken
  - Editing software
  - GPS coordinates when available
- Detects objects using YOLO11n
- Reports object confidence scores and bounding boxes
- Detects human faces using OpenCV YuNet
- Extracts visible text using EasyOCR
- Displays detected regions using bounding boxes
- Generates an evidence-grounded analysis report
- Reports unavailable metadata as "Could not be determined"
- Marks low-confidence detections as unverified

### Video Analysis

- Extracts:
  - Duration
  - Resolution
  - Video codec
  - Creation date when available
- Extracts keyframes at 5-second intervals
- Performs object detection on sampled keyframes
- Displays detected regions with bounding boxes
- Detects faces in sampled keyframes
- Performs OCR on sampled keyframes
- Tracks objects across the video using ByteTrack
- Reports first and last observed timestamps for tracked objects
- Transcribes speech using Faster-Whisper
- Provides timestamped transcript segments
- Generates an evidence-grounded report

## Technology Stack

- Python 3.11
- Streamlit
- Ultralytics YOLO11n
- OpenCV
- OpenCV YuNet
- EasyOCR
- Faster-Whisper
- PyTorch
- Pandas
- NumPy
- FFmpeg / FFprobe

## Project Structure

```text
media-analysis-tool/
│
├── analyzers/
│   ├── metadata.py
│   ├── object_detection.py
│   ├── face_detection.py
│   ├── ocr.py
│   ├── video_metadata.py
│   ├── keyframes.py
│   ├── bytetrack.py
│   ├── video_faces.py
│   ├── video_tracking.py
│   ├── transcription.py
│   ├── pipeline.py
│   └── report_generator.py
│
├── models/
│   └── face_detection_yunet_2023mar.onnx
│
├── sample_media/
├── outputs/
│
├── app.py
├── requirements.txt
├── README.md
├── SUBMISSION.md
├── yolo11n.pt
└── .gitignore