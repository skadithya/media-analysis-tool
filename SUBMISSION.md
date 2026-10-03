\# Media Analysis Tool - Submission



\## 1. Project Overview



The Media Analysis Tool is a local web application that analyzes one uploaded image or video and produces an evidence-grounded report.



The application follows this pipeline:



Upload Media

→ Metadata Extraction

→ Visual Analysis

→ Face Detection

→ OCR

→ Video Keyframes / Tracking

→ Speech Transcription

→ Evidence Preparation

→ Local AI Report

→ Final Report



The application is implemented using Python and Streamlit and runs locally.



\---



\## 2. Implemented Features



\### Image Analysis



For uploaded images, the application performs:



\- EXIF metadata extraction

\- Camera make and model detection

\- Date/time extraction

\- Editing software extraction

\- GPS extraction when available

\- Object detection

\- Object confidence scores

\- Object bounding boxes

\- Human face detection

\- Face count

\- Face bounding boxes

\- OCR for visible text

\- OCR confidence scores

\- Evidence-grounded report generation

\- Explicit handling of unavailable metadata



\### Video Analysis



For uploaded videos, the application performs:



\- Duration extraction

\- Resolution extraction

\- Video codec detection

\- Creation-date extraction when available

\- Keyframe extraction

\- Object detection on sampled keyframes

\- Bounding-box visualization

\- Face detection on sampled keyframes

\- Face counts on sampled keyframes

\- OCR on sampled keyframes

\- Object tracking using ByteTrack

\- First and last appearance timestamps for tracked objects

\- Speech transcription

\- Timestamped transcript segments

\- Evidence-grounded report generation



\---



\## 3. Pipeline



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │    Upload Image /   │

&#x20;                   │        Video        │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Detect Media Type │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                ┌─────────────┴─────────────┐

&#x20;                │                           │

&#x20;                ▼                           ▼

&#x20;       ┌─────────────────┐         ┌─────────────────┐

&#x20;       │      IMAGE      │         │      VIDEO      │

&#x20;       └────────┬────────┘         └────────┬────────┘

&#x20;                │                           │

&#x20;                ▼                           ▼

&#x20;       ┌─────────────────┐         ┌─────────────────┐

&#x20;       │ EXIF Metadata   │         │ FFprobe Metadata│

&#x20;       └────────┬────────┘         └────────┬────────┘

&#x20;                │                           │

&#x20;                └─────────────┬─────────────┘

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Object Detection  │

&#x20;                   │       YOLO11n       │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │   Face Detection    │

&#x20;                   │        YuNet        │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │        OCR          │

&#x20;                   │      EasyOCR        │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                   ┌──────────┴──────────┐

&#x20;                   │                     │

&#x20;                   ▼                     ▼

&#x20;            Image Report          Video Processing

&#x20;                                         │

&#x20;                        ┌────────────────┼────────────────┐

&#x20;                        │                │                │

&#x20;                        ▼                ▼                ▼

&#x20;                    Keyframes        ByteTrack        Whisper

&#x20;                        │             Tracking       Transcription

&#x20;                        │                │                │

&#x20;                        └────────────────┼────────────────┘

&#x20;                                         │

&#x20;                                         ▼

&#x20;                             ┌─────────────────────┐

&#x20;                             │ Evidence Preparation│

&#x20;                             └──────────┬──────────┘

&#x20;                                        │

&#x20;                                        ▼

&#x20;                             ┌─────────────────────┐

&#x20;                             │ Local Qwen2.5 3B    │

&#x20;                             │ Evidence Review     │

&#x20;                             └──────────┬──────────┘

&#x20;                                        │

&#x20;                                        ▼

&#x20;                             ┌─────────────────────┐

&#x20;                             │ Evidence-Grounded   │

&#x20;                             │ Final Report        │

&#x20;                             └─────────────────────┘

4. Technology Stack

Application

\- Python 3.11

\- Streamlit

\- OpenCV

\- Pillow

\- NumPy

\- Pandas

\- FFmpeg / FFprobe

Computer Vision

\- Ultralytics YOLO11n

\- OpenCV YuNet face detector

OCR

\- EasyOCR

Video Analysis

\- FFmpeg

\- ByteTrack through Ultralytics tracking

\- Keyframe sampling at approximately 5-second intervals

Speech

\- faster-whisper

\- Whisper Small model

Report Generation

\- Ollama

\- Qwen2.5 3B

\- Local HTTP API at http://localhost:11434

5\. Models and Licences

The assignment requires every model used by the application to be listed with its source and licence.

Model / Component	Purpose	Source	Licence

YOLO11n	Object detection	https://github.com/ultralytics/ultralytics	AGPL-3.0

YuNet face\_detection\_yunet\_2023mar.onnx	Human face detection	https://github.com/opencv/opencv\_zoo/tree/main/models/face\_detection\_yunet	MIT

EasyOCR models	Visible-text OCR	https://github.com/JaidedAI/EasyOCR	Apache License 2.0

Whisper Small	Speech transcription	https://github.com/openai/whisper	MIT

faster-whisper	Local Whisper inference implementation	https://github.com/SYSTRAN/faster-whisper	MIT

ByteTrack	Video object tracking	https://github.com/ifzhang/ByteTrack	MIT

Qwen2.5 3B	Local evidence-review/report generation	https://ollama.com/library/qwen2.5:3b	Qwen License





Important licensing note

YOLO11 is distributed by Ultralytics under the AGPL-3.0 open-source licence, with a separate Enterprise licensing option. This project uses the open-source YOLO11n model.

The complete source code of this project is included in the submission repository so that the project can be inspected and reproduced.

6\. Setup

Requirements

\- Windows / Linux / macOS

\- Python 3.11 recommended

\- FFmpeg

\- Ollama

\- Internet connection for initial Python/model installation

1\. Clone or copy the project

git clone <repository-url>

cd media-analysis-tool



If the project is provided as a folder instead of a Git repository, simply open the project directory.

2\. Create a virtual environment

Windows:

python -m venv venv



Activate it:

venv\\Scripts\\activate



3\. Install Python dependencies

python -m pip install -r requirements.txt



4\. Install FFmpeg

FFmpeg must be available in the system PATH.

Verify:

ffmpeg -version

ffprobe -version



5\. Install Ollama

Install Ollama and make sure the local Ollama service is running.

Pull the report-generation model:

ollama pull qwen2.5:3b



The application expects the Ollama local API at:

http://localhost:11434/api/generate



6\. Run the application

streamlit run app.py



Open:

http://localhost:8501



7\. Usage

Image

1\. Open the Streamlit application.

2\. Upload a JPG, JPEG, PNG, WEBP or BMP image.

3\. The application analyzes the image.

4\. Detected objects and faces are displayed with bounding boxes.

5\. OCR results are displayed with confidence scores.

6\. EXIF metadata is displayed when available.

7\. The final evidence-grounded report is displayed.

Video

1\. Upload an MP4, MOV, AVI, MKV or WEBM video.

2\. Video metadata is extracted using FFprobe.

3\. Keyframes are sampled approximately every 5 seconds.

4\. Object detection and face detection are performed on sampled frames.

5\. Detected regions are displayed with bounding boxes.

6\. ByteTrack tracks detected objects through the video.

7\. OCR is performed on sampled frames.

8\. Speech is transcribed using faster-whisper.

9\. The transcript contains timestamps.

10\. The collected evidence is used to generate the final report.

8\. Evidence and Uncertainty Handling

The application is designed to avoid presenting uncertain model outputs as verified facts.

Object Detection

Object detections include confidence scores.

Low-confidence detections are marked:

Unverified / low confidence



OCR

OCR results include confidence scores.

Low-confidence OCR results are marked as unverified.

OCR output is treated as machine-detected text and is not independently verified.

Face Detection

Face detection reports detected face regions and counts.

The system does not identify people.

For videos, face counts refer to sampled frames and are not treated as a unique-person count for the entire video.

Tracking

Tracking IDs are model-generated tracking instances.

They are not treated as confirmed identities.

First and last timestamps refer to the tracking observations produced by the model.

Speech

Speech transcription is machine-generated and may contain transcription errors.

Transcript statements are therefore attributed to the machine-generated transcription rather than treated as independently verified facts.

Missing Metadata

When metadata is absent, the application does not guess.

It reports:

Could not be determined.



9\. Real Image Test

Input

sample\_media/test.jpg



Observed Results

The tested image produced:

\- 1 detected human face

\- Object detections including:

&#x20; - person

&#x20; - potted plant

&#x20; - bowl

&#x20; - dining table

\- Bounding boxes were generated for detected objects.

\- OCR detected visible text.

\- EXIF fields were unavailable in the test image.

\- The report explicitly marked unavailable metadata as unable to be determined.

\- Low-confidence object/OCR detections were marked as unverified.

Example object confidence values from the test:

person: 0.9061

potted plant: 0.5596

bowl: 0.3874 - unverified / low confidence

potted plant: 0.3834 - unverified / low confidence

dining table: 0.3439 - unverified / low confidence



The application displayed the annotated image with detected regions.

10\. Real Video Test

Input

sample\_media/test.mp4



Video Metadata

The tested video produced:

Duration: 60.54 seconds

Resolution: 1920x1080

Codec: h264

Creation date: Could not be determined



Keyframes

The video was sampled approximately every 5 seconds:

0s

5s

10s

15s

20s

25s

30s

35s

40s

45s

50s

55s



Each sampled keyframe was analyzed for objects, faces and OCR.

Object Tracking

ByteTrack produced model-generated tracking records containing:

\- Track ID

\- Object class

\- First appearance timestamp

\- Last appearance timestamp

\- Maximum confidence

\- Observation count

Low-confidence tracks are explicitly treated as unverified.

Speech

The tested video contained machine-generated English transcription.

The transcription contained timestamped segments covering the video.

The detected language was:

English



The language confidence was approximately:

0.9888



The transcript is explicitly treated as machine-generated and potentially imperfect.

11\. Screenshots / Saved Evidence

The final submission should include screenshots showing the working application.

Recommended files:

submission\_assets/

├── image\_result.png

├── video\_result.png

├── image\_report.md

└── video\_report.md



Image screenshot should show

\- Uploaded image

\- Detected bounding boxes

\- Metadata

\- Object detections

\- Face detection

\- OCR

\- Final report

Video screenshot should show

\- Video metadata

\- Annotated keyframe

\- Tracking information

\- Speech transcription

\- Final report

12\. Project Structure

media-analysis-tool/

│

├── analyzers/

│   ├── \_\_init\_\_.py

│   ├── metadata.py

│   ├── object\_detection.py

│   ├── face\_detection.py

│   ├── ocr.py

│   ├── video\_metadata.py

│   ├── keyframes.py

│   ├── bytetrack.py

│   ├── video\_faces.py

│   ├── video\_tracking.py

│   ├── transcription.py

│   └── report\_generator.py

│

├── models/

│   └── face\_detection\_yunet\_2023mar.onnx

│

├── sample\_media/

│   ├── test.jpg

│   └── test.mp4

│

├── utils/

│   └── \_\_init\_\_.py

│

├── app.py

├── requirements.txt

├── README.md

├── SUBMISSION.md

├── .gitignore

├── yolo11n.pt

└── test\_\*.py



Generated analysis outputs are excluded from normal Git tracking through .gitignore.

13\. Testing

The project contains individual tests for the major pipeline components, including:

\- Metadata extraction

\- Object detection

\- Face detection

\- Face filtering

\- Face thresholds

\- OCR

\- Keyframe extraction

\- Video metadata

\- Video objects

\- Video OCR

\- Video faces

\- Video tracking

\- ByteTrack

\- Speech transcription

\- Image pipeline

\- Video pipeline

\- Image report generation

\- Video report generation

The complete Streamlit application was also tested end-to-end with a real image and a real video.

14\. Known Limitations

Metadata

\- Metadata can only be extracted if the information is actually stored in the file.

\- Missing EXIF/video metadata cannot be reconstructed.

\- GPS is reported only when available in the source metadata.

Object Detection

\- YOLO11n can produce false positives and false negatives.

\- Low-confidence detections may be incorrect.

\- Detection confidence does not prove that an object is actually present.

Face Detection

\- Detection accuracy depends on face size, angle, lighting and image quality.

\- Video face counts are based on sampled keyframes.

\- The system counts detected face regions and does not identify people.

\- Face counts across different frames must not be interpreted as the number of unique people.

OCR

\- OCR can produce incorrect characters.

\- Small, blurry, rotated or stylized text may not be recognized correctly.

\- Low-confidence OCR results are marked as unverified.

\- OCR results are not independently verified.

Video Tracking

\- Tracking is performed using sampled/detected objects and can produce fragmented or incorrect tracks.

\- Tracking IDs do not represent confirmed real-world identities.

\- First/last appearance timestamps describe model tracking observations.

Speech Transcription

\- Speech transcription can contain errors.

\- Background noise, overlapping speakers and unclear speech can reduce accuracy.

\- Transcribed names or statements are not treated as independently verified facts.

AI Report

\- The local language model receives collected analysis evidence rather than directly interpreting the original media.

\- The report therefore depends on the quality of upstream metadata, detection, OCR and transcription.

\- Unsupported information is intentionally excluded or marked as unable to be determined.

Performance

\- Video analysis is computationally expensive on CPU.

\- Whisper transcription can take significant time for longer videos.

\- Processing time depends on video length, resolution and hardware.

15\. Design Decisions

The project prioritizes:

1\. Local execution

2\. Deterministic processing

3\. Evidence-grounded reporting

4\. Honest handling of missing information

5\. Explicit uncertainty

6\. Simple reproducible components

7\. Visual evidence through bounding boxes

8\. No person identification

The final report is generated from structured evidence collected by the analysis pipeline rather than allowing the language model to freely invent observations about the media.

16\. Optional Features

The following optional assignment features were not implemented as core functionality:

\- C2PA provenance validation

\- Audio sound-event classification

\- Automated reverse-image search

\- Dedicated vehicle license-plate recognition

The application focuses on completing the core image and video analysis pipeline reliably.

17\. Final Status

The application currently provides an end-to-end local workflow:

Upload

→ Metadata

→ Object Detection

→ Face Detection

→ OCR

→ Video Keyframes

→ Video Tracking

→ Speech Transcription

→ Evidence Preparation

→ Local AI Review

→ Evidence-Grounded Report



The application has been tested using one real image and one real video and produces annotated visual results, structured analysis data and a final report.

