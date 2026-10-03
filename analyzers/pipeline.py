import os
import cv2

from analyzers.metadata import extract_image_metadata
from analyzers.object_detection import detect_objects
from analyzers.face_detection import detect_faces
from analyzers.ocr import extract_text
from analyzers.video_metadata import extract_video_metadata
from analyzers.keyframes import extract_keyframes
from analyzers.bytetrack import track_video
from analyzers.transcription import transcribe_video


def analyze_image(image_path):
    """
    Run the complete analysis pipeline on an image.
    """

    print("Analyzing image...")

    print("Extracting metadata...")
    metadata = extract_image_metadata(image_path)

    print("Detecting objects...")
    detections, annotated_image = detect_objects(image_path)

    os.makedirs("outputs", exist_ok=True)

    annotated_image_path = os.path.join(
        "outputs",
        "annotated_image.jpg"
)

    cv2.imwrite(
        annotated_image_path,
        annotated_image
)

    print("Detecting faces...")
    face_count, face_boxes, _ = detect_faces(image_path)

    print("Extracting text...")
    text_results = extract_text(image_path)

    analysis = {
        "media_type": "image",
        "file_name": os.path.basename(image_path),

        "metadata": metadata,

        "objects": detections,

        "annotated_image": annotated_image_path,

        "faces": {
            "count": face_count,
            "detections": face_boxes
        },

        "ocr": text_results,

        "speech": {
            "applicable": False,
            "reason": "Speech is not applicable to an image."
        }
    }

    return analysis


def analyze_video(video_path, keyframe_interval=5):
    """
    Run the complete analysis pipeline on a video.
    """

    print("Analyzing video...")

    # --------------------------------
    # 1. Video metadata
    # --------------------------------
    print("Extracting video metadata...")

    metadata = extract_video_metadata(video_path)

    # --------------------------------
    # 2. Keyframes
    # --------------------------------
    print("Extracting keyframes...")

    keyframes = extract_keyframes(
        video_path,
        interval=keyframe_interval
    )

    # --------------------------------
    # 3. Analyze keyframes
    # --------------------------------
    print("Analyzing keyframes...")

    frame_analysis = []

    for frame_number, frame_path in enumerate(keyframes):

        timestamp = frame_number * keyframe_interval

        print(
            f"Analyzing keyframe "
            f"{frame_number + 1}/{len(keyframes)} "
            f"at {timestamp}s..."
        )

        # Object detection
        objects, annotated_frame = detect_objects(frame_path)

        annotated_dir = os.path.join(
             "outputs",
             "keyframes",
             "annotated"
        )
        os.makedirs(
            annotated_dir,
            exist_ok=True
        )
        annotated_frame_path = os.path.join(
               annotated_dir,
               os.path.basename(frame_path)
        )
        cv2.imwrite(
            annotated_frame_path,
            annotated_frame
        )
            


        # Face detection
        face_count, face_boxes, _ = detect_faces(
        frame_path,
        score_threshold=0.9,
        min_face_size=50
        )
        # OCR
        text_results = extract_text(frame_path)

        frame_analysis.append({
            "timestamp": timestamp,
            "frame": os.path.basename(frame_path),
            "annotated_frame": annotated_frame_path,

            "objects": objects,

            "faces": {
                "count": face_count,
                "detections": face_boxes
            },

            "ocr": text_results
        })

    # --------------------------------
    # 4. Video object tracking
    # --------------------------------
    print("Tracking objects across video...")

    tracks, fps, frame_count = track_video(
        video_path
    )

    tracking = {
        "fps": fps,
        "frame_count": frame_count,
        "tracks": tracks
    }

    # --------------------------------
    # 5. Speech transcription
    # --------------------------------
    print("Transcribing speech...")

    speech = transcribe_video(video_path)

    # --------------------------------
    # Final structured result
    # --------------------------------
    analysis = {
        "media_type": "video",
        "file_name": os.path.basename(video_path),

        "metadata": metadata,

        "keyframe_interval_seconds": keyframe_interval,

        "keyframes": frame_analysis,

        "tracking": tracking,

        "speech": speech
    }

    return analysis