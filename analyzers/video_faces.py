from analyzers.keyframes import extract_keyframes
from analyzers.face_detection import detect_faces


def detect_faces_in_video(video_path, interval=5):
    """
    Detect faces in sampled video keyframes.

    Returns a list containing:
    - timestamp
    - face count
    - face bounding boxes
    """

    keyframes = extract_keyframes(
        video_path,
        interval=interval
    )

    results = []

    for frame_number, frame_path in enumerate(keyframes):

        timestamp = frame_number * interval

        face_count, face_boxes, _ = detect_faces(
            frame_path
        )

        results.append({
            "timestamp": timestamp,
            "face_count": face_count,
            "faces": face_boxes
        })

    return results