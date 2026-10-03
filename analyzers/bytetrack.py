import cv2
from ultralytics import YOLO


MODEL_PATH = "yolo11n.pt"

model = YOLO(MODEL_PATH)


def track_video(video_path):
    """
    Track objects throughout a video using ByteTrack.

    Returns:
        tracks: list of tracked objects with first/last timestamps
        fps: video FPS
        frame_count: total number of frames
    """

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    video.release()

    tracks = {}

    results = model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        verbose=False
    )

    for frame_number, result in enumerate(results):

        timestamp = frame_number / fps

        if result.boxes is None:
            continue

        boxes = result.boxes

        if boxes.id is None:
            continue

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()
        confidences = boxes.conf.cpu().tolist()

        for track_id, class_id, confidence in zip(
            track_ids,
            class_ids,
            confidences
        ):

            object_name = model.names[class_id]

            if track_id not in tracks:

                tracks[track_id] = {
                    "track_id": int(track_id),
                    "object": object_name,
                    "first_timestamp": round(timestamp, 2),
                    "last_timestamp": round(timestamp, 2),
                    "max_confidence": round(float(confidence), 4),
                    "observations": 1
                }

            else:

                track = tracks[track_id]

                track["last_timestamp"] = round(timestamp, 2)

                track["max_confidence"] = max(
                    track["max_confidence"],
                    round(float(confidence), 4)
                )

                track["observations"] += 1

    return list(tracks.values()), fps, frame_count