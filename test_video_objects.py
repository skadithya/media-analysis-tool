from analyzers.keyframes import extract_keyframes
from analyzers.object_detection import detect_objects


video_path = "sample_media/test.mp4"

keyframes = extract_keyframes(video_path)

print()
print("VIDEO OBJECT DETECTION")
print("=" * 60)

for frame_number, frame_path in enumerate(keyframes):
    detections, _ = detect_objects(frame_path)

    timestamp = frame_number * 5

    print()
    print(f"Timestamp: {timestamp} seconds")
    print(f"Frame: {frame_path}")

    if not detections:
        print("No objects detected.")
        continue

    for detection in detections:
        print(
            f"Object: {detection['object']} | "
            f"Confidence: {detection['confidence']} | "
            f"Box: {detection['bounding_box']}"
        )