from analyzers.keyframes import extract_keyframes
from analyzers.object_detection import detect_objects
from analyzers.video_tracking import track_objects


video_path = "sample_media/test.mp4"

keyframes = extract_keyframes(video_path)

frame_detections = []

for frame_number, frame_path in enumerate(keyframes):

    detections, _ = detect_objects(frame_path)

    timestamp = frame_number * 5

    frame_detections.append({
        "timestamp": timestamp,
        "detections": detections
    })


tracks = track_objects(frame_detections)


print()
print("VIDEO OBJECT TRACKING")
print("=" * 60)

if not tracks:
    print("No objects were tracked.")
else:

    for track in tracks:

        print(f"Track ID: {track['track_id']}")
        print(f"Object: {track['object']}")
        print(
            f"First appearance: "
            f"{track['first_timestamp']} seconds"
        )
        print(
            f"Last appearance: "
            f"{track['last_timestamp']} seconds"
        )
        print(
            f"Number of observations: "
            f"{len(track['detections'])}"
        )

        print("-" * 60)