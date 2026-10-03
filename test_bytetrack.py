from analyzers.bytetrack import track_video


video_path = "sample_media/test.mp4"

tracks, fps, frame_count = track_video(video_path)

print()
print("BYTETRACK VIDEO TRACKING")
print("=" * 60)

print(f"FPS: {fps:.2f}")
print(f"Total frames: {frame_count}")
print()

if not tracks:
    print("No objects tracked.")
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
            f"Max confidence: "
            f"{track['max_confidence']}"
        )
        print(
            f"Observations: "
            f"{track['observations']}"
        )
        print("-" * 60)