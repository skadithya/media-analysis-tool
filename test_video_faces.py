from analyzers.video_faces import detect_faces_in_video


video_path = "sample_media/test.mp4"

results = detect_faces_in_video(video_path)


print()
print("VIDEO FACE DETECTION")
print("=" * 60)


for result in results:

    print(
        f"Timestamp: {result['timestamp']} seconds"
    )

    print(
        f"Faces detected: {result['face_count']}"
    )

    for index, face in enumerate(
        result["faces"],
        start=1
    ):
        print(
            f"Face {index}: {face}"
        )

    print("-" * 60)