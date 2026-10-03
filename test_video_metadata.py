from analyzers.video_metadata import extract_video_metadata


video_path = "sample_media/test.mp4"

metadata = extract_video_metadata(video_path)

print()
print("VIDEO METADATA")
print("=" * 50)

for key, value in metadata.items():
    print(f"{key}: {value}")