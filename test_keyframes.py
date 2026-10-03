from analyzers.keyframes import extract_keyframes


video_path = "sample_media/test.mp4"

keyframes = extract_keyframes(video_path)

print()
print("VIDEO KEYFRAMES")
print("=" * 50)

print(f"Total keyframes extracted: {len(keyframes)}")

for index, frame in enumerate(keyframes, start=1):
    print(f"{index}. {frame}")