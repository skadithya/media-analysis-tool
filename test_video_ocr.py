from analyzers.keyframes import extract_keyframes
from analyzers.ocr import extract_text


video_path = "sample_media/test.mp4"

keyframes = extract_keyframes(video_path)

print()
print("VIDEO OCR")
print("=" * 60)

for frame_number, frame_path in enumerate(keyframes):

    timestamp = frame_number * 5

    results = extract_text(frame_path)

    print()
    print(f"Timestamp: {timestamp} seconds")

    if not results:
        print("No text detected.")
        continue

    for item in results:
        print(
            f"Text: {item['text']} | "
            f"Confidence: {item['confidence']} | "
            f"Box: {item['bounding_box']}"
        )