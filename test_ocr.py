from analyzers.ocr import extract_text


image_path = "sample_media/test.jpg"

results = extract_text(image_path)

print()
print("OCR RESULTS")
print("=" * 60)

if not results:
    print("No text detected.")
else:
    for index, item in enumerate(results, start=1):
        print(f"Text {index}: {item['text']}")
        print(f"Confidence: {item['confidence']}")
        print(f"Bounding box: {item['bounding_box']}")
        print("-" * 60)