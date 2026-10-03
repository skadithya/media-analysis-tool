import easyocr

reader = easyocr.Reader(["en"], gpu=False)


def extract_text(image_path):
    results = reader.readtext(image_path)

    extracted_text = []

    for result in results:
        bounding_box = result[0]
        text = result[1]
        confidence = float(result[2])

        # Ignore extremely low-confidence detections
        if confidence < 0.3:
            continue

        clean_box = [
            [int(point[0]), int(point[1])]
            for point in bounding_box
        ]

        extracted_text.append({
            "text": text,
            "confidence": round(confidence, 4),
            "bounding_box": clean_box
        })

    return extracted_text