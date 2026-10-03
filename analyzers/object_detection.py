from ultralytics import YOLO


# Load the pretrained YOLO model once.
# 'yolo11n.pt' is a lightweight model suitable for local inference.
model = YOLO("yolo11n.pt")


def detect_objects(image_path):
    """
    Detect objects in an image.

    Returns:
        detections: list of dictionaries containing
        object name, confidence, and bounding box.
        annotated_image: image with bounding boxes drawn.
    """

    results = model(image_path, verbose=False)

    detections = []

    result = results[0]

    boxes = result.boxes

    for box in boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        object_name = model.names[class_id]

        detection = {
            "object": object_name,
            "confidence": round(confidence, 4),
            "bounding_box": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2)
            }
        }

        detections.append(detection)

    # YOLO automatically draws bounding boxes.
    annotated_image = result.plot()

    return detections, annotated_image