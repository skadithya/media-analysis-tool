import cv2

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

_detectors = {}


def get_face_detector(score_threshold=0.6):
    """
    Create/reuse a YuNet face detector for the requested
    confidence threshold.
    """

    if score_threshold not in _detectors:
        _detectors[score_threshold] = cv2.FaceDetectorYN.create(
            MODEL_PATH,
            "",
            (320, 320),
            score_threshold,
            0.3,
            5000
        )

    return _detectors[score_threshold]


def detect_faces(
    image_path,
    score_threshold=0.6,
    min_face_size=0
):
    """
    Detect human faces in an image.

    Parameters:
        image_path: Path to image.
        score_threshold: Minimum YuNet confidence.
        min_face_size: Minimum width and height of detected face.

    Returns:
        face_count,
        face_boxes,
        annotated_image
    """

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    detector = get_face_detector(score_threshold)

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    face_boxes = []

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]
            confidence = float(face[14])

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # Ignore very small detections.
            if w < min_face_size or h < min_face_size:
                continue

            box = {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "confidence": round(confidence, 4)
            }

            face_boxes.append(box)

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

    return len(face_boxes), face_boxes, image