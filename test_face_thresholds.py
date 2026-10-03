import cv2
import glob

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

keyframes = sorted(glob.glob("outputs/keyframes/*.jpg"))

if not keyframes:
    print("No keyframes found in outputs/keyframes/")
    exit()

thresholds = [0.6, 0.75, 0.8, 0.85, 0.9]

for threshold in thresholds:

    detector = cv2.FaceDetectorYN.create(
        MODEL_PATH,
        "",
        (320, 320),
        threshold,
        0.3,
        5000
    )

    counts = []

    for frame_path in keyframes:

        image = cv2.imread(frame_path)

        if image is None:
            continue

        height, width = image.shape[:2]

        detector.setInputSize((width, height))

        _, faces = detector.detect(image)

        count = 0 if faces is None else len(faces)

        counts.append(count)

    average = sum(counts) / len(counts)

    print(
        f"Threshold {threshold}: "
        f"{counts} | Average: {average:.2f}"
    )