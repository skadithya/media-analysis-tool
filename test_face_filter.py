import cv2
import glob

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

keyframes = sorted(glob.glob("outputs/keyframes/*.jpg"))

threshold = 0.90
min_sizes = [0, 30, 50, 70, 100]

for min_size in min_sizes:

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

        count = 0

        if faces is not None:
            for face in faces:
                x, y, w, h = face[:4]

                if w >= min_size and h >= min_size:
                    count += 1

        counts.append(count)

    average = sum(counts) / len(counts)

    print(
        f"Min face size {min_size}px: "
        f"{counts} | Average: {average:.2f}"
    )