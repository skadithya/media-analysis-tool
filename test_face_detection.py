import cv2

from analyzers.face_detection import detect_faces


image_path = "sample_media/test.jpg"

face_count, face_boxes, annotated_image = detect_faces(image_path)

print()
print("FACE DETECTION RESULTS")
print("=" * 50)

print(f"Faces detected: {face_count}")

for index, box in enumerate(face_boxes, start=1):
    print(f"Face {index}: {box}")

cv2.imwrite(
    "outputs/test_faces.jpg",
    annotated_image
)

print()
print("Annotated face image saved to:")
print("outputs/test_faces.jpg")