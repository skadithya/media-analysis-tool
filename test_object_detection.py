from analyzers.object_detection import detect_objects
import cv2


image_path = "sample_media/test.jpg"

detections, annotated_image = detect_objects(image_path)

print()
print("OBJECT DETECTION RESULTS")
print("=" * 50)

if not detections:
    print("No objects detected.")

else:
    for detection in detections:
        print(f"Object: {detection['object']}")
        print(f"Confidence: {detection['confidence']}")
        print(f"Bounding box: {detection['bounding_box']}")
        print("-" * 50)

cv2.imwrite("outputs/test_annotated.jpg", annotated_image)

print("\nAnnotated image saved to:")
print("outputs/test_annotated.jpg")