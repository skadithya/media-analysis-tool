from analyzers.metadata import extract_image_metadata


image_path = "sample_media/test.jpg"

metadata = extract_image_metadata(image_path)

print()
print("IMAGE METADATA")
print("=" * 40)

for key, value in metadata.items():
    print(f"{key}: {value}")