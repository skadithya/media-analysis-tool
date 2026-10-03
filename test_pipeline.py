import json
from analyzers.pipeline import analyze_image

image_path = "sample_media/test.jpg"

result = analyze_image(image_path)

output_path = "outputs/image_analysis.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\nImage analysis completed.")
print(f"Saved result to: {output_path}")