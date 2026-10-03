import json
from analyzers.pipeline import analyze_video

video_path = "sample_media/test.mp4"

result = analyze_video(video_path)

output_path = "outputs/video_analysis.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("\nVideo analysis completed.")
print(f"Saved result to: {output_path}")