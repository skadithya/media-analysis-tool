import json

from analyzers.report_generator import generate_report


INPUT_PATH = "outputs/video_analysis.json"
OUTPUT_PATH = "outputs/video_report.md"


print("Loading video analysis evidence...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    analysis = json.load(f)


print("Sending video analysis to Qwen2.5...")
print("This may take some time because the video evidence is larger.")


report = generate_report(analysis)


with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)


print("\nVideo AI report generated successfully.")
print(f"Saved report to: {OUTPUT_PATH}")

print("\n" + "=" * 70)
print(report)
print("=" * 70)