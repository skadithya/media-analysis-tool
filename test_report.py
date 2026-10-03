import json

from analyzers.report_generator import generate_report


input_path = "outputs/image_analysis.json"
output_path = "outputs/image_report.md"


with open(input_path, "r", encoding="utf-8") as f:
    analysis = json.load(f)


print("Sending image analysis to Qwen2.5...")

report = generate_report(analysis)


with open(output_path, "w", encoding="utf-8") as f:
    f.write(report)


print("\nAI report generated successfully.")
print(f"Saved report to: {output_path}")

print("\n" + "=" * 70)
print(report)
print("=" * 70)