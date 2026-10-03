from analyzers.transcription import transcribe_video


video_path = "sample_media/test.mp4"

result = transcribe_video(video_path)


print()
print("VIDEO SPEECH TRANSCRIPTION")
print("=" * 60)

print(f"Detected language: {result['language']}")
print(
    f"Language probability: "
    f"{result['language_probability']}"
)

print()
print("TRANSCRIPT")
print("-" * 60)

if not result["segments"]:
    print("No speech detected.")
else:

    for segment in result["segments"]:

        print(
            f"[{segment['start']}s - "
            f"{segment['end']}s] "
            f"{segment['text']}"
        )