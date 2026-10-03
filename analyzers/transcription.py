from faster_whisper import WhisperModel


# Small model gives a good balance between accuracy and CPU speed.
model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)


def transcribe_video(video_path):
    """
    Transcribe speech from a video and return
    timestamped segments.
    """

    segments, info = model.transcribe(
        video_path,
        beam_size=5,
        vad_filter=True
    )

    transcript = []

    for segment in segments:

        transcript.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return {
        "language": info.language,
        "language_probability": round(
            info.language_probability,
            4
        ),
        "segments": transcript
    }