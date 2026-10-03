import os
import subprocess


def extract_keyframes(video_path, output_dir="outputs/keyframes", interval=5):
    os.makedirs(output_dir, exist_ok=True)

    # Remove previously generated keyframes
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)

        if os.path.isfile(file_path):
            os.remove(file_path)

    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")

    command = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        output_pattern,
        "-y"
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=True
    )

    keyframes = []

    for filename in sorted(os.listdir(output_dir)):
        if filename.lower().endswith(".jpg"):
            keyframes.append(os.path.join(output_dir, filename))

    return keyframes