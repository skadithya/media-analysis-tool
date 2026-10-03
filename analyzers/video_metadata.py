import json
import subprocess


def extract_video_metadata(video_path):
    metadata = {
        "duration": "Not available",
        "resolution": "Not available",
        "codec": "Not available",
        "creation_date": "Not available"
    }

    try:
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        data = json.loads(result.stdout)

        # -----------------------------
        # Video stream information
        # -----------------------------
        video_stream = None

        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if video_stream:
            width = video_stream.get("width")
            height = video_stream.get("height")

            if width and height:
                metadata["resolution"] = f"{width}x{height}"

            codec = video_stream.get("codec_name")

            if codec:
                metadata["codec"] = codec

            duration = video_stream.get("duration")

            if duration:
                metadata["duration"] = round(float(duration), 2)

        # -----------------------------
        # Format information
        # -----------------------------
        format_data = data.get("format", {})

        if metadata["duration"] == "Not available":
            duration = format_data.get("duration")

            if duration:
                metadata["duration"] = round(float(duration), 2)

        # -----------------------------
        # Creation date
        # -----------------------------
        tags = format_data.get("tags", {})

        creation_date = (
            tags.get("creation_time")
            or tags.get("date")
            or tags.get("creation_date")
        )

        if creation_date:
            metadata["creation_date"] = creation_date

        return metadata

    except Exception as e:
        print(f"Error extracting video metadata: {e}")
        return metadata