import json
import copy
import urllib.request
import urllib.error


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


def prepare_evidence(analysis):
    """
    Normalize raw machine-generated evidence.

    The original analysis JSON is never modified.
    """

    evidence = copy.deepcopy(analysis)

    evidence.pop("annotated_image", None)

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    metadata = evidence.get("metadata", {})

    for key, value in metadata.items():
        if value == "Not available":
            metadata[key] = "Could not be determined."

    # -----------------------------------------------------
    # Image objects
    # -----------------------------------------------------

    for obj in evidence.get("objects", []):

        confidence = obj.get("confidence")

        if isinstance(confidence, (int, float)):

            if confidence < 0.50:
                obj["evidence_status"] = "Unverified / low confidence"
            else:
                obj["evidence_status"] = "Detected by model"

    # -----------------------------------------------------
    # Image OCR
    # -----------------------------------------------------

    for item in evidence.get("ocr", []):

        confidence = item.get("confidence")

        if isinstance(confidence, (int, float)):

            if confidence < 0.70:
                item["evidence_status"] = "Unverified / low confidence"
            else:
                item["evidence_status"] = "Detected by OCR model"

    # -----------------------------------------------------
    # Video keyframes
    # -----------------------------------------------------

    for frame in evidence.get("keyframes", []):

        for obj in frame.get("objects", []):

            confidence = obj.get("confidence")

            if isinstance(confidence, (int, float)):

                if confidence < 0.50:
                    obj["evidence_status"] = (
                        "Unverified / low confidence"
                    )
                else:
                    obj["evidence_status"] = "Detected by model"

        for item in frame.get("ocr", []):

            confidence = item.get("confidence")

            if isinstance(confidence, (int, float)):

                if confidence < 0.70:
                    item["evidence_status"] = (
                        "Unverified / low confidence"
                    )
                else:
                    item["evidence_status"] = (
                        "Detected by OCR model"
                    )

    # -----------------------------------------------------
    # Video tracking
    # -----------------------------------------------------

    tracking = evidence.get("tracking", {})

    for track in tracking.get("tracks", []):

        confidence = track.get("max_confidence")

        if isinstance(confidence, (int, float)):

            if confidence < 0.50:
                track["evidence_status"] = (
                    "Unverified / low confidence"
                )
            else:
                track["evidence_status"] = "Tracked by model"

    return evidence


# ---------------------------------------------------------
# Compact video evidence
# ---------------------------------------------------------

def build_video_evidence_summary(evidence):

    metadata = evidence.get("metadata", {})

    lines = []

    lines.append("VIDEO METADATA")
    lines.append(
        f"Duration: {metadata.get('duration', 'Could not be determined')}"
    )
    lines.append(
        f"Resolution: {metadata.get('resolution', 'Could not be determined')}"
    )
    lines.append(
        f"Codec: {metadata.get('codec', 'Could not be determined')}"
    )
    lines.append(
        f"Creation date: {metadata.get('creation_date', 'Could not be determined')}"
    )

    # -----------------------------------------------------
    # Keyframes
    # -----------------------------------------------------

    lines.append("")
    lines.append("KEYFRAME OBSERVATIONS")

    for frame in evidence.get("keyframes", []):

        timestamp = frame.get("timestamp")

        lines.append(f"Timestamp: {timestamp}s")

        objects = frame.get("objects", [])

        if objects:

            for obj in objects:

                lines.append(
                    f"  Object: {obj.get('object')} | "
                    f"Confidence: {obj.get('confidence')} | "
                    f"Status: {obj.get('evidence_status')}"
                )

        faces = frame.get("faces", {})

        face_count = faces.get("count", 0)

        lines.append(
            f"  Faces detected in sampled frame: {face_count}"
        )

        ocr_items = frame.get("ocr", [])

        for item in ocr_items:

            lines.append(
                f"  OCR: {item.get('text')} | "
                f"Confidence: {item.get('confidence')} | "
                f"Status: {item.get('evidence_status')}"
            )

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------

    lines.append("")
    lines.append("OBJECT TRACKING")

    tracks = evidence.get(
        "tracking",
        {}
    ).get(
        "tracks",
        []
    )

    # Keep useful tracks.
    useful_tracks = [
        track
        for track in tracks
        if track.get("observations", 0) >= 30
    ]

    # Limit the number sent to the small LLM.
    useful_tracks = useful_tracks[:40]

    for track in useful_tracks:

        lines.append(
            f"Track ID: {track.get('track_id')} | "
            f"Object: {track.get('object')} | "
            f"First: {track.get('first_timestamp')}s | "
            f"Last: {track.get('last_timestamp')}s | "
            f"Max confidence: {track.get('max_confidence')} | "
            f"Observations: {track.get('observations')} | "
            f"Status: {track.get('evidence_status')}"
        )

    # -----------------------------------------------------
    # Speech
    # -----------------------------------------------------

    lines.append("")
    lines.append("SPEECH TRANSCRIPTION")

    speech = evidence.get("speech", {})

    lines.append(
        f"Language: {speech.get('language', 'Could not be determined')}"
    )

    lines.append(
        f"Language probability: "
        f"{speech.get('language_probability', 'Could not be determined')}"
    )

    for segment in speech.get("segments", []):

        lines.append(
            f"[{segment.get('start')}s - "
            f"{segment.get('end')}s] "
            f"{segment.get('text')}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------
# AI evidence review + deterministic findings
# ---------------------------------------------------------

def generate_llm_findings(evidence_summary):
    """
    Generate deterministic, evidence-grounded findings.

    The local LLM is still called as an evidence-review step, but its
    free-form prose is NOT inserted directly into the final report.
    The final findings are constructed only from the supplied evidence.
    """

    # ---------------------------------------------------------
    # 1. Ask the local LLM to review the evidence.
    #    We intentionally do NOT trust its free-form response.
    # ---------------------------------------------------------
    review_prompt = f"""
Review the following media-analysis evidence.

Return ONLY this JSON object:

{{
  "reviewed": true
}}

Do not provide any explanation, summary, names, identities,
predictions, interpretations, or additional text.

EVIDENCE:
{evidence_summary}
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": review_prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 42
        }
    }

    try:
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=180) as response:
            response_data = json.loads(response.read().decode("utf-8"))

        # The model response is intentionally not used as report text.
        _ = response_data.get("response", "")

    except Exception as e:
        print(f"LLM evidence review unavailable: {e}")

    # ---------------------------------------------------------
    # 2. Parse only factual values already present in evidence.
    # ---------------------------------------------------------
    lines = evidence_summary.splitlines()

    def get_value(prefix):
        for line in lines:
            if line.strip().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return "Could not be determined from the provided evidence."

    duration = get_value("Duration")
    resolution = get_value("Resolution")
    codec = get_value("Codec")
    creation_date = get_value("Creation date")
    language = get_value("Language")
    language_probability = get_value("Language probability")

    # ---------------------------------------------------------
    # 3. Face observations from sampled keyframes
    # ---------------------------------------------------------
    face_counts = []

    for line in lines:
        if "Faces detected in sampled frame:" in line:
            try:
                value = line.split(":", 1)[1].strip()
                face_counts.append(int(value))
            except ValueError:
                pass

    # ---------------------------------------------------------
    # 4. Count object/OCR evidence without inventing facts
    # ---------------------------------------------------------
    object_count = 0
    unverified_object_count = 0

    ocr_count = 0
    unverified_ocr_count = 0

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Object:"):
            object_count += 1

        if stripped.startswith("Object:") and "Unverified" in stripped:
            unverified_object_count += 1

        if stripped.startswith("OCR:"):
            ocr_count += 1

        if stripped.startswith("OCR:") and "Unverified" in stripped:
            unverified_ocr_count += 1

    # ---------------------------------------------------------
    # 5. Tracking evidence
    # ---------------------------------------------------------
    track_lines = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("Track ID:"):
            track_lines.append(stripped)

    # ---------------------------------------------------------
    # 6. Speech segment count
    # ---------------------------------------------------------
    speech_segments = 0

    in_speech_section = False

    for line in lines:
        stripped = line.strip()

        if stripped == "SPEECH TRANSCRIPTION":
            in_speech_section = True
            continue

        if in_speech_section:
            if (
                ("->" in stripped and "s" in stripped)
                or (" - " in stripped and "s" in stripped) 
            ):
                speech_segments += 1

    # ---------------------------------------------------------
    # 7. Build deterministic findings
    # ---------------------------------------------------------
    findings = []

    findings.append(
    f"- Video metadata reports a duration of {duration}, "
    f"resolution of {resolution}, and codec of {codec}. "
    f"Creation date: {creation_date.rstrip('.') if isinstance(creation_date, str) else creation_date}."
)

    if face_counts:
        minimum_faces = min(face_counts)
        maximum_faces = max(face_counts)

        findings.append(
            f"- Face detection on the sampled keyframes detected "
            f"between {minimum_faces} and {maximum_faces} faces per "
            f"sampled frame. These detections are machine-generated "
            f"and do not establish unique identities."
        )
    else:
        findings.append(
            "- Face-count information could not be determined from "
            "the supplied evidence."
        )

    findings.append(
        f"- Object detection produced {object_count} object detections "
        f"across the sampled keyframes. {unverified_object_count} "
        f"detections were marked unverified because of low confidence."
    )

    findings.append(
        f"- OCR produced {ocr_count} text detections across the sampled "
        f"keyframes. {unverified_ocr_count} OCR detections were marked "
        f"unverified because of low confidence."
    )

    findings.append(
        f"- Object tracking produced {len(track_lines)} tracked "
        f"object records. Tracking timestamps describe when the "
        f"tracking model observed those objects and do not establish "
        f"identity."
    )

    if language != "Could not be determined from the provided evidence.":
        findings.append(
            f"- The speech model detected language '{language}' with "
            f"a language probability of {language_probability}. "
            f"The transcript is machine-generated and may contain "
            f"recognition errors."
        )

    if speech_segments > 0:
        findings.append(
            f"- The speech transcription contains {speech_segments} "
            f"timestamped transcript segments. The transcription is "
            f"reported as machine-generated evidence and is not "
            f"independent verification of the statements within it."
        )
    else:
        findings.append(
            "- No timestamped speech segments were available in the "
            "provided evidence."
        )

    findings.append(
        "- Information not directly measured or contained in the "
        "collected metadata, detections, OCR results, tracking data, "
        "or transcription cannot be determined from the provided "
        "evidence."
    )

    return "\n".join(findings)


def build_image_report(evidence):

    metadata = evidence.get("metadata", {})

    report = []

    report.append("# Media Analysis Report")
    report.append("")

    report.append("## 1. Media Overview")
    report.append("")
    report.append(
        f"- Media type: {evidence.get('media_type', 'image')}"
    )
    report.append(
        f"- File name: {evidence.get('file_name', 'Could not be determined')}"
    )

    report.append("")
    report.append("## 2. Metadata")
    report.append("")

    report.append(
        f"- Camera make: "
        f"{metadata.get('camera_make', 'Could not be determined')}"
    )

    report.append(
        f"- Camera model: "
        f"{metadata.get('camera_model', 'Could not be determined')}"
    )

    report.append(
        f"- Date taken: "
        f"{metadata.get('date_taken', 'Could not be determined')}"
    )

    report.append(
        f"- Editing software: "
        f"{metadata.get('software', 'Could not be determined')}"
    )

    report.append(
        f"- GPS: "
        f"{metadata.get('gps', 'Could not be determined')}"
    )

    # Objects
    report.append("")
    report.append("## 3. Object Detection")
    report.append("")

    for obj in evidence.get("objects", []):

        report.append(
            f"- **{obj.get('object')}** "
            f"(confidence: {obj.get('confidence')})"
        )

        if obj.get("evidence_status") == "Unverified / low confidence":
            report.append(
                "  - **Unverified / low confidence**"
            )

    # Faces
    faces = evidence.get("faces", {})

    report.append("")
    report.append("## 4. Face Detection")
    report.append("")

    report.append(
        f"- Detected faces: {faces.get('count', 0)}"
    )

    report.append(
        "- Face detection is machine-generated and does not identify "
        "the person."
    )

    # OCR
    report.append("")
    report.append("## 5. OCR / Visible Text")
    report.append("")

    for item in evidence.get("ocr", []):

        report.append(
            f"- {item.get('text')} "
            f"(confidence: {item.get('confidence')})"
        )

        if item.get("evidence_status") == "Unverified / low confidence":
            report.append(
                "  - **Unverified / low confidence**"
            )

    # Speech
    report.append("")
    report.append("## 6. Speech")
    report.append("")

    report.append(
        "- Speech analysis is not applicable to an image."
    )

    # Findings
    report.append("")
    report.append("## 7. Evidence-Based Findings")
    report.append("")

    report.append(
        "- The machine-analysis pipeline detected the objects, "
        "face, and text listed above."
    )

    # Unverified
    report.append("")
    report.append("## 8. Unverified / Could Not Be Determined")
    report.append("")

    for obj in evidence.get("objects", []):

        if obj.get("evidence_status") == "Unverified / low confidence":

            report.append(
                f"- Object detection: {obj.get('object')} "
                f"(confidence {obj.get('confidence')})"
            )

    for item in evidence.get("ocr", []):

        if item.get("evidence_status") == "Unverified / low confidence":

            report.append(
                f"- OCR: {item.get('text')} "
                f"(confidence {item.get('confidence')})"
            )

    for key, value in metadata.items():

        if value == "Could not be determined.":

            report.append(
                f"- {key}: Could not be determined."
            )

    return "\n".join(report)


# ---------------------------------------------------------
# Video report
# ---------------------------------------------------------

def build_video_report(evidence, llm_findings):

    metadata = evidence.get("metadata", {})

    report = []

    report.append("# Media Analysis Report")
    report.append("")

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------

    report.append("## 1. Media Overview")
    report.append("")

    report.append(
        f"- Media type: video"
    )

    report.append(
        f"- File name: "
        f"{evidence.get('file_name', 'Could not be determined')}"
    )

    # -----------------------------------------------------
    # Metadata
    # -----------------------------------------------------

    report.append("")
    report.append("## 2. Video Metadata")
    report.append("")

    report.append(
        f"- Duration: "
        f"{metadata.get('duration', 'Could not be determined')}"
    )

    report.append(
        f"- Resolution: "
        f"{metadata.get('resolution', 'Could not be determined')}"
    )

    report.append(
        f"- Codec: "
        f"{metadata.get('codec', 'Could not be determined')}"
    )

    report.append(
        f"- Creation date: "
        f"{metadata.get('creation_date', 'Could not be determined')}"
    )

    # -----------------------------------------------------
    # Keyframes
    # -----------------------------------------------------

    report.append("")
    report.append("## 3. Keyframe Analysis")
    report.append("")

    for frame in evidence.get("keyframes", []):

        timestamp = frame.get("timestamp")

        report.append(
            f"### Keyframe at approximately {timestamp}s"
        )
        report.append("")

        objects = frame.get("objects", [])

        if objects:

            for obj in objects:

                line = (
                    f"- Object: {obj.get('object')} "
                    f"(confidence: {obj.get('confidence')})"
                )

                if (
                    obj.get("evidence_status")
                    == "Unverified / low confidence"
                ):
                    line += " - **Unverified / low confidence**"

                report.append(line)

        faces = frame.get("faces", {})

        report.append(
            f"- Faces detected in sampled frame: "
            f"{faces.get('count', 0)}"
        )

        for item in frame.get("ocr", []):

            line = (
                f"- OCR: {item.get('text')} "
                f"(confidence: {item.get('confidence')})"
            )

            if (
                item.get("evidence_status")
                == "Unverified / low confidence"
            ):
                line += " - **Unverified / low confidence**"

            report.append(line)

    # -----------------------------------------------------
    # Tracking
    # -----------------------------------------------------

    report.append("")
    report.append("## 4. Object Tracking")
    report.append("")

    report.append(
        "Tracking IDs are model-generated tracking instances and "
        "do not represent confirmed real-world identities."
    )

    tracks = evidence.get(
        "tracking",
        {}
    ).get(
        "tracks",
        []
    )

    useful_tracks = [
        track
        for track in tracks
        if track.get("observations", 0) >= 30
    ]

    useful_tracks = useful_tracks[:40]

    for track in useful_tracks:

        line = (
            f"- Track {track.get('track_id')}: "
            f"{track.get('object')} | "
            f"{track.get('first_timestamp')}s - "
            f"{track.get('last_timestamp')}s | "
            f"max confidence: "
            f"{track.get('max_confidence')} | "
            f"observations: "
            f"{track.get('observations')}"
        )

        if (
            track.get("evidence_status")
            == "Unverified / low confidence"
        ):
            line += " - **Unverified / low confidence**"

        report.append(line)

    # -----------------------------------------------------
    # Speech
    # -----------------------------------------------------

    report.append("")
    report.append("## 5. Speech Transcription")
    report.append("")

    speech = evidence.get("speech", {})

    report.append(
        f"- Detected language: "
        f"{speech.get('language', 'Could not be determined')}"
    )

    report.append(
        f"- Language probability: "
        f"{speech.get('language_probability', 'Could not be determined')}"
    )

    report.append("")
    report.append(
        "The following is a machine-generated transcription and "
        "may contain errors:"
    )
    report.append("")

    for segment in speech.get("segments", []):

        report.append(
            f"- [{segment.get('start')}s - "
            f"{segment.get('end')}s] "
            f"{segment.get('text')}"
        )

    # -----------------------------------------------------
    # AI findings
    # -----------------------------------------------------

    report.append("")
    report.append("## 6. Evidence-Based Findings")
    report.append("")

    report.append(llm_findings)

    # -----------------------------------------------------
    # Limitations
    # -----------------------------------------------------

    report.append("")
    report.append("## 7. Unverified / Could Not Be Determined")
    report.append("")

    report.append(
        "- Face counts are based on sampled keyframes and do not "
        "represent a confirmed unique-person count for the entire video."
    )

    report.append(
        "- Tracking IDs are model-generated tracking instances and "
        "are not confirmed identities."
    )

    report.append(
        "- OCR results with confidence below 0.70 are marked "
        "unverified."
    )

    report.append(
        "- Object detections with confidence below 0.50 are marked "
        "unverified."
    )

    for key, value in metadata.items():

        if value == "Could not be determined.":

            report.append(
                f"- {key}: Could not be determined."
            )

    return "\n".join(report)


# ---------------------------------------------------------
# Main public function
# ---------------------------------------------------------

def generate_report(analysis):

    evidence = prepare_evidence(analysis)

    media_type = evidence.get(
        "media_type",
        "unknown"
    )

    # -----------------------------------------------------
    # Image
    # -----------------------------------------------------

    if media_type == "image":

        return build_image_report(evidence)

    # -----------------------------------------------------
    # Video
    # -----------------------------------------------------

    if media_type == "video":

        print("Preparing compact video evidence...")

        evidence_summary = build_video_evidence_summary(
            evidence
        )

        print("Sending compact evidence to Qwen2.5...")

        llm_findings = generate_llm_findings(
            evidence_summary
        )

        return build_video_report(
            evidence,
            llm_findings
        )

    # -----------------------------------------------------
    # Unknown
    # -----------------------------------------------------

    return (
        "# Media Analysis Report\n\n"
        "Media type could not be determined."
    )