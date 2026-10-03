import os
import json
import tempfile

import streamlit as st

from analyzers.pipeline import analyze_image, analyze_video
from analyzers.report_generator import generate_report


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Media Analysis Tool",
    page_icon="🔎",
    layout="wide"
)


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def save_uploaded_file(uploaded_file):
    """Save an uploaded file temporarily and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    )

    temp_file.write(uploaded_file.getbuffer())
    temp_file.close()

    return temp_file.name


def display_metadata(metadata):
    """Display metadata as a simple table."""
    if not metadata:
        st.info("No metadata available.")
        return

    rows = []

    for key, value in metadata.items():
        if isinstance(value, dict):
            value = json.dumps(value)

        rows.append({
            "Field": key.replace("_", " ").title(),
            "Value": value
        })

    st.table(rows)


def display_objects(objects):
    """Display object detections."""
    if not objects:
        st.info("No objects detected.")
        return

    rows = []

    for detection in objects:
        box = detection.get("bounding_box", {})

        rows.append({
            "Object": detection.get("object", "Not available"),
            "Confidence": detection.get("confidence", "Not available"),
            "X1": box.get("x1", "Not available"),
            "Y1": box.get("y1", "Not available"),
            "X2": box.get("x2", "Not available"),
            "Y2": box.get("y2", "Not available")
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True
    )


def display_faces(faces):
    """Display face detection information."""
    if not faces:
        st.info("Face information not available.")
        return

    st.metric(
        "Detected faces",
        faces.get("count", "Not available")
    )

    detections = faces.get("detections", [])

    if detections:
        rows = []

        for face in detections:
            rows.append({
                "Confidence": face.get(
                    "confidence",
                    "Not available"
                ),
                "X1": face.get("x1", "Not available"),
                "Y1": face.get("y1", "Not available"),
                "X2": face.get("x2", "Not available"),
                "Y2": face.get("y2", "Not available")
            })

        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True
        )


def display_ocr(ocr_results):
    """Display OCR results."""
    if not ocr_results:
        st.info("No text detected.")
        return

    rows = []

    for item in ocr_results:
        rows.append({
            "Text": item.get("text", "Not available"),
            "Confidence": item.get(
                "confidence",
                "Not available"
            ),
            "Bounding Box": item.get(
                "bounding_box",
                "Not available"
            )
        })

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True
    )


# ---------------------------------------------------------
# Application
# ---------------------------------------------------------

st.title("🔎 Media Analysis Tool")

st.write(
    "Upload an image or video to extract metadata, "
    "detect objects and faces, read visible text, "
    "analyze video tracking, transcribe speech, "
    "and generate an evidence-grounded report."
)

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "bmp",
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm"
    ]
)


if uploaded_file is not None:

    st.success(
        f"Selected file: {uploaded_file.name}"
    )

    # Show uploaded media
    file_extension = os.path.splitext(
        uploaded_file.name
    )[1].lower()

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    }

    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    }

    if file_extension in image_extensions:
        media_type = "image"

        st.image(
            uploaded_file,
            caption="Uploaded image",
            use_container_width=True
        )

    elif file_extension in video_extensions:
        media_type = "video"

        st.video(
            uploaded_file
        )

    else:
        media_type = None
        st.error("Unsupported media type.")

    # -----------------------------------------------------
    # Analyze button
    # -----------------------------------------------------

    if media_type:

        analyze_button = st.button(
            "🔍 Analyze Media",
            type="primary",
            use_container_width=True
        )

        if analyze_button:

            temp_path = None

            try:

                temp_path = save_uploaded_file(
                    uploaded_file
                )

                # -------------------------------------------------
                # IMAGE
                # -------------------------------------------------

                if media_type == "image":

                    with st.spinner(
                        "Analyzing image..."
                    ):

                        analysis = analyze_image(
                            temp_path
                        )

                        analysis["file_name"] = uploaded_file.name

                        report = generate_report(
                            analysis
                        )

                    st.success(
                        "Image analysis completed."
                    )
                    st.header(
                        "Detected Regions"
                    )

                    annotated_image_path = analysis.get(
                        "annotated_image"
                    )

                    if (
                        annotated_image_path
                        and os.path.exists(annotated_image_path)
                    ):
                        st.image(
                            annotated_image_path,
                            caption="Detected objects with bounding boxes",
                            use_container_width=True
                    )
                    else:
                        st.info(
                            "Annotated detection image could not be generated."
                        )

                    # ---------------------------------------------
                    # Metadata
                    # ---------------------------------------------

                    st.header("1. Metadata")

                    display_metadata(
                        analysis.get(
                            "metadata",
                            {}
                        )
                    )

                    # ---------------------------------------------
                    # Objects
                    # ---------------------------------------------

                    st.header(
                        "2. Object Detection"
                    )

                    display_objects(
                        analysis.get(
                            "objects",
                            []
                        )
                    )

                    # ---------------------------------------------
                    # Faces
                    # ---------------------------------------------

                    st.header(
                        "3. Face Detection"
                    )

                    display_faces(
                        analysis.get(
                            "faces",
                            {}
                        )
                    )

                    # ---------------------------------------------
                    # OCR
                    # ---------------------------------------------

                    st.header(
                        "4. OCR"
                    )

                    display_ocr(
                        analysis.get(
                            "ocr",
                            []
                        )
                    )

                    # ---------------------------------------------
                    # Speech
                    # ---------------------------------------------

                    st.header(
                        "5. Speech Analysis"
                    )

                    speech = analysis.get(
                        "speech",
                        {}
                    )

                    st.info(
                        speech.get(
                            "reason",
                            "Speech is not applicable to an image."
                        )
                    )

                    # ---------------------------------------------
                    # Report
                    # ---------------------------------------------

                    st.header(
                        "6. Evidence-Grounded Report"
                    )

                    st.markdown(
                        report
                    )

                    # ---------------------------------------------
                    # Raw analysis
                    # ---------------------------------------------

                    with st.expander(
                        "View raw analysis JSON"
                    ):

                        st.json(
                            analysis
                        )

                # -------------------------------------------------
                # VIDEO
                # -------------------------------------------------

                elif media_type == "video":

                    with st.spinner(
                        "Analyzing video. "
                        "This may take several minutes..."
                    ):

                        analysis = analyze_video(
                            temp_path,
                            keyframe_interval=5
                        )
                        analysis["file_name"] = uploaded_file.name

                        report = generate_report(
                            analysis
                        )

                    st.success(
                        "Video analysis completed."
                    )

                    # ---------------------------------------------
                    # Metadata
                    # ---------------------------------------------

                    st.header(
                        "1. Video Metadata"
                    )

                    display_metadata(
                        analysis.get(
                            "metadata",
                            {}
                        )
                    )

                    # ---------------------------------------------
                    # Keyframes
                    # ---------------------------------------------

                    st.header(
                        "2. Keyframe Analysis"
                    )

                    keyframes = analysis.get(
                        "keyframes",
                        []
                    )

                    st.write(
                        f"Analyzed {len(keyframes)} "
                        "sampled keyframes."
                    )

                    for frame in keyframes:

                        timestamp = frame.get(
                            "timestamp",
                            "Not available"
                        )

                        with st.expander(
                            f"Keyframe at {timestamp}s"
                        ):

                            annotated_frame_path = frame.get(
                                "annotated_frame"
                            )

                            if (
                                annotated_frame_path
                                and os.path.exists(
                                    annotated_frame_path
                                )
                            ):
                                st.image(
                                    annotated_frame_path,
                                    caption=(
                                        f"Detected regions at "
                                        f"{timestamp}s"
                                    ),
                                    use_container_width=True
                                )

                            else:
                                frame_path = os.path.join(
                                    "outputs",
                                    "keyframes",
                                    frame.get(
                                        "frame",
                                        ""
                                    )
                                )

                                if os.path.exists(
                                    frame_path
                                ):
                                    st.image(
                                        frame_path,
                                        caption=(
                                            f"Timestamp: "
                                            f"{timestamp}s"
                                        ),
                                        use_container_width=True
                                    )
                            st.subheader(
                                "Objects"
                            )

                            display_objects(
                                frame.get(
                                    "objects",
                                    []
                                )
                            )

                            st.subheader(
                                "Faces"
                            )

                            display_faces(
                                frame.get(
                                    "faces",
                                    {}
                                )
                            )

                            st.subheader(
                                "OCR"
                            )

                            display_ocr(
                                frame.get(
                                    "ocr",
                                    []
                                )
                            )

                    # ---------------------------------------------
                    # Tracking
                    # ---------------------------------------------

                    st.header(
                        "3. Object Tracking"
                    )

                    tracking = analysis.get(
                        "tracking",
                        {}
                    )

                    tracks = tracking.get(
                        "tracks",
                        []
                    )

                    if tracks:

                        tracking_rows = []

                        for track in tracks:

                            tracking_rows.append({
                                "Track ID": track.get(
                                    "track_id",
                                    "Not available"
                                ),
                                "Object": track.get(
                                    "object",
                                    "Not available"
                                ),
                                "First Timestamp": track.get(
                                    "first_timestamp",
                                    "Not available"
                                ),
                                "Last Timestamp": track.get(
                                    "last_timestamp",
                                    "Not available"
                                ),
                                "Max Confidence": track.get(
                                    "max_confidence",
                                    "Not available"
                                ),
                                "Observations": track.get(
                                    "observations",
                                    "Not available"
                                )
                            })

                        st.dataframe(
                            tracking_rows,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:
                        st.info(
                            "No tracking information available."
                        )

                    # ---------------------------------------------
                    # Speech transcription
                    # ---------------------------------------------

                    st.header(
                        "4. Speech Transcription"
                    )

                    speech = analysis.get(
                        "speech",
                        {}
                    )

                    language = speech.get(
                        "language",
                        "Not available"
                    )

                    probability = speech.get(
                        "language_probability",
                        "Not available"
                    )

                    st.write(
                        f"**Detected language:** {language}"
                    )

                    st.write(
                        f"**Language probability:** "
                        f"{probability}"
                    )

                    segments = speech.get(
                        "segments",
                        []
                    )

                    if segments:

                        transcript_rows = []

                        for segment in segments:

                            transcript_rows.append({
                                "Start": segment.get(
                                    "start",
                                    "Not available"
                                ),
                                "End": segment.get(
                                    "end",
                                    "Not available"
                                ),
                                "Text": segment.get(
                                    "text",
                                    "Not available"
                                )
                            })

                        st.dataframe(
                            transcript_rows,
                            use_container_width=True,
                            hide_index=True
                        )

                    else:
                        st.info(
                            "No speech transcript available."
                        )

                    # ---------------------------------------------
                    # Report
                    # ---------------------------------------------

                    st.header(
                        "5. Evidence-Grounded Report"
                    )

                    st.markdown(
                        report
                    )

                    # ---------------------------------------------
                    # Raw analysis
                    # ---------------------------------------------

                    with st.expander(
                        "View raw analysis JSON"
                    ):

                        st.json(
                            analysis
                        )

            except Exception as e:

                st.error(
                    "Analysis failed."
                )

                st.exception(e)

            finally:

                if temp_path and os.path.exists(
                    temp_path
                ):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass