import tempfile
from collections import defaultdict, deque
from pathlib import Path

import subprocess
import imageio_ffmpeg
import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO


# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ADAS Collision Risk Demo",
    page_icon="🚗",
    layout="wide",
)

st.title("🚗 ADAS Computer Vision & Collision Risk Demo")

st.write(
    "Upload driving footage to detect objects, track vehicles, "
    "and inspect an experimental bounding-box growth indicator."
)

st.warning(
    "Research prototype only. This indicator is not validated "
    "collision prediction, calibrated time-to-collision, or a "
    "safety-certified ADAS system."
)


# --------------------------------------------------
# Model configuration
# --------------------------------------------------
MODEL_PATH = Path("yolo26n.pt")
CONFIDENCE_THRESHOLD = 0.25
HISTORY_LENGTH = 15
MIN_HISTORY = 8
GROWTH_THRESHOLD = 1.25

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


@st.cache_resource
def load_model():
    """Load YOLO once and reuse it across Streamlit reruns."""
    return YOLO(str(MODEL_PATH))


# --------------------------------------------------
# Video upload
# --------------------------------------------------
uploaded_video = st.file_uploader(
    "Upload a driving video",
    type=["mp4", "avi", "mov"],
)

if uploaded_video is None:
    st.info("Upload a video to begin the analysis.")
    st.stop()

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. "
        "Place yolo26n.pt in the project folder."
    )
    st.stop()

st.video(uploaded_video.getvalue())

process_video = st.button(
    "Run object detection and tracking",
    type="primary",
)

if not process_video:
    st.info("Click the button above to process the uploaded video.")
    st.stop()


# --------------------------------------------------
# Prepare temporary input and output files
# --------------------------------------------------
input_suffix = Path(uploaded_video.name).suffix.lower()

if input_suffix not in {".mp4", ".avi", ".mov"}:
    input_suffix = ".mp4"

with tempfile.TemporaryDirectory(prefix="adas_demo_") as temp_dir:
    temp_dir = Path(temp_dir)

    input_path = temp_dir / f"input{input_suffix}"
    output_path = temp_dir / "processed_video.mp4"
    csv_path = temp_dir / "analysis_results.csv"

    input_path.write_bytes(uploaded_video.getvalue())

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        st.error(
            "OpenCV could not open this video. "
            "Try another video encoded in a supported format."
        )
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        st.error("The video metadata is invalid or unsupported.")
        st.stop()

    # Keep the uploaded video's dimensions and frame rate.
    # Use an MP4-compatible codec for browser playback.
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        st.error("Could not initialize the output video writer.")
        st.stop()

    model = load_model()

    # Store recent bounding-box areas by tracked object ID.
    area_history = defaultdict(
        lambda: deque(maxlen=HISTORY_LENGTH)
    )

    records = []
    frame_number = 0
    possible_approach_count = 0
    unique_track_ids = set()

    progress = st.progress(0)
    status_text = st.empty()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # YOLO detection and ByteTrack tracking.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=CONFIDENCE_THRESHOLD,
                verbose=False,
            )

            annotated_frame = frame.copy()

            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()

                if boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = [None] * len(xyxy)

                for bbox, class_id, confidence, track_id in zip(
                    xyxy,
                    class_ids,
                    confidences,
                    track_ids,
                ):
                    class_name = model.names[int(class_id)]

                    # Keep the focus on vehicle tracking.
                    if class_name not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, bbox)

                    box_width = max(0, x2 - x1)
                    box_height = max(0, y2 - y1)
                    area = box_width * box_height

                    growth_ratio = None
                    indicator_status = "OBSERVING"

                    if track_id is not None:
                        unique_track_ids.add(track_id)

                        history = area_history[track_id]

                        if len(history) >= MIN_HISTORY:
                            previous_area = history[0]

                            if previous_area > 0:
                                growth_ratio = area / previous_area

                                if growth_ratio >= GROWTH_THRESHOLD:
                                    indicator_status = (
                                        "POSSIBLE APPROACH"
                                    )
                                    possible_approach_count += 1
                                else:
                                    indicator_status = (
                                        "NO STRONG GROWTH"
                                    )

                        history.append(area)

                    # Visualization only: green for no strong growth,
                    # orange for observing, red for possible approach.
                    if indicator_status == "POSSIBLE APPROACH":
                        box_color = (0, 0, 255)
                    elif indicator_status == "NO STRONG GROWTH":
                        box_color = (0, 200, 0)
                    else:
                        box_color = (0, 165, 255)

                    label = f"{class_name}"

                    if track_id is not None:
                        label += f" ID:{track_id}"

                    label += f" | {indicator_status}"

                    cv2.rectangle(
                        annotated_frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2,
                    )

                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        box_color,
                        2,
                    )

                    records.append(
                        {
                            "frame": frame_number,
                            "track_id": track_id,
                            "vehicle_class": class_name,
                            "confidence": round(
                                float(confidence), 4
                            ),
                            "bbox_area": area,
                            "area_growth_ratio": (
                                round(float(growth_ratio), 4)
                                if growth_ratio is not None
                                else None
                            ),
                            "indicator_status": indicator_status,
                        }
                    )

            writer.write(annotated_frame)

            if total_frames > 0:
                progress.progress(
                    min(frame_number / total_frames, 1.0)
                )

            status_text.write(
                f"Processed {frame_number:,} of "
                f"{total_frames:,} frames"
                if total_frames > 0
                else f"Processed {frame_number:,} frames"
            )

    finally:
        cap.release()
        writer.release()

    # Convert the processed video to browser-compatible H.264
    st.info("Converting video for browser playback...")

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    h264_path = temp_dir / "processed_video_h264.mp4"

    conversion = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", str(output_path),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            str(h264_path),
        ],
        capture_output=True,
        text=True,
    )

    if conversion.returncode != 0 or not h264_path.exists() or h264_path.stat().st_size == 0:
        st.error("Video conversion failed.")
        st.code(conversion.stderr[-3000:])
        st.stop()

    output_path = h264_path
    st.success("Video conversion complete!")

    progress.progress(1.0)
    status_text.success("Video processing complete!")

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------
    if not output_path.exists() or output_path.stat().st_size == 0:
        st.error("No processed video was generated.")
        st.stop()

    st.subheader("Processed video")
    st.video(output_path.read_bytes())

    results_df = pd.DataFrame(records)

    if not results_df.empty:
        results_df.to_csv(csv_path, index=False)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Vehicle observations",
            f"{len(results_df):,}",
        )

        col2.metric(
            "Unique tracked IDs",
            f"{results_df['track_id'].nunique():,}",
        )

        col3.metric(
            "Possible-approach observations",
            f"{possible_approach_count:,}",
        )

        st.subheader("Analysis results")

        st.bar_chart(
            results_df["indicator_status"].value_counts()
        )

        st.subheader("Vehicle observations")
        st.dataframe(results_df.head(100), use_container_width=True)

        st.download_button(
            "Download analysis CSV",
            data=csv_path.read_bytes(),
            file_name="adas_analysis_results.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "No cars, trucks, buses, or motorcycles were detected "
            "with the current confidence threshold."
        )

st.caption(
    "Experimental computer vision demonstration. "
    "Bounding-box growth is affected by perspective, camera motion, "
    "occlusion, and object movement. It does not establish collision risk."
)