import csv
from collections import defaultdict, deque
from pathlib import Path
import cv2
from ultralytics import YOLO


# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "yolo26n.pt"
VIDEO_PATH = "driving.mp4"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "collision_risk_analysis_v2.mp4"

OUTPUT_CSV = OUTPUT_DIR / "approach_indicator_log.csv"

CSV_COLUMNS = [
    "frame",
    "track_id",
    "vehicle_class",
    "confidence",
    "bbox_area",
    "area_growth_ratio",
    "indicator_status",
]

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}

HISTORY_LENGTH = 15
CONFIDENCE_THRESHOLD = 0.25

# Require multiple observations before evaluating apparent growth.
MIN_HISTORY = 8

# Experimental threshold for the image-based indicator.
GROWTH_THRESHOLD = 1.25


# -----------------------------
# Initialize
# -----------------------------
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Cannot open {VIDEO_PATH}. "
        "Check that driving.mp4 is in the project folder."
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    fps = 30.0

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create the output video.")

# Store recent bounding-box areas for each tracking ID.
area_history = defaultdict(lambda: deque(maxlen=HISTORY_LENGTH))

frame_count = 0
vehicle_ids = set()
risk_counts = defaultdict(int)

print("Starting experimental ADAS risk analysis...")
print(f"Input video: {VIDEO_PATH}")

csv_file = open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8",
)

csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=CSV_COLUMNS,
)

csv_writer.writeheader()


# -----------------------------
# Process video
# -----------------------------
try:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        frame_count += 1

        result = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
        )[0]

        # Draw detection boxes without Ultralytics' text labels.
        annotated = result.plot(labels=False, conf=False)

        if result.boxes is not None and result.boxes.id is not None:
            boxes = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()

            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, class_id, box_confidence in zip(
                boxes, track_ids, class_ids, confidences
            ):
                class_name = result.names[class_id]

                if class_name not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box

                box_width = max(0.0, x2 - x1)
                box_height = max(0.0, y2 - y1)
                area = box_width * box_height

                if area <= 0:
                    continue

                history = area_history[track_id]
                history.append(area)
                growth_ratio = None
                vehicle_ids.add(track_id)

                label = f"ID {track_id} | {class_name}"

                old_area = history[0] if len(history) >= MIN_HISTORY else None


                # Keep the vehicle class and tracking ID in a compact label.
                label = f"{class_name} ID:{track_id}"

                # Default to insufficient evidence.
                status = "OBSERVING"

                # Compare the current area with an earlier observation.
                if len(history) >= MIN_HISTORY:
                    old_area = history[0]
                    growth_ratio = area / max(old_area, 1.0)

                    if growth_ratio >= GROWTH_THRESHOLD:
                        status = "POSSIBLE APPROACH"
                    else:
                        status = "NO STRONG GROWTH"

                risk_counts[status] += 1

                # Calculate the center of the vehicle bounding box.
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # Draw the center point.
                cv2.circle(
                    annotated,
                    (center_x, center_y),
                    4,
                    (255, 255, 255),
                    -1,
                )
                # Save the vehicle observation to CSV.
                confidence = float(box_confidence)

                csv_writer.writerow(
                   {
                        "frame": frame_count,
                        "track_id": track_id,
                        "vehicle_class": class_name,
                        "confidence": round(float(box_confidence), 4),
                        "bbox_area": round(area, 2),
                        "area_growth_ratio": (
                            round(growth_ratio, 4)
                            if growth_ratio is not None
                            else ""
                        ),
                        "indicator_status": status,
                    }
                )

                # Draw a short label above the vehicle's bounding box.
                label_y = max(55, int(y1) - 8)

                cv2.putText(
                    annotated,
                    "ADAS | Experimental Approach Indicator",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                # Add a dark panel so summary text remains readable.
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (width, 45),
                    (30, 30, 30),
                    -1,
                )

                cv2.putText(
                    annotated,
                    "ADAS | Experimental Approach Indicator",
                    (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

        writer.write(annotated)

        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames...")


finally:
    cap.release()
    writer.release()
    csv_file.close()


# -----------------------------
# Report
# -----------------------------
print("\n" + "=" * 50)
print("ADAS APPROACH INDICATOR REPORT")
print("=" * 50)

print(f"Frames processed: {frame_count}")
print(f"Unique vehicle tracking IDs: {len(vehicle_ids)}")

print("\nFrame-level indicator observations:")

for status, count in sorted(risk_counts.items()):
    print(f"{status:22s}: {count}")

print(f"\nOutput video: {OUTPUT_VIDEO.resolve()}")

print(
    "\nIMPORTANT: These are experimental image-based indicators, "
    "not validated collision predictions or calibrated TTC estimates."
)

print("Analysis complete.")