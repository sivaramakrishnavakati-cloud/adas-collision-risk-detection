from pathlib import Path

import cv2
from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Open the sample video stream
video_url = "driving.mp4"

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

print("Starting ADAS object tracking...")
print("Loading video. Please wait...")

# Run YOLO with ByteTrack on consecutive video frames.
# Ultralytics handles video reading and saving.
results = model.track(
    source=video_url,
    tracker="bytetrack.yaml",
    conf=0.25,
    save=True,
    stream=True,
    verbose=False,
)

frame_count = 0
tracked_ids = set()
class_counts = {}

try:
    for result in results:
        frame_count += 1

        if result.boxes is None or len(result.boxes) == 0:
            continue

        boxes = result.boxes

        for index, box in enumerate(boxes):
            class_id = int(box.cls.item())
            class_name = result.names[class_id]
            confidence = float(box.conf.item())

            # Focus on objects relevant to driving
            relevant_classes = {
                "person",
                "bicycle",
                "car",
                "motorcycle",
                "bus",
                "truck",
            }

            if class_name not in relevant_classes:
                continue

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            track_id = None

            if boxes.id is not None:
                track_id = int(boxes.id[index].item())
                tracked_ids.add(track_id)

            print(
                f"Frame {frame_count:04d} | "
                f"Class: {class_name:12s} | "
                f"ID: {track_id} | "
                f"Confidence: {confidence:.2%}"
            )

finally:
    if hasattr(results, "close"):
        results.close()

print("\n" + "=" * 45)
print("ADAS VIDEO TRACKING REPORT")
print("=" * 45)

print(f"Frames processed: {frame_count}")
print(f"Unique tracking IDs observed: {len(tracked_ids)}")

print("\nObject detections across frames:")

for class_name, count in sorted(class_counts.items()):
    print(f"{class_name:12s}: {count}")

print("\nTracking run finished.")
print(
    "Check the Ultralytics runs/detect or runs/track "
    "folder for the annotated video."
)