from pathlib import Path

from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo26n.pt")

# Use a sample road-scene image
image_url = "https://ultralytics.com/images/bus.jpg"

# Run object detection
results = model.predict(
    source=image_url,
    conf=0.25,
    save=True,
    verbose=True,
)

# COCO categories relevant to our ADAS project
target_classes = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
}

counts = {name: 0 for name in sorted(target_classes)}

print("\n" + "=" * 45)
print("ADAS OBJECT DETECTION REPORT")
print("=" * 45)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        confidence = float(box.conf.item())

        if class_name in target_classes:
            counts[class_name] += 1

            print(
                f"Detected: {class_name:12s} "
                f"| Confidence: {confidence:.2%}"
            )

print("\nOBJECT COUNTS")

for class_name, count in counts.items():
    print(f"{class_name:12s}: {count}")

total_objects = sum(counts.values())

print(f"\nTotal relevant objects: {total_objects}")
print("Detection complete.")