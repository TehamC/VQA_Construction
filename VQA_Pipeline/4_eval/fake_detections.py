import os
import json
import random
from pathlib import Path


### video 5 (different construction site than training) had completely different colors as it was recorded on a different season
### and the sun was very bright, completely different than what I trained my Yolo model for
### Including this data into the old dataset and retraining the model is easy but time consuming
### Yolo being able to detect piles is completely clear so I focused testing the LLM here rather than retraining
### this code basically creates a "fake detection" json from the labeled data which is used for the LLM inference
### The LLm doesnt care if these are eral infereced detections or not, only matters are the features





# === CONFIGURATION ===
IMAGE_FOLDER = Path("yolov12/896_anchor/detections/v5_detections/images")  # adjusts if needed
LABEL_FOLDER = Path("yolov12/896_anchor/detections/v5_detections/labels")
OUTPUT_JSON = Path("yolov12/896_anchor/detections/v5_detections/v5_detections.json")

IMAGE_SIZE = 896  # assuming square images 896×896
CONFIDENCE_MIN = 0.65
CONFIDENCE_MAX = 0.85

CLASS_NAMES = {
    0: "anchor",
    1: "pile of gravel"
}

def yolo_to_bbox(cx, cy, w, h, img_size):
    """
    Convert normalized YOLO format (cx, cy, w, h) to absolute pixel bbox [x1, y1, x2, y2].
    Coordinates are clamped to [0, img_size].
    """
    abs_cx = cx * img_size
    abs_cy = cy * img_size
    abs_w = w * img_size
    abs_h = h * img_size

    x1 = abs_cx - abs_w / 2
    y1 = abs_cy - abs_h / 2
    x2 = abs_cx + abs_w / 2
    y2 = abs_cy + abs_h / 2

    # Clamp to image bounds
    x1 = max(0, min(img_size, x1))
    y1 = max(0, min(img_size, y1))
    x2 = max(0, min(img_size, x2))
    y2 = max(0, min(img_size, y2))

    # Convert to integers
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]

def generate_detections(label_folder, image_folder, img_size):
    """
    Read all YOLO label files in label_folder, convert to detection dicts,
    and return a mapping {image_filename: [ {bbox, confidence, class_id, class_name}, ... ] }.
    """
    detections = {}

    # Iterate over each .txt file in the label folder
    for label_path in label_folder.glob("*.txt"):
        stem = label_path.stem  # e.g. "vid1_frame_12660"
        image_filename = f"{stem}.jpg"
        image_path = image_folder / image_filename

        # If the corresponding image doesn't exist, skip
        if not image_path.is_file():
            continue

        # Prepare a list to hold all detections for this image
        image_detections = []

        # Read the YOLO-format label file
        with label_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    # Unexpected format; skip this line
                    continue

                class_id = int(parts[0])
                cx_norm = float(parts[1])
                cy_norm = float(parts[2])
                w_norm = float(parts[3])
                h_norm = float(parts[4])

                # Convert to pixel bbox
                bbox = yolo_to_bbox(cx_norm, cy_norm, w_norm, h_norm, img_size)

                # Assign a random confidence between CONFIDENCE_MIN and CONFIDENCE_MAX
                confidence = round(random.uniform(CONFIDENCE_MIN, CONFIDENCE_MAX), 4)

                class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                det_dict = {
                    "bbox": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                }
                image_detections.append(det_dict)

        # Only add to dictionary if there is at least one detection
        if image_detections:
            detections[image_filename] = image_detections

    return detections

def main():
    detections = generate_detections(LABEL_FOLDER, IMAGE_FOLDER, IMAGE_SIZE)

    # Write out the detections to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w") as out_f:
        json.dump(detections, out_f, indent=2)

    print(f"✅ Saved detections for {len(detections)} images to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
