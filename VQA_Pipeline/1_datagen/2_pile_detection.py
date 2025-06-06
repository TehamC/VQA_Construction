from ultralytics import YOLO
import cv2
import os
import json


### this will detect piles on given images and create a json containing all detections (later used for VQA generation dataset)


# Paths and config
output_annotated_dir = "frames/vid5/inference_old"
output_json          = output_annotated_dir + "/inference_detections.json" 
image_folder         = "frames/hq_piles/aio_896"
model_path           = "yolov12/896_anchor/runs/detect/train/weights/best.pt"

# --- Configuration for Classes ---

PILE_CLASS_ID = 1  
ANCHOR_CLASS_ID = 0 

# Colors for drawing
PILE_COLOR = (0, 255, 0)  # Green for piles (B, G, R)
ANCHOR_COLOR = (255, 0, 0) # Blue for anchor (B, G, R)
TEXT_COLOR = (0, 0, 0)   # Black for text (on light background)
TEXT_BG_COLOR = (255, 255, 255) # White background for text for clarity

# Create output folder
os.makedirs(output_annotated_dir, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Print class names from the model to verify their IDs
print("Model Class Names and their IDs:")
for class_id, class_name in model.names.items():
    print(f"  ID: {class_id}, Name: {class_name}")



# Run detection on all frames
all_frames_detections = {} 

for image_name in os.listdir(image_folder):
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(image_folder, image_name)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Warning: Could not read image {img_path}. Skipping.")
        continue

    # Run model

    results = model(img_path, conf=0.6, verbose=False)[0] # verbose=False to reduce console output

    # Prepare a list to store detections for the current image
    current_image_detections = []
    
    # Store anchor specific data if detected for later use in evaluation visuals
    anchor_detection = None 

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = model.names[cls] # Get the class name from the model

        detection_data = {
            "bbox": [x1, y1, x2, y2],
            "confidence": conf,
            "class_id": cls,
            "class_name": class_name # Store the class name as well
        }
        current_image_detections.append(detection_data)

        # Determine color and label based on class ID
        draw_color = None
        label = ""

        if cls == PILE_CLASS_ID:
            draw_color = PILE_COLOR
            label = f"Pile {conf:.2f}"
        elif cls == ANCHOR_CLASS_ID:
            draw_color = ANCHOR_COLOR
            label = f"Anchor {conf:.2f}"
            anchor_detection = detection_data # Store the anchor detection for this frame
        else:
            # Handle other classes if they exist and you want to visualize them differently
            draw_color = (128, 128, 128) # Grey for unknown classes
            label = f"{class_name} {conf:.2f}"

        # Draw box on image if a known class
        if draw_color:
            cv2.rectangle(image, (x1, y1), (x2, y2), draw_color, 2)
            
            # Put text with background for better visibility
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), TEXT_BG_COLOR, -1)
            cv2.putText(image, label, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)

    all_frames_detections[image_name] = current_image_detections # Store all detections for this image

    # Save annotated image
    annotated_path = os.path.join(output_annotated_dir, image_name)
    cv2.imwrite(annotated_path, image)

# Save all detection data to a single JSON file
os.makedirs(os.path.dirname(output_json), exist_ok=True) # Ensure directory exists
with open(output_json, 'w') as f:
    json.dump(all_frames_detections, f, indent=2)

print(f"\n[✓] Saved detections to {output_json}")
print(f"[✓] Annotated frames saved to {output_annotated_dir}")