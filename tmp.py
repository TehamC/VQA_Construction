import cv2
import json
from ultralytics import YOLO
from tqdm import tqdm

# === CONFIG ===
video_path = '/home/teham/case_study/VQA/testv1_cut.mp4'
output_json = '/home/teham/case_study/VQA/testv1_detections.json'
yolo_model_path = '/home/teham/case_study/VQA/yolov12/896_anchor/runs/detect/train/weights/best.pt'
frame_stride = 1  # Process every Nth frame

# === Load model ===
model = YOLO(yolo_model_path)
model.fuse()
model.to("cuda")

# === Open video ===
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# === Storage ===
detections_per_frame = []
frame_idx = 0

print(f"[INFO] Processing video: {video_path}")

with tqdm(total=total_frames, desc="Processing frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            results = model(frame, verbose=False)[0]

            frame_data = {
                'frame_index': frame_idx,
                'detections': []
            }

            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()

                frame_data['detections'].append({
                    'class_id': cls_id,
                    'class_name': class_name,
                    'confidence': conf,
                    'bbox': xyxy
                })

            detections_per_frame.append(frame_data)

        frame_idx += 1
        pbar.update(1)

cap.release()

# === Write JSON ===
with open(output_json, 'w') as f:
    json.dump(detections_per_frame, f, indent=2)

print(f"[DONE] Saved {len(detections_per_frame)} annotated frames to {output_json}")
