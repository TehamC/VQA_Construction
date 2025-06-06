import cv2
import os
import json
import torch
import time
import math
import warnings
import re
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


### this will take an img do inference on yolo to detect objects, then using this info to do inference on LLM
### using the LLMs response images are annotated
### it will also create a json file containing various data such as LLM response, correct response, inference time etc
### takes very long due to double inference per img


# --- Configuration ---
YOLO_MODEL_PATH = "yolov12/896_anchor/runs/detect/train/weights/best.pt"
IMAGE_INPUT_DIR = "frames/hq_piles/aio_896"
ANNOTATED_OUTPUT_BASE_DIR = "LLM/evaluation_visual/6Q_context"
LOG_OUTPUT_PATH = os.path.join(ANNOTATED_OUTPUT_BASE_DIR, "6Q_results_corrected.json")
SAVE_ANNOTATED_IMAGES = True  # Flag to control saving annotated images

# Class IDs
PILE_CLASS_ID = 1
ANCHOR_CLASS_ID = 0

# Colors for drawing (B, G, R)
LLM_TARGET_COLOR = (0, 0, 255)
OTHER_PILE_COLOR = (0, 255, 0)
ANCHOR_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)
TEXT_BG_COLOR = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# LLM Questions
LLM_QUESTIONS = [
    "Start at the rightmost pile",
    "Clear a remote pile",
    "Clear a pile as fast as possible",
    "Start at the leftmost pile",
    "Process the largest pile",
    "Fill the shovel"
]

# --- Configuration for LLM ---
TRAINED_MODEL_PATH = "/home/teham/case_study/VQA/LLM/Meta-Llama-3.2-1B/Q6_context"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# --- Suppress Warnings, still doesnt work ---
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

@contextmanager
def suppress_output():
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# --- Load LLM Model and Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
with suppress_output():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
llm_model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
llm_model.eval()

# --- Load YOLO Model ---
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- Helper Functions ---
def generate_llm_response(model, tokenizer, prompt_text, max_new_tokens=64):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    inference_time = time.time() - start_time
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "[/INST]" in decoded_output:
        answer = decoded_output.split("[/INST]")[-1].strip()
    else:
        answer = decoded_output.strip()
    return answer.replace("</s>", "").strip(), inference_time

def parse_llm_target_pile(llm_response):
    match = re.search(r"(pile\d+)", llm_response, re.IGNORECASE)
    return match.group(1).lower() if match else None

def calculate_centroid(x1, y1, x2, y2):
    return (x1 + x2) / 2, (y1 + y2) / 2

def calculate_bbox_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def calculate_ground_truth(piles, anchor, task):
    if not piles or not anchor:
        return None
    anchor_pos = anchor["position"]
    if "rightmost pile" in task.lower():
        return max(piles, key=lambda p: p["position"][0])["name"]
    elif "remote pile" in task.lower():
        return max(piles, key=lambda p: math.dist(anchor_pos, p["position"]))["name"]
    elif "fast as possible" in task.lower():
        return min(piles, key=lambda p: p["area"])["name"]
    elif "leftmost pile" in task.lower():
        return min(piles, key=lambda p: p["position"][0])["name"]
    elif "largest pile" in task.lower():
        return max(piles, key=lambda p: p["area"])["name"]
    elif "fill the shovel" in task.lower():
        return min(piles, key=lambda p: math.dist(anchor_pos, p["position"]))["name"]
    return None

# --- Create Base Output Directory ---
os.makedirs(ANNOTATED_OUTPUT_BASE_DIR, exist_ok=True)

# --- Main Demo Loop ---
image_files = sorted([f for f in os.listdir(IMAGE_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not image_files:
    print(f"No image files found in: {IMAGE_INPUT_DIR}. Exiting.")
    exit()

results_log = []
for idx, image_name in enumerate(image_files, 1):
    img_path = os.path.join(IMAGE_INPUT_DIR, image_name)
    start_pre_time = time.time()
    original_image = cv2.imread(img_path)
    if original_image is None:
        continue

    # YOLO Detection
    yolo_pre_time = time.time()
    yolo_results = yolo_model(img_path, conf=0.5, verbose=False)[0]
    yolo_inference_time = time.time() - yolo_pre_time
    detected_piles = []
    detected_anchor = None
    all_detections_data = []

    for box in yolo_results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = yolo_model.names[cls]
        bbox_data = {"bbox": [x1, y1, x2, y2], "confidence": conf, "class_id": cls, "class_name": class_name}
        all_detections_data.append(bbox_data)
        if cls == PILE_CLASS_ID:
            centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
            area = calculate_bbox_area(x1, y1, x2, y2)
            detected_piles.append({
                "name": f"pile{len(detected_piles) + 1}",
                "position": [centroid_x, centroid_y],
                "area": area,
                "bbox": [x1, y1, x2, y2]
            })
        elif cls == ANCHOR_CLASS_ID:
            centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
            detected_anchor = {"position": [centroid_x, centroid_y], "bbox": [x1, y1, x2, y2]}

    # Construct LLM Context
    llm_context_string = ""
    llm_pre_post_time = 0
    if detected_anchor and detected_piles:
        llm_pre_time = time.time()
        context_lines = [
            "You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use as a reference to determine distances and relative positions.",
            f"Anchor position: ({detected_anchor['position'][0]:.1f}, {detected_anchor['position'][1]:.1f})",
            "Following piles are present:"
        ]
        for pile in sorted(detected_piles, key=lambda p: int(p['name'][4:])):
            context_lines.append(f"{pile['name']}: position=({pile['position'][0]:.1f}, {pile['position'][1]:.1f}), area={pile['area']:.1f}")
        llm_context_string = "\n".join(context_lines)
        llm_pre_post_time += time.time() - llm_pre_time

    # Process Each Task
    image_results = {
        "image": image_name,
        "piles": detected_piles,
        "anchor": detected_anchor,
        "tasks": [],
        "computation_times": {
            "yolo_inference": yolo_inference_time,
            "yolo_pre_post_processing": 0,  # Updated below
            "llm_inference": 0,  # Accumulated below
            "llm_pre_post_processing": llm_pre_post_time
        }
    }
    for current_llm_question in LLM_QUESTIONS:
        llm_pre_time = time.time()
        image_to_annotate = original_image.copy() if SAVE_ANNOTATED_IMAGES else None
        llm_predicted_target_pile_name = None
        llm_response = None
        llm_inf_time = 0

        if llm_context_string:
            full_llm_prompt = f"<s>[INST] {llm_context_string}\nTask: {current_llm_question} [/INST]"
            llm_response, llm_inf_time = generate_llm_response(llm_model, tokenizer, full_llm_prompt)
            llm_predicted_target_pile_name = parse_llm_target_pile(llm_response)
            image_results["computation_times"]["llm_inference"] += llm_inf_time
            image_results["computation_times"]["llm_pre_post_processing"] += time.time() - llm_pre_time - llm_inf_time

        # Calculate Ground Truth
        ground_truth_pile = calculate_ground_truth(detected_piles, detected_anchor, current_llm_question)

        # Log Results
        image_results["tasks"].append({
            "task": current_llm_question,
            "prediction": llm_predicted_target_pile_name,
            "calculated": ground_truth_pile,
            "llm_response": llm_response
        })

        # Visualize Detections (if enabled)
        if SAVE_ANNOTATED_IMAGES:
            text_overlay_y = 50
            cv2.putText(image_to_annotate, current_llm_question, (10, text_overlay_y),
                        FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
            if llm_response:
                cv2.putText(image_to_annotate, f"Predicted Answer: {llm_response}", (10, text_overlay_y + 30),
                            FONT, FONT_SCALE * 0.8, TEXT_COLOR, FONT_THICKNESS - 1, cv2.LINE_AA)

            for detection in all_detections_data:
                x1, y1, x2, y2 = detection['bbox']
                cls = detection['class_id']
                conf = detection['confidence']
                draw_color = None
                label = ""

                if cls == ANCHOR_CLASS_ID:
                    draw_color = ANCHOR_COLOR
                    label = f"Anchor {conf:.2f}"
                elif cls == PILE_CLASS_ID:
                    current_pile_name = next((p['name'] for p in detected_piles if p['bbox'] == [x1, y1, x2, y2]), None)
                    if current_pile_name and current_pile_name == llm_predicted_target_pile_name:
                        draw_color = LLM_TARGET_COLOR
                        label = f"LLM Target: {current_pile_name} {conf:.2f}"
                    else:
                        draw_color = OTHER_PILE_COLOR
                        label = f"{current_pile_name or 'Pile'} {conf:.2f}"

                if draw_color:
                    cv2.rectangle(image_to_annotate, (x1, y1), (x2, y2), draw_color, FONT_THICKNESS + 1)
                    (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE * 0.8, 1)
                    label_y = y1 - 10 if y1 - 10 > text_overlay_y + 50 else y2 + text_height + 10
                    cv2.rectangle(image_to_annotate, (x1, label_y - text_height - 5), (x1 + text_width, label_y), TEXT_BG_COLOR, -1)
                    cv2.putText(image_to_annotate, label, (x1, label_y), FONT, FONT_SCALE * 0.8, TEXT_COLOR, 1, cv2.LINE_AA)

            # Save Annotated Image
            folder_name = current_llm_question.replace(" ", "_").replace("?", "").replace(".", "").replace("/", "_").replace(":", "").replace("__", "_").strip("_")
            annotated_output_prompt_dir = os.path.join(ANNOTATED_OUTPUT_BASE_DIR, folder_name)
            os.makedirs(annotated_output_prompt_dir, exist_ok=True)
            annotated_img_path = os.path.join(annotated_output_prompt_dir, image_name)
            cv2.imwrite(annotated_img_path, image_to_annotate)

    image_results["computation_times"]["yolo_pre_post_processing"] = time.time() - start_pre_time - yolo_inference_time - image_results["computation_times"]["llm_inference"] - image_results["computation_times"]["llm_pre_post_processing"]
    results_log.append(image_results)
    print(f"{idx}/{len(image_files)} frames processed")

# Save Results Log
with open(LOG_OUTPUT_PATH, 'w') as f:
    json.dump({"results": results_log}, f, indent=2)