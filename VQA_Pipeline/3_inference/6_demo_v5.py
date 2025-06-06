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
from ultralytics import YOLO # Keep for class_id mapping, though not for inference
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


### this creates the eval (annotated frames) of a different video (not used for training)
### here Yolo is not used for detection, rather detections are directly taken from json file



# --- Configuration ---
# here yolo is not used for detection but detections are loaded from json
IMAGE_INPUT_DIR = "yolov12/896_anchor/detections/v5_detections/images"
ANNOTATED_OUTPUT_BASE_DIR = "yolov12/896_anchor/detections/v5_detections"
LOG_OUTPUT_PATH = os.path.join(ANNOTATED_OUTPUT_BASE_DIR, "v5_results.json")
SAVE_ANNOTATED_IMAGES = True  # Flag to control saving annotated images


# Path to JSON file containing precomputed detections
PRECOMPUTED_DETECTIONS_PATH = "yolov12/896_anchor/detections/v5_detections/v5_detections.json"

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
TRAINED_MODEL_PATH = "LLM/Meta-Llama-3.2-1B/Q6_context"
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# --- Suppress Warnings ---
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

# --- Load Precomputed Detections ---
print("Loading precomputed detections from JSON...")
try:
    with open(PRECOMPUTED_DETECTIONS_PATH, 'r') as f:
        precomputed_detections_data = json.load(f)
    print(f"Successfully loaded {len(precomputed_detections_data)} image entries from {PRECOMPUTED_DETECTIONS_PATH}")
except FileNotFoundError:
    print(f"Error: Precomputed detections file not found at {PRECOMPUTED_DETECTIONS_PATH}. Exiting.")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {PRECOMPUTED_DETECTIONS_PATH}. Please check file format. Exiting.")
    exit()

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
        # Smallest pile by area
        return min(piles, key=lambda p: p["area"])["name"]
    elif "leftmost pile" in task.lower():
        return min(piles, key=lambda p: p["position"][0])["name"]
    elif "largest pile" in task.lower():
        return max(piles, key=lambda p: p["area"])["name"]
    elif "fill the shovel" in task.lower():
        # Closest pile to anchor by distance
        return min(piles, key=lambda p: math.dist(anchor_pos, p["position"]))["name"]
    return None

# --- Create Base Output Directory ---
os.makedirs(ANNOTATED_OUTPUT_BASE_DIR, exist_ok=True)

# --- Main Processing Loop ---
# We iterate through the image files in the directory to load the images for annotation,
# but we will get detection data from the precomputed JSON.
image_files = sorted([f for f in os.listdir(IMAGE_INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
if not image_files:
    print(f"No image files found in: {IMAGE_INPUT_DIR}. Exiting.")
    exit()

results_log = []
for idx, image_name in enumerate(image_files, 1):
    img_path = os.path.join(IMAGE_INPUT_DIR, image_name)
    start_overall_time = time.time() # This will now capture all processing including image load and LLM

    original_image = cv2.imread(img_path)
    if original_image is None:
        print(f"Warning: Could not read image {image_name}. Skipping.")
        continue

    # --- Retrieve Detections from Precomputed JSON ---
    detected_piles = []
    detected_anchor = None
    all_detections_data = precomputed_detections_data.get(image_name, [])

    if not all_detections_data:
        print(f"Warning: No precomputed detections found for {image_name}. Skipping LLM tasks and annotation for this image.")
        # We still want to log this image in results_log even if no detections
        image_results = {
            "image": image_name,
            "piles": [],
            "anchor": None,
            "tasks": [],
            "computation_times": {
                "detection_pre_post_processing": 0,
                "llm_inference": 0,
                "llm_pre_post_processing": 0
            }
        }
        results_log.append(image_results)
        print(f"{idx}/{len(image_files)} frames processed")
        continue # Skip to the next image if no detections

    pile_count = 0 # To assign pile names correctly, starting fresh for each image
    for detection in all_detections_data:
        x1, y1, x2, y2 = detection['bbox']
        cls = detection['class_id']
        conf = detection['confidence']

        # We are using the hardcoded PILE_CLASS_ID and ANCHOR_CLASS_ID.
        # If your JSON uses different IDs or relies on 'class_name' string,
        # you might need to adjust this logic.
        if cls == PILE_CLASS_ID:
            centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
            area = calculate_bbox_area(x1, y1, x2, y2)
            pile_count += 1
            detected_piles.append({
                "name": f"pile{pile_count}",
                "position": [centroid_x, centroid_y],
                "area": area,
                "bbox": [x1, y1, x2, y2]
            })
        elif cls == ANCHOR_CLASS_ID:
            centroid_x, centroid_y = calculate_centroid(x1, y1, x2, y2)
            detected_anchor = {"position": [centroid_x, centroid_y], "bbox": [x1, y1, x2, y2]}

    # Construct LLM Context
    llm_context_string = ""
    llm_pre_post_time_llm_context = 0 # Time specifically for LLM context generation
    if detected_anchor and detected_piles:
        llm_context_gen_start_time = time.time()
        context_lines = [
            "You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use as a reference to determine distances and relative positions.",
            f"Anchor position: ({detected_anchor['position'][0]:.1f}, {detected_anchor['position'][1]:.1f})",
            "Following piles are present:"
        ]
        for pile in sorted(detected_piles, key=lambda p: int(p['name'][4:])): # Sort by pile number for consistent context
            context_lines.append(f"{pile['name']}: position=({pile['position'][0]:.1f}, {pile['position'][1]:.1f}), area={pile['area']:.1f}")
        llm_context_string = "\n".join(context_lines)
        llm_pre_post_time_llm_context = time.time() - llm_context_gen_start_time

    # Initialize computation times for this image
    image_results = {
        "image": image_name,
        "piles": detected_piles,
        "anchor": detected_anchor,
        "tasks": [],
        "computation_times": {
            "detection_pre_post_processing": 0, # This will include JSON read and parsing to internal structure
            "llm_inference": 0,  # Accumulated below
            "llm_pre_post_processing": llm_pre_post_time_llm_context # This now only includes LLM context preparation time
        }
    }

    # Process Each Task
    for current_llm_question in LLM_QUESTIONS:
        llm_task_start_time = time.time() # Time for processing this specific LLM question
        image_to_annotate = original_image.copy() if SAVE_ANNOTATED_IMAGES else None
        llm_predicted_target_pile_name = None
        llm_response = None
        llm_inf_time = 0

        if llm_context_string: # Only run LLM if context was successfully generated (i.e., detections exist)
            full_llm_prompt = f"<s>[INST] {llm_context_string}\nTask: {current_llm_question} [/INST]"
            llm_response, llm_inf_time = generate_llm_response(llm_model, tokenizer, full_llm_prompt)
            llm_predicted_target_pile_name = parse_llm_target_pile(llm_response)
            image_results["computation_times"]["llm_inference"] += llm_inf_time
            # Pre/post processing for LLM inference (e.g., tokenization) will be minimal here
            # as context generation is separate.
            # We'll just add tokenization/decoding time for this specific LLM call
            # which is already part of generate_llm_response if not explicitly separated.

        # Calculate Ground Truth
        ground_truth_pile = calculate_ground_truth(detected_piles, detected_anchor, current_llm_question)

        # Log Results for this task
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
                    # Find the corresponding detected_pile entry to get its assigned name (pile1, pile2, etc.)
                    current_pile_name = None
                    for p in detected_piles:
                        # Compare bounding box coordinates to link the detection to the assigned pile name
                        if p['bbox'] == [x1, y1, x2, y2]:
                            current_pile_name = p['name']
                            break

                    if current_pile_name and current_pile_name == llm_predicted_target_pile_name:
                        draw_color = LLM_TARGET_COLOR
                        label = f"LLM Target: {current_pile_name} {conf:.2f}"
                    else:
                        draw_color = OTHER_PILE_COLOR
                        label = f"{current_pile_name or 'Pile'} {conf:.2f}" # Fallback if name not found

                if draw_color:
                    cv2.rectangle(image_to_annotate, (x1, y1), (x2, y2), draw_color, FONT_THICKNESS + 1)
                    (text_width, text_height), baseline = cv2.getTextSize(label, FONT, FONT_SCALE * 0.8, 1)
                    label_y = y1 - 10 if y1 - text_height - 5 > text_overlay_y + 50 else y2 + text_height + 10
                    # Ensure label is not drawn on top of the task question, adjust y if needed
                    if label_y < text_overlay_y + 60: # If label too high, try below bbox
                         label_y = y2 + text_height + 10
                         if label_y + text_height > original_image.shape[0]: # If still out of bounds, place inside
                            label_y = y2 - 5

                    cv2.rectangle(image_to_annotate, (x1, label_y - text_height - 5), (x1 + text_width, label_y), TEXT_BG_COLOR, -1)
                    cv2.putText(image_to_annotate, label, (x1, label_y), FONT, FONT_SCALE * 0.8, TEXT_COLOR, 1, cv2.LINE_AA)

            # Save Annotated Image
            folder_name = current_llm_question.replace(" ", "_").replace("?", "").replace(".", "").replace("/", "_").replace(":", "").replace("__", "_").strip("_")
            annotated_output_prompt_dir = os.path.join(ANNOTATED_OUTPUT_BASE_DIR, folder_name)
            os.makedirs(annotated_output_prompt_dir, exist_ok=True)
            annotated_img_path = os.path.join(annotated_output_prompt_dir, image_name)
            cv2.imwrite(annotated_img_path, image_to_annotate)

    # Calculate overall detection pre/post processing time for this image
    # This now reflects the time taken to read and parse the JSON detections for this image.
    image_results["computation_times"]["detection_pre_post_processing"] = time.time() - start_overall_time - image_results["computation_times"]["llm_inference"] - image_results["computation_times"]["llm_pre_post_processing"]
    results_log.append(image_results)
    print(f"{idx}/{len(image_files)} frames processed")

# Save Results Log
with open(LOG_OUTPUT_PATH, 'w') as f:
    json.dump({"results": results_log}, f, indent=2)