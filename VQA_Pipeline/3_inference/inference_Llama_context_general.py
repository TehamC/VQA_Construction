import os
import torch
import random
import json
import math
import warnings
import re # Import regex module
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import logging
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import io

### this is a inference code i used during training of my LLM models
### this was not explicitly mentioned in the documentation, basically what it does it makes inference of trained models easy
### it will read the jsonl and randomize sample examples and verify the answer by using calculations

# --- Configuration ---
# Path to trained model 
TRAINED_MODEL_PATH = "LLM/Meta-Llama-3.2-1B/Q6_context"
# Path to your original training data JSONL file
DATA_FILE_PATH = "yolov12/896_anchor/detections/yolov12n/vqa_context_6Q.jsonl" 
# Base model name 
BASE_MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# ---  Configuration for Context Grouping and Randomization ---
N_PROMPTS_PER_CONTEXT = 6 # 6 prompts for the first context
NUM_RANDOM_CONTEXT_GROUPS = 3 # How many unique context groups to test

# --- Randomization Ranges (Adjust if needed) ---
COORD_MIN, COORD_MAX = 0.0, 1000.0
AREA_MIN, AREA_MAX = 1000.0, 60000.0 # Keeping area positive and within a reasonable range

# --- Suppress Warnings and Output ---
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

@contextmanager
def suppress_output():
    """
    A context manager to suppress stdout and stderr.
    Useful for suppressing verbose loading messages.
    """
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            yield

# --- Load Model and Tokenizer ---
print(f"Loading tokenizer from: {TRAINED_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Llama models prefer right padding for generation
tokenizer.model_max_length = 2048 # Ensure this matches training max_seq_length

print(f"Loading base model: {BASE_MODEL_NAME} with 4-bit quantization...")
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

with suppress_output():
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16, # Ensure consistent dtype
    )

print(f"Loading LoRA adapters from: {TRAINED_MODEL_PATH}")
model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
model.eval() # Set model to evaluation mode for inference

print("Model loaded successfully and set to evaluation mode.")

# --- Helper Function for Inference ---
def generate_response(model, tokenizer, prompt_text, max_new_tokens=64):
    """
    Generates a response from the model for a given prompt.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False, # Use greedy decoding for consistent results
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part after [/INST]
    if "[/INST]" in decoded_output:
        answer = decoded_output.split("[/INST]")[-1].strip()
    else:
        answer = decoded_output.strip()

    # Cleanup trailing tokens like </s>
    answer = answer.replace("</s>", "").strip()
    return answer

# --- Helper Functions for Context Parsing and Randomization ---

def parse_context_string(context_str):
    """Parses context string to extract anchor and pile data."""
    data = {"anchor": None, "piles": []}

    # Extract anchor position
    anchor_match = re.search(r"Anchor position: \((\d+\.\d+), (\d+\.\d+)\)", context_str)
    if anchor_match:
        data["anchor"] = (float(anchor_match.group(1)), float(anchor_match.group(2)))

    # Extract pile data
    pile_matches = re.finditer(r"pile(\d+): position=\((\d+\.\d+), (\d+\.\d+)\), area=(\d+\.\d+)", context_str)
    for match in pile_matches:
        data["piles"].append({
            "name": f"pile{match.group(1)}",
            "position": (float(match.group(2)), float(match.group(3))),
            "area": float(match.group(4))
        })
    return data

def reconstruct_context_string(parsed_data, base_text_start="You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use as a reference to determine distances and relative positions."):
    """Reconstructs context string from parsed data."""
    context_lines = [base_text_start]
    context_lines.append(f"Anchor position: ({parsed_data['anchor'][0]:.1f}, {parsed_data['anchor'][1]:.1f})")
    context_lines.append("Following piles are present:")
    for pile in parsed_data["piles"]:
        context_lines.append(f"pile{pile['name'][4:]}: position=({pile['position'][0]:.1f}, {pile['position'][1]:.1f}), area={pile['area']:.1f}")
    return "\n".join(context_lines)

def randomize_geometric_data(parsed_data):
    """Randomizes anchor and pile positions/areas."""
    randomized_data = {
        "anchor": (random.uniform(COORD_MIN, COORD_MAX), random.uniform(COORD_MIN, COORD_MAX)),
        "piles": []
    }
    for pile in parsed_data["piles"]: # Maintain the number of piles
        randomized_data["piles"].append({
            "name": pile["name"], # Keep original pile names (e.g., pile1, pile2)
            "position": (random.uniform(COORD_MIN, COORD_MAX), random.uniform(COORD_MIN, COORD_MAX)),
            "area": random.uniform(AREA_MIN, AREA_MAX)
        })
    return randomized_data


# calculate ground truth

def calculate_true_answer(parsed_data, task_str):
    """Calculates the true answer for a given task and parsed geometric data."""
    anchor_pos = parsed_data["anchor"]
    piles = parsed_data["piles"]

    target_pile = None
    descriptor = ""

    if "nearest pile" in task_str or "Fill the shovel" in task_str:
        target_pile = min(piles, key=lambda p: calculate_distance(anchor_pos, p["position"]))
        descriptor = "the nearest"
    elif "farthest pile" in task_str or "Clear a remote pile" in task_str:
        target_pile = max(piles, key=lambda p: calculate_distance(anchor_pos, p["position"]))
        descriptor = "the farthest"
    elif "biggest pile" in task_str or "Process the largest pile" in task_str:
        target_pile = max(piles, key=lambda p: p["area"])
        descriptor = "the largest"
    elif "smallest pile" in task_str or "Clear a pile as fast as possible" in task_str:
        target_pile = min(piles, key=lambda p: p["area"])
        descriptor = "the smallest"
    elif "rightmost pile" in task_str or "Start at the rightmost pile" in task_str:
        target_pile = max(piles, key=lambda p: p["position"][0]) # Max X-coordinate
        descriptor = "the rightmost"
    elif "leftmost pile" in task_str or "Start at the leftmost pile" in task_str:
        target_pile = min(piles, key=lambda p: p["position"][0]) # Min X-coordinate
        descriptor = "the leftmost"
    elif "topmost pile" in task_str or "Start at the topmost pile" in task_str:
        target_pile = min(piles, key=lambda p: p["position"][1]) # Min Y-coordinate (assuming y-axis increases downwards in image coords)
        descriptor = "the topmost"
    elif "bottommost pile" in task_str or "Start at the bottommost pile" in task_str:
        target_pile = max(piles, key=lambda p: p["position"][1]) # Max Y-coordinate
        descriptor = "the bottommost"
    # Add more task mappings if you have other types of tasks

    if target_pile:
        return (f"drive to {descriptor} pile: {target_pile['name']}"
                f"[{target_pile['position'][0]:.1f}, {target_pile['position'][1]:.1f}] and initiate digging")
    return "Error: Could not determine true answer for this task." # Fallback

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# --- Data Loading and Context Grouping ---
print(f"\nLoading and grouping data from: {DATA_FILE_PATH}")
all_data = []
try:
    with open(DATA_FILE_PATH, 'r') as f:
        for line in f:
            all_data.append(json.loads(line))
except Exception as e:
    print(f"Error loading data from JSONL file: {e}")
    exit()

context_group_indices = []
i = 0
while i < len(all_data):
    if i + N_PROMPTS_PER_CONTEXT <= len(all_data):
        current_context = all_data[i]["context"]
        is_valid_group = True
        for j in range(1, N_PROMPTS_PER_CONTEXT):
            if all_data[i + j]["context"] != current_context:
                is_valid_group = False
                break
        if is_valid_group:
            context_group_indices.append(i)
            i += N_PROMPTS_PER_CONTEXT # Move to the next potential group
        else:
            i += 1 # Move to the next line if this isn't a valid start of a group
    else:
        break # Not enough lines left for a full group

if not context_group_indices:
    print(f"Error: No complete context groups of size {N_PROMPTS_PER_CONTEXT} found in data.")
    exit()

print(f"Found {len(context_group_indices)} valid context groups.")

# --- Select Random Context Groups for Inference ---
selected_group_start_indices = random.sample(context_group_indices, min(NUM_RANDOM_CONTEXT_GROUPS, len(context_group_indices)))
print(f"Selected {len(selected_group_start_indices)} random context groups for testing.")


# --- Part 1: Inference on Original Data Context Groups ---
print("\n" + "="*80)
print("--- Performing Inference on Selected Original Data Context Groups ---")
print("="*80)

for group_num, start_idx in enumerate(selected_group_start_indices):
    print(f"\n--- Context Group {group_num + 1} (Original Data) ---")
    
    # Store all samples for this context group
    context_group_samples = [all_data[start_idx + j] for j in range(N_PROMPTS_PER_CONTEXT)]
    
    # Print the shared context for clarity
    print(f"Shared Context:\n{context_group_samples[0]['context']}")

    for i, sample in enumerate(context_group_samples):
        prompt = sample["text"]
        true_answer = sample["answer"]

        print(f"\n--- Prompt {i+1} (Original Data) ---")
        print(f"Task: {sample['task']}") # Print the specific task
        
        predicted_answer = generate_response(model, tokenizer, prompt)

        print(f"True Answer: {true_answer}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct: {true_answer.strip().lower() == predicted_answer.strip().lower()}")

# --- Part 2: Inference on Randomized Data Context Groups ---
print("\n" + "="*80)
print("--- Performing Inference on Selected Randomized Data Context Groups ---")
print("="*80)

for group_num, start_idx in enumerate(selected_group_start_indices):
    print(f"\n--- Context Group {group_num + 1} (Randomized Data) ---")

    # Get the original context and tasks for this group
    original_context_samples = [all_data[start_idx + j] for j in range(N_PROMPTS_PER_CONTEXT)]
    original_context_str = original_context_samples[0]["context"]
    original_tasks = [sample["task"] for sample in original_context_samples]

    # Parse and randomize the geometric data
    parsed_original_data = parse_context_string(original_context_str)
    randomized_data = randomize_geometric_data(parsed_original_data)
    randomized_context_str = reconstruct_context_string(randomized_data)
    
    # Print the shared randomized context for clarity
    print(f"Shared Randomized Context:\n{randomized_context_str}")

    for i, task in enumerate(original_tasks):
        # Construct the full prompt with the randomized context and original task
        full_randomized_prompt = f"<s>[INST] {randomized_context_str}\nTask: {task} [/INST]"
        
        # Calculate the true answer for this randomized scenario
        script_calculated_true_answer = calculate_true_answer(randomized_data, task)

        print(f"\n--- Prompt {i+1} (Randomized Data) ---")
        print(f"Task: {task}") # Print the specific task
        
        predicted_answer = generate_response(model, tokenizer, full_randomized_prompt)

        print(f"True Answer (Script-calculated): {script_calculated_true_answer}")
        print(f"Predicted Answer: {predicted_answer}")
        print(f"Correct: {script_calculated_true_answer.strip().lower() == predicted_answer.strip().lower()}")

print("\nInference script finished.")