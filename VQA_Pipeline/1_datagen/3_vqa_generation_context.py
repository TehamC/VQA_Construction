import random
import json
from pathlib import Path
from statistics import mean, stdev
import math



### there the VQA-pairs will be generated to be used for LLM training


# --- Configuration ---
IMAGE_SIZE = 896
CONTENT_HEIGHT = 672
PADDING_TOP = (IMAGE_SIZE - CONTENT_HEIGHT) // 2
MIN_BBOX_WIDTH = 50
MAX_BBOX_WIDTH = 200
MIN_BBOX_HEIGHT = 30
MAX_BBOX_HEIGHT = 150
MIN_AREA_THRESHOLD = 1000


# bbox calculations
def bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# statistics for augmentation
def extract_statistics(detections_json_path):

    detections = json.load(open(detections_json_path))
    pile_areas = []
    pile_centers = []
    
    for image_data in detections.values():
        for obj in image_data:
            if obj["class_name"] == "pile of gravel":
                x1, y1, x2, y2 = obj["bbox"]
                area = (x2 - x1) * (y2 - y1)
                cx, cy = bbox_center(obj["bbox"])
                pile_areas.append(area)
                pile_centers.append((cx, cy))
    
    if not pile_areas or not pile_centers:
        print("Warning: No pile detections found for statistics. Using default values.")
        return {
            "area_mean": 5000.0, "area_std": 2000.0,
            "cx_mean": IMAGE_SIZE / 2, "cx_std": IMAGE_SIZE / 4,
            "cy_mean": (PADDING_TOP + CONTENT_HEIGHT / 2), "cy_std": CONTENT_HEIGHT / 4,
        }

    area_std_val = stdev(pile_areas) if len(pile_areas) > 1 else 0
    cx_std_val = stdev(p[0] for p in pile_centers) if len(pile_centers) > 1 else 0
    cy_std_val = stdev(p[1] for p in pile_centers) if len(pile_centers) > 1 else 0

    return {
        "area_mean": mean(pile_areas),
        "area_std": area_std_val,
        "cx_mean": mean(p[0] for p in pile_centers),
        "cx_std": cx_std_val,
        "cy_mean": mean(p[1] for p in pile_centers),
        "cy_std": cy_std_val,
    }


def create_bbox_within_bounds(x1, y1, width, height):

    x2 = x1 + width
    y2 = y1 + height

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(IMAGE_SIZE, x2)
    y2 = min(IMAGE_SIZE, y2)

    y1 = max(PADDING_TOP, y1)
    y2 = min(IMAGE_SIZE - PADDING_TOP, y2)
    
    if x2 - x1 < MIN_BBOX_WIDTH: x2 = x1 + MIN_BBOX_WIDTH
    if y2 - y1 < MIN_BBOX_HEIGHT: y2 = y1 + MIN_BBOX_HEIGHT

    x2 = min(IMAGE_SIZE, x2)
    y2 = min(IMAGE_SIZE - PADDING_TOP, y2)
    
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def sample_bbox_from_stats(stats):

    while True:
        area = max(MIN_AREA_THRESHOLD, int(random.gauss(stats["area_mean"], stats["area_std"])))
        width = max(MIN_BBOX_WIDTH, int(random.uniform(0.8, 1.2) * (area ** 0.5)))
        height = max(MIN_BBOX_HEIGHT, int(area / width))
        
        cx = int(random.gauss(stats["cx_mean"], stats["cx_std"]))
        cy = int(random.gauss(stats["cy_mean"], stats["cy_std"])) 
        
        x1 = cx - width // 2
        y1 = cy - height // 2

        bbox = create_bbox_within_bounds(x1, y1, width, height)
        if bbox:
            return bbox

def random_bbox():
    # ... (same as before)
    while True:
        width = random.randint(MIN_BBOX_WIDTH, MAX_BBOX_WIDTH)
        height = random.randint(MIN_BBOX_HEIGHT, MAX_BBOX_HEIGHT)
        
        x1 = random.randint(0, IMAGE_SIZE - width)
        y1 = random.randint(PADDING_TOP, IMAGE_SIZE - height - PADDING_TOP)
        
        bbox = create_bbox_within_bounds(x1, y1, width, height)
        if bbox:
            return bbox

def generate_piles_data(bbox_generator, num_piles_min=3, num_piles_max=6):
    """
    Generates a list of pile data (bbox, center, area) using a given bbox generator function.
    Ensures unique centers and at least 2 valid piles.
    This function is for contexts *with* piles.
    """
    piles_bboxes = [bbox_generator() for _ in range(random.randint(num_piles_min, num_piles_max))]
    
    current_piles_data = []
    seen_centers = set() 
    for bbox in piles_bboxes:
        center = bbox_center(bbox)
        rounded_center = (round(center[0], 1), round(center[1], 1)) 
        if rounded_center not in seen_centers:
            current_piles_data.append({
                "id": f"pile{len(current_piles_data) + 1}",
                "bbox": bbox,
                "center": center,
                "area": bbox_area(bbox)
            })
            seen_centers.add(rounded_center)
    
    # Ensure at least 2 unique piles
    while len(current_piles_data) < 2: 
        new_bbox = bbox_generator()
        new_center = bbox_center(new_bbox)
        rounded_new_center = (round(new_center[0], 1), round(new_center[1], 1))
        if rounded_new_center not in seen_centers:
            current_piles_data.append({
                "id": f"pile{len(current_piles_data) + 1}",
                "bbox": new_bbox,
                "center": new_center,
                "area": bbox_area(new_bbox)
            })
            seen_centers.add(rounded_new_center)
            
    return current_piles_data

def generate_prompt_and_answer(anchor_bbox, piles_data, task_type):
    """
    Generates a single prompt-answer pair for a given context and task type.
    Handles 'no piles detected' scenarios.
    """
    pile_descriptions = []
    if piles_data: # Only generate descriptions if there are piles
        for pile in piles_data:
            pile_descriptions.append(
                f"{pile['id']}: position=({pile['center'][0]:.1f}, {pile['center'][1]:.1f}), area={pile['area']:.1f}"
            )
        piles_section = "Following piles are present:\n" + "\n".join(pile_descriptions)
    else: # No piles detected
        piles_section = "No piles of gravel are currently detected." 

    context = (
        f"You are a caterpillar in a construction site. In the following you will be given "
        f"geometric data of piles of gravel such as their position and size. "
        f"Alongside the piles you will be given an anchor (your position), which you can use as a reference "
        f"to determine distances and relative positions.\n"
        f"Anchor position: ({bbox_center(anchor_bbox)[0]:.1f}, {bbox_center(anchor_bbox)[1]:.1f})\n"
        f"{piles_section}" 
    )

    task_prompt = ""
    answer_text = ""

    if not piles_data: # If no piles, the answer is always fixed
        task_prompt = random.choice(["Fill the shovel", "Clear a remote pile", "Process the largest pile",
                                     "Clear a pile as fast as possible", "Start at the rightmost pile", "Start at the leftmost pile"])
        answer_text = "No piles detected"
    elif task_type == "nearest":
        target_pile_info = min(piles_data, key=lambda p: euclidean_distance(bbox_center(anchor_bbox), p["center"]))
        task_prompt = "Fill the shovel"
        answer_text = f"drive to the nearest pile: {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    elif task_type == "farthest":
        target_pile_info = max(piles_data, key=lambda p: euclidean_distance(bbox_center(anchor_bbox), p["center"]))
        task_prompt = "Clear a remote pile"
        answer_text = f"drive to the farthest pile: {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    elif task_type == "largest":
        target_pile_info = max(piles_data, key=lambda p: p["area"])
        task_prompt = "Process the largest pile"
        answer_text = f"drive to the largest pile: {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    elif task_type == "smallest":
        target_pile_info = min(piles_data, key=lambda p: p["area"])
        task_prompt = "Clear a pile as fast as possible"
        answer_text = f"drive to the smallest pile: {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    elif task_type == "rightmost":
        target_pile_info = max(piles_data, key=lambda p: p["center"][0])
        task_prompt = "Start at the rightmost pile"
        answer_text = f"drive to {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    elif task_type == "leftmost":
        target_pile_info = min(piles_data, key=lambda p: p["center"][0])
        task_prompt = "Start at the leftmost pile"
        answer_text = f"drive to {target_pile_info['id']}[{target_pile_info['center'][0]:.1f}, {target_pile_info['center'][1]:.1f}] and initiate digging"
    else:
        raise ValueError(f"Invalid task type: {task_type}")

    return {
        "context": context,
        "task": task_prompt,
        "answer": answer_text,
        "text": f"<s>[INST] {context}\nTask: {task_prompt} [/INST] {answer_text}</s>"
    }

def process_and_validate_real_context(image_data):
    """
    Extracts anchor and piles from real image data, calculates properties, and validates.
    This function will still only return contexts with 2+ unique piles,
    as real "no pile" cases need special handling.
    """
    anchor = None
    piles_raw = []
    for obj in image_data:
        if obj["class_name"] == "anchor":
            anchor = obj["bbox"]
        elif obj["class_name"] == "pile of gravel":
            piles_raw.append(obj["bbox"])
    
    if anchor and len(piles_raw) >= 2:
        current_piles_data = []
        seen_centers = set()
        for bbox in piles_raw:
            center = bbox_center(bbox)
            rounded_center = (round(center[0], 1), round(center[1], 1))
            if rounded_center not in seen_centers:
                current_piles_data.append({
                    "id": f"pile{len(current_piles_data) + 1}",
                    "bbox": bbox,
                    "center": center,
                    "area": bbox_area(bbox)
                })
                seen_centers.add(rounded_center)
        
        if len(current_piles_data) < 2: 
            return None 
        
        return {"anchor": anchor, "piles": current_piles_data}
    return None # Invalid context (missing anchor or not enough piles)


def generate_dataset(detection_json_path, output_file, total_contexts=30000, target_hybrid_ratio=0.3, no_piles_ratio=0.10):
    """
    Generates a dataset with a specific mix of context types, including 'no piles detected' scenarios.
    
    Args:
        detection_json_path (str): Path to the real detections JSON file.
        output_file (str): Path where the generated JSONL dataset will be saved.
        total_contexts (int): The total number of unique contexts (scenes) to generate.
        target_hybrid_ratio (float): Proportion of (real + stat-fitted) contexts from the *piles-present* contexts.
        no_piles_ratio (float): Proportion of contexts that will have an anchor but zero piles.
    """
    output = Path(output_file)
    output.parent.mkdir(parents=True, exist_ok=True)

    all_raw_detections = json.load(open(detection_json_path))
    valid_real_contexts = []
    for image_key in all_raw_detections:
        context = process_and_validate_real_context(all_raw_detections[image_key])
        if context:
            valid_real_contexts.append(context)
    
    num_real_contexts_available = len(valid_real_contexts)
    print(f"Found {num_real_contexts_available} valid real detection contexts (with 2+ piles).")

    stats = extract_statistics(detection_json_path)

    all_tasks = ["nearest", "farthest", "largest", "smallest", "rightmost", "leftmost"]
    num_task_types = len(all_tasks)

    # --- Calculate context counts with "no piles" consideration ---
    num_no_piles_contexts = int(total_contexts * no_piles_ratio)
    num_piles_present_contexts = total_contexts - num_no_piles_contexts
    
    # Distribute the 'piles-present' contexts based on target_hybrid_ratio
    num_hybrid_contexts_target_piles_present = int(num_piles_present_contexts * target_hybrid_ratio)
    
    num_real_contexts_to_use = min(num_real_contexts_available, num_hybrid_contexts_target_piles_present)
    
    num_stat_contexts_to_generate = max(0, num_hybrid_contexts_target_piles_present - num_real_contexts_to_use)
    
    num_random_contexts_to_generate = num_piles_present_contexts - num_real_contexts_to_use - num_stat_contexts_to_generate
    
    # Adjust for potential rounding issues (ensure non-negative counts)
    if num_random_contexts_to_generate < 0:
        num_random_contexts_to_generate = 0
        num_stat_contexts_to_generate = num_piles_present_contexts - num_real_contexts_to_use 
        if num_stat_contexts_to_generate < 0: num_stat_contexts_to_generate = 0
        print("Warning: Context ratios might be off due to rounding or limited real data. Adjusting counts.")


    final_total_contexts_generated = (
        num_real_contexts_to_use + 
        num_stat_contexts_to_generate + 
        num_random_contexts_to_generate +
        num_no_piles_contexts # Add the no-piles contexts back to the total
    )
    final_total_qa_pairs_generated = final_total_contexts_generated * num_task_types

    print(f"\n--- Data Generation Plan ---")
    print(f"Target total contexts: {total_contexts}")
    print(f"Actual contexts breakdown:")
    print(f"  Real Detections contexts (2+ piles): {num_real_contexts_to_use}")
    print(f"  Stat-fitted Synthetic contexts (2+ piles): {num_stat_contexts_to_generate}")
    print(f"  Purely Random Synthetic contexts (2+ piles): {num_random_contexts_to_generate}")
    print(f"  No Piles Detected contexts (0 piles): {num_no_piles_contexts}")
    print(f"Total contexts generated: {final_total_contexts_generated}")
    print(f"Total QA pairs to be generated: {final_total_qa_pairs_generated}")
    print(f"--------------------------")

    with output.open("w") as f:
        # --- 1. Generate from Real Data (2+ piles) ---
        random.shuffle(valid_real_contexts)
        for i in range(num_real_contexts_to_use):
            context_data = valid_real_contexts[i]
            random.shuffle(all_tasks) 
            for task_type in all_tasks:
                example = generate_prompt_and_answer(context_data["anchor"], context_data["piles"], task_type)
                f.write(json.dumps(example) + "\n")

        # --- 2. Generate Statistically-fitted Synthetic Data (2+ piles) ---
        for _ in range(num_stat_contexts_to_generate):
            anchor = sample_bbox_from_stats(stats)
            piles_data = generate_piles_data(lambda: sample_bbox_from_stats(stats))
            
            random.shuffle(all_tasks)
            for task_type in all_tasks:
                example = generate_prompt_and_answer(anchor, piles_data, task_type)
                f.write(json.dumps(example) + "\n")

        # --- 3. Generate Purely Random Synthetic Data (2+ piles) ---
        for _ in range(num_random_contexts_to_generate):
            anchor = random_bbox()
            piles_data = generate_piles_data(random_bbox)
            
            random.shuffle(all_tasks)
            for task_type in all_tasks:
                example = generate_prompt_and_answer(anchor, piles_data, task_type)
                f.write(json.dumps(example) + "\n")

        # --- 4. Generate "No Piles Detected" Contexts ---
        for _ in range(num_no_piles_contexts):
            anchor = random_bbox() # Always generate an anchor
            piles_data = [] # No piles
            
            random.shuffle(all_tasks) # Still generate all task types for consistency
            for task_type in all_tasks:
                example = generate_prompt_and_answer(anchor, piles_data, task_type) # This will detect no piles and give fixed answer
                f.write(json.dumps(example) + "\n")


    print(f"\n✅ Successfully generated {final_total_qa_pairs_generated} QA pairs from {final_total_contexts_generated} contexts → {output.resolve()}")

# --- Entry Point ---
if __name__ == "__main__":
    detection_json_path = "yolov12/896_anchor/detections/yolov12n/detections.json"
    output_file = "yolov12/896_anchor/detections/yolov12n/vqa_context_6Q.jsonl"
    
    # Generate 30,000 total contexts
    # 10% of contexts will have no piles detected.
    # The remaining 90% (27,000 contexts) will be split with 30% real/stat-fitted and 70% random.
    generate_dataset(detection_json_path, output_file, total_contexts=30000, target_hybrid_ratio=0.3, no_piles_ratio=0.10)