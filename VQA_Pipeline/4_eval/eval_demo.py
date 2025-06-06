import json
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from pathlib import Path


### this will do further eval of the demo by using the json file created

# === Configuration ===
FONT_SIZE = 12
BAR_COLOR_SUCCESS   = '#4CAF50'  # Green for success‐rate bars
BAR_COLOR_INFERENCE = '#2196F3'  # Blue for inference‐time bars
OUTPUT_DIR = Path('yolov12/896_anchor/detections/v5_demo/')
ERROR_JSON = OUTPUT_DIR / 'LLM_errors_v5.json'

# === Load JSON data ===
with open(OUTPUT_DIR / 'v5_results.json', 'r') as f:
    data = json.load(f)

# === Initialize variables for evaluation ===
task_success    = defaultdict(list)  # collects 1/0 per task name
inference_times = defaultdict(list)  # collects LLM time per number of piles
error_entries   = []                 # list of dicts for failures
skipped_images  = 0                  # count images with no piles

# === Process each image result ===
for result in data.get('results', []):
    image       = result.get('image', 'UNKNOWN_IMAGE')
    pile_list   = result.get('piles', [])   # expecting list of {name, position, area, bbox}
    num_piles   = len(pile_list)

    # Skip if no piles detected
    if num_piles == 0:
        skipped_images += 1
        continue

    # Record inference time under this pile‐count (if present)
    llm_time = result.get('computation_times', {}).get('llm_inference', None)
    if llm_time is not None:
        inference_times[num_piles].append(llm_time)

    # Build a list of {name, bbox} for error reporting
    piles_for_error = []
    for pile in pile_list:
        pile_name = pile.get('name', 'UNKNOWN_PILE')
        bbox      = pile.get('bbox', [])
        # Only record if bbox has 4 elements
        if isinstance(bbox, list) and len(bbox) == 4:
            piles_for_error.append({
                "name": pile_name,
                "bbox": bbox
            })

    # Evaluate each task under this image
    for task_info in result.get('tasks', []):
        task_name  = task_info.get('task', 'UNKNOWN_TASK')
        predicted  = task_info.get('prediction', None)
        calculated = task_info.get('calculated', None)

        success = (predicted == calculated) and (predicted is not None)
        task_success[task_name].append(1 if success else 0)

        if not success:
            error_entry = {
                "image": image,
                # include the list of piles (name + bbox) so you can inspect coordinates
                "piles": piles_for_error,
                "task": task_name,
                "calculated": calculated,
                "predicted": predicted
            }
            error_entries.append(error_entry)

# === Plot: Task Success Bar Chart ===
task_avg_success = {
    task: float(np.mean(successes)) 
    for task, successes in task_success.items()
}

plt.figure(figsize=(10, 6))
bars = plt.bar(task_avg_success.keys(), task_avg_success.values(), color=BAR_COLOR_SUCCESS)

for bar in bars:
    height = bar.get_height()
    offset = -0.02
    va = 'bottom' if offset > 0 else 'top'
    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        height + offset,
        f"{height:.2f}",
        ha='center',
        va=va,
        fontsize=FONT_SIZE,
        color='white'
    )

plt.xlabel('Task', fontsize=FONT_SIZE)
plt.ylabel('Average Success Rate', fontsize=FONT_SIZE)
plt.title('Average Success Rate per Task', fontsize=FONT_SIZE + 2)
plt.xticks(rotation=45, ha='right', fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v5_task_success_rate.png')
plt.show()

# === Plot: Inference Time by Number of Piles Bar Chart ===
avg_inference_times = {
    piles: float(np.mean(times)) 
    for piles, times in inference_times.items()
}

sorted_pile_counts     = sorted(avg_inference_times.keys())
sorted_inference_times = [avg_inference_times[p] for p in sorted_pile_counts]
str_pile_labels        = [str(p) for p in sorted_pile_counts]

plt.figure(figsize=(10, 6))
# Skip index 0 since we skip images with 0 piles
bars = plt.bar(str_pile_labels[1:], sorted_inference_times[1:], color=BAR_COLOR_INFERENCE)

for bar in bars:
    height = bar.get_height()
    offset = 0.03 * max(sorted_inference_times)
    inside = height > offset * 2
    va     = 'center' if inside else 'bottom'
    y_pos  = (height - offset) if inside else (height + offset)
    color  = 'white' if inside else 'black'

    plt.text(
        bar.get_x() + bar.get_width() / 2.0,
        y_pos,
        f"{height:.2f}s",
        ha='center',
        va=va,
        fontsize=FONT_SIZE,
        color=color
    )

plt.xlabel('Number of Piles Detected', fontsize=FONT_SIZE)
plt.ylabel('Average LLM Inference Time (s)', fontsize=FONT_SIZE)
plt.title('Average LLM Inference Time by Number of Piles', fontsize=FONT_SIZE + 2)
plt.xticks(fontsize=FONT_SIZE)
plt.yticks(fontsize=FONT_SIZE)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'v5_inference_time_by_piles.png')
plt.show()

# === Write Errors to error.json ===
with open(ERROR_JSON, 'w') as f:
    json.dump({"errors": error_entries, "skipped_images": skipped_images}, f, indent=2)

print(f"Skipped {skipped_images} images with 0 detected piles.")
print(f"Saved {len(error_entries)} error entries to {ERROR_JSON}")
