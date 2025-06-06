import re
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np

### here the txt file created by the eval LLM limits will be analyzed


# === CONFIGURATION ===
INPUT_TXT = "inference_50_piles.txt"  

# === HELPERS FOR PARSING ===

def extract_predicted_pile(model_answer_line):
    """
    Return 'pileN' if found in the model answer string, else None.
    """
    m = re.search(r"pile(\d+)", model_answer_line)
    if m:
        return f"pile{m.group(1)}"
    return None

def extract_ground_truth_pile(gt_line):
    """
    Return 'pileN' if found in the ground truth string, else None.
    """
    m = re.search(r"pile(\d+)", gt_line)
    if m:
        return f"pile{m.group(1)}"
    return None

# === STEP 1: READ AND SPLIT INTO “Number of Piles” SECTIONS ===

with open(INPUT_TXT, "r") as f:
    content = f.read()

# Split on the header marker; each section starts with "<num> ==="
sections = content.strip().split("=== Number of Piles:")
sections = sections[1:]  # discard anything before the first split

# Data structure:
#   task_data[task_name][num_piles] = { "times": [...], "codes": [...] }
# where codes are: +1 = correct, 0 = wrong, -1 = weird.
task_data = defaultdict(lambda: defaultdict(lambda: {"times": [], "codes": []}))

for section in sections:
    lines = section.splitlines()
    header = lines[0].strip()  # e.g. " 3 ==="
    try:
        num_piles = int(header.split("===")[0].strip())
    except ValueError:
        continue

    # Group nonblank lines into task blocks
    task_blocks = []
    current_block = []
    for line in lines[1:]:
        if line.strip() == "":
            if current_block:
                task_blocks.append(current_block)
                current_block = []
        else:
            current_block.append(line)
    if current_block:
        task_blocks.append(current_block)

    # Parse each block
    for block in task_blocks:
        block_text = "\n".join(block)

        # Extract Task name
        tm = re.search(r"Task:\s*(.+)", block_text)
        if not tm:
            continue
        task_name = tm.group(1).strip()

        # Extract Model Answer line
        mm = re.search(r"Model Answer:\s*(.+)", block_text)
        if not mm:
            continue
        model_answer_line = mm.group(1).strip()

        # Extract Ground Truth line
        gm = re.search(r"Ground Truth:\s*(.+)", block_text)
        if not gm:
            continue
        gt_line = gm.group(1).strip()

        # Extract Inference Time
        tm2 = re.search(r"Inference Time:\s*([\d.]+)", block_text)
        if not tm2:
            continue
        inference_time = float(tm2.group(1))

        # Determine correctness code
        pred_pile = extract_predicted_pile(model_answer_line)
        gt_pile = extract_ground_truth_pile(gt_line)

        if pred_pile is None:
            correctness_code = -1
        else:
            correctness_code = 1 if (pred_pile == gt_pile) else 0

        # Store
        task_data[task_name][num_piles]["times"].append(inference_time)
        task_data[task_name][num_piles]["codes"].append(correctness_code)

# === STEP 2: AGGREGATE AND CHOOSE MARKER COLOR/SHAPE ===

# Will hold for each task:
#   aggregated[task][num_piles] = {
#       "avg_time": float,
#       "marker": 'o' or 'x',
#       "color": 'green' or 'red' or 'black'
#   }
aggregated = {}

for task_name, pile_dict in task_data.items():
    aggregated[task_name] = {}
    for num_piles, info in pile_dict.items():
        times = info["times"]
        codes = info["codes"]

        if not times:
            continue
        avg_time = sum(times) / len(times)

        # Decide marker & color:
        # If any code == -1 → weird → black 'x'
        # Else if any code == 0  → wrong → red  'o'
        # Else → all correct → green 'o'
        if any(c == -1 for c in codes):
            marker = "x"
            color = "black"
        elif any(c == 0 for c in codes):
            marker = "o"
            color = "red"
        else:
            marker = "o"
            color = "green"

        aggregated[task_name][num_piles] = {
            "avg_time": avg_time,
            "marker": marker,
            "color": color
        }

# === STEP 3: PLOTTING ===

all_tasks = sorted(aggregated.keys())
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
axs = axs.flatten()

for idx, task_name in enumerate(all_tasks):
    ax = axs[idx]
    data_for_task = aggregated[task_name]
    if not data_for_task:
        ax.set_title(task_name + "\n(no data)")
        continue

    pile_counts = sorted(data_for_task.keys())
    for p in pile_counts:
        entry = data_for_task[p]
        ax.scatter(
            p,
            entry["avg_time"],
            marker=entry["marker"],
            color=entry["color"],
            s=100,
            label="_nolegend_"
        )

    # Draw a (light gray) line connecting the points (for visibility)
    times_line = [data_for_task[p]["avg_time"] for p in pile_counts]
    ax.plot(pile_counts, times_line, linestyle="--", color="gray", alpha=0.5)

    ax.set_title(task_name, fontsize=12)
    ax.set_xlabel("Number of Piles", fontsize=10)
    ax.set_ylabel("Avg Inference Time (s)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.3)

    # Build a custom legend patch
    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="Correct", markerfacecolor="green", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Wrong", markerfacecolor="red", markersize=8),
        Line2D([0], [0], marker="x", color="w", label="Weird Reply", markeredgecolor="black", markersize=8),
    ]
    ax.legend(handles=legend_elems, fontsize=9, loc="upper left")

plt.tight_layout()
plt.suptitle("Avg Inference Time vs Number of Piles (Marker Color=Correctness)", fontsize=16, y=1.02)
plt.subplots_adjust(top=0.90)
plt.show()
plt.savefig("LLM/evaluation_visual/6Q_context/task_inference_2.png")