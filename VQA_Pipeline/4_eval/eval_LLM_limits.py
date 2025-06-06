import random
import math
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
import re


### this is to test the LLMs limits and try out edge cases ie behavior for increasing number of piles
### everything else is randomized
### this will just record the data(in .txt) eval will happen seperately


warnings.filterwarnings(
    "ignore",
    message=r"The following generation flags are not valid and may be ignored: \['temperature', 'top_p'\].*",
    category=UserWarning,
    module="transformers" # This makes sure it only applies to warnings from the transformers library
)

# --- Configuration for LLM ---
TRAINED_MODEL_PATH = "LLM/Meta-Llama-3.2-1B/Q6_context"
BASE_MODEL_NAME     = "meta-llama/Llama-3.2-1B-Instruct"

# If GPU is available, use it; otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_random_context(num_piles):
    anchor_x = random.uniform(0, 1000)
    anchor_y = random.uniform(0, 1000)
    piles = []
    for i in range(num_piles):
        x = random.uniform(0, 1000)
        y = random.uniform(0, 1000)
        area = random.uniform(1000, 30000)
        piles.append((f"pile{i+1}", x, y, area))

    context = (
        "You are a caterpillar in a construction site. In the following you will be given geometric data of piles of gravel "
        "such as their position and size. Alongside the piles you will be given an anchor (your position), which you can use "
        "as a reference to determine distances and relative positions.\n"
        f"Anchor position: ({anchor_x:.1f}, {anchor_y:.1f})\n"
        "Following piles are present:\n"
        + "\n".join([
            f"{name}: position=({x:.1f}, {y:.1f}), area={area:.1f}"
            for name, x, y, area in piles
        ])
    )
    return context, piles, (anchor_x, anchor_y)


def get_ground_truth(task, piles, anchor):
    """
    Returns (ground_truth_str, metric_str).  metric_str is used to compare correctness.
    """
    if not piles:
        return "No piles", ""

    if task == "Start at the rightmost pile":
        pile = max(piles, key=lambda p: p[1])  # compare x coordinate
        metric = f"x-coordinate={pile[1]:.1f}"
        return (
            f"drive to {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    elif task == "Clear a remote pile":
        pile = max(
            piles,
            key=lambda p: math.hypot(p[1] - anchor[0], p[2] - anchor[1])
        )
        dist = math.hypot(pile[1] - anchor[0], pile[2] - anchor[1])
        metric = f"distance={dist:.1f}"
        return (
            f"drive to the farthest pile: {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    elif task == "Clear a pile as fast as possible":
        pile = min(piles, key=lambda p: p[3])  # compare area
        metric = f"area={pile[3]:.1f}"
        return (
            f"drive to the smallest pile: {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    elif task == "Start at the leftmost pile":
        pile = min(piles, key=lambda p: p[1])
        metric = f"x-coordinate={pile[1]:.1f}"
        return (
            f"drive to {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    elif task == "Process the largest pile":
        pile = max(piles, key=lambda p: p[3])
        metric = f"area={pile[3]:.1f}"
        return (
            f"drive to the largest pile: {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    elif task == "Fill the shovel":
        pile = min(
            piles,
            key=lambda p: math.hypot(p[1] - anchor[0], p[2] - anchor[1])
        )
        dist = math.hypot(pile[1] - anchor[0], pile[2] - anchor[1])
        metric = f"distance={dist:.1f}"
        return (
            f"drive to the nearest pile: {pile[0]}[{pile[1]:.1f}, {pile[2]:.1f}] and initiate digging",
            metric
        )

    return "Unknown task", ""


def run_inference(context, task, model, tokenizer, piles, anchor):
    """
    Builds the prompt, runs model.generate, and extracts just the first 'pileX[…]' answer.
    Returns (answer_str, metric_str, inference_time).
    """
    prompt = f"<s>[INST] {context}\nTask: {task} [/INST]"

    # Tokenize with attention_mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="longest",
        max_length=2048
    )
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    # Define all possible generation arguments, setting sampling ones to None
    generation_params = {
        "max_new_tokens": 256, # Adjust this value as needed for full responses
        "do_sample": False,
        "temperature": None,  # Will be filtered out when do_sample is False
        "top_p": None,        # Will be filtered out when do_sample is False
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
        # Add any other generation parameters you might use here (e.g., num_beams, repetition_penalty)
    }

    # Filter out parameters that have None values
    # This ensures that 'temperature' and 'top_p' are not passed when they are None
    filtered_generation_params = {
        k: v for k, v in generation_params.items() if v is not None
    }


    start_time = time.time()
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attn_mask,
        **filtered_generation_params
    )

    inference_time = time.time() - start_time

    full_decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the substring immediately after "[/INST]"
    if "[/INST]" in full_decoded:
        after = full_decoded.split("[/INST]")[-1].strip()
    else:
        after = full_decoded

    # Only keep the first “pileX[…]" phrase (stop at first newline or period)
    # e.g. "pile2[724.4, 980.4] and initiate digging"
    # If the model accidentally appends extra text, we ignore it.
    first_line = after.split("\n")[0].split(".")[0].strip()

    # Recompute metric for the model’s predicted pile (so we can compare correctness later)
    # We compare that metric to the ground‐truth metric from get_ground_truth.
    if task == "Start at the rightmost pile":
        pile = max(piles, key=lambda p: p[1]) if piles else None
        metric = f"x-coordinate={pile[1]:.1f}" if pile else ""

    elif task == "Clear a remote pile":
        pile = max(
            piles,
            key=lambda p: math.hypot(p[1] - anchor[0], p[2] - anchor[1])
        ) if piles else None
        if pile:
            dist = math.hypot(pile[1] - anchor[0], pile[2] - anchor[1])
            metric = f"distance={dist:.1f}"
        else:
            metric = ""

    elif task == "Clear a pile as fast as possible":
        pile = min(piles, key=lambda p: p[3]) if piles else None
        metric = f"area={pile[3]:.1f}" if pile else ""

    elif task == "Start at the leftmost pile":
        pile = min(piles, key=lambda p: p[1]) if piles else None
        metric = f"x-coordinate={pile[1]:.1f}" if pile else ""

    elif task == "Process the largest pile":
        pile = max(piles, key=lambda p: p[3]) if piles else None
        metric = f"area={pile[3]:.1f}" if pile else ""

    elif task == "Fill the shovel":
        pile = min(
            piles,
            key=lambda p: math.hypot(p[1] - anchor[0], p[2] - anchor[1])
        ) if piles else None
        if pile:
            dist = math.hypot(pile[1] - anchor[0], pile[2] - anchor[1])
            metric = f"distance={dist:.1f}"
        else:
            metric = ""

    else:
        metric = ""

    return first_line, metric, inference_time


def main():
    # Load the fine‐tuned model and move to device
    print("Loading model and tokenizer... (may take a moment)")
    tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL_PATH) # Load tokenizer from the fine-tuned path
    tokenizer.pad_token = tokenizer.eos_token # This is good practice
    model = AutoModelForCausalLM.from_pretrained(TRAINED_MODEL_PATH).to(device)

    tasks = [
        "Start at the rightmost pile",
        "Clear a remote pile",
        "Clear a pile as fast as possible",
        "Start at the leftmost pile",
        "Process the largest pile",
        "Fill the shovel"
    ]

    success_counts = {task: 0 for task in tasks}
    total_counts   = {task: 0 for task in tasks}

    with open("inference_50_piles_3.txt", "w") as log_file:
        for num_piles in range(1, 11):
            context, piles, anchor = generate_random_context(num_piles)

            header = (
                f"\n=== Number of Piles: {num_piles} ===\n"
                f"Context:\n{context}\n\n"
            )
            print(header.strip())
            log_file.write(header + "\n")

            for task in tasks:
                answer, model_metric, inference_time = run_inference(
                    context, task, model, tokenizer, piles, anchor
                )
                ground_truth, gt_metric = get_ground_truth(task, piles, anchor)
                is_correct = (model_metric == gt_metric)

                success_counts[task] += 1 if is_correct else 0
                total_counts[task]   += 1

                task_output = (
                    f"Task: {task}\n"
                    f"  Model Answer: {answer}    (Metric: {model_metric})\n"
                    f"  Ground Truth: {ground_truth}    (Metric: {gt_metric})\n"
                    f"  Correct: {is_correct}\n"
                    f"  Inference Time: {inference_time:.2f} s\n"
                )
                print(task_output)
                log_file.write(task_output + "\n")

        # Finally, compute average success rates
        summary = "\n=== Average Success Rates ===\n"
        for task in tasks:
            if total_counts[task] > 0:
                rate = success_counts[task] / total_counts[task] * 100
            else:
                rate = 0.0
            summary += f"{task}: {rate:.2f}%\n"

        print(summary)
        log_file.write(summary)

    print("Done.  See inference_log.txt for details.")


if __name__ == "__main__":
    main()
