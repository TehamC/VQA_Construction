import json
import matplotlib.pyplot as plt
import pandas as pd
import os

### here training plots are created


# training progress data
trainer_state_path = "LLM/Meta-Llama-3.2-1B/Q6_context/checkpoint-40500/trainer_state.json"

# path for saving figures 
figures_save_dir = "LLM/evaluation_visual/6Q_context"

# Create the save directory if it doesn't exist
os.makedirs(figures_save_dir, exist_ok=True)
print(f"Figures will be saved to: {figures_save_dir}")

# --- Load Data ---
if not os.path.exists(trainer_state_path):
    print(f"Error: trainer_state.json not found at {trainer_state_path}.")
    print("Please ensure the path is correct.")
    exit()

with open(trainer_state_path, "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state.get("log_history", [])

if not log_history:
    print("No log history found in trainer_state.json.")
    exit()

# Prepare data for plotting
train_metrics = []
eval_metrics = []

for entry in log_history:
    if "loss" in entry and "learning_rate" in entry: # Training step log
        train_metrics.append({
            "step": entry["step"],
            "epoch": entry["epoch"],
            "loss": entry["loss"],
            "learning_rate": entry["learning_rate"],
            "mean_token_accuracy": entry.get("mean_token_accuracy") 
        })
    elif "eval_loss" in entry: # Evaluation epoch log
        eval_metrics.append({
            "step": entry["step"], # Evaluation logs also have a step field
            "epoch": entry["epoch"],
            "eval_loss": entry["eval_loss"],
            "eval_mean_token_accuracy": entry.get("mean_token_accuracy") 
        })

df_train = pd.DataFrame(train_metrics)
df_eval = pd.DataFrame(eval_metrics)

# --- Plotting and Saving Function ---

def plot_and_save_metric(df, x_col, y_col, title, x_label, y_label, color='blue', filename_prefix="plot"):
    """Generates a single plot for a given metric and saves it."""
    if df.empty or y_col not in df.columns or x_col not in df.columns:
        print(f"No data to plot for {y_col} or {x_col}. Skipping {title}.")
        return

    plt.figure(figsize=(10, 6)) 
    ax = plt.gca() 

    plt.plot(df[x_col], df[y_col], label=y_label, color=color)

    # Set font sizes
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12) 

    plt.legend(fontsize=12) 
    plt.grid(True)
    plt.tight_layout() 


    sanitized_title = "".join(c for c in title if c.isalnum() or c == ' ').replace(" ", "_").lower()
    filename = f"{filename_prefix}_{sanitized_title}.png"
    save_path = os.path.join(figures_save_dir, filename)
    plt.savefig(save_path, dpi=300) 
    plt.close() 
    print(f"Saved plot: {save_path}")

# --- Generate and Save Plots ---

# 1. Training Loss
plot_and_save_metric(df_train, "step", "loss",
                     "Training Loss Over Steps", "Steps", "Loss", color='blue',
                     filename_prefix="training_loss")

# 2. Validation Loss
plot_and_save_metric(df_eval, "step", "eval_loss",
                     "Validation Loss Over Steps", "Steps", "Loss", color='orange',
                     filename_prefix="validation_loss")

# 3. Training Mean Token Accuracy
plot_and_save_metric(df_train, "step", "mean_token_accuracy",
                     "Training Mean Token Accuracy Over Steps", "Steps", "Accuracy", color='green',
                     filename_prefix="training_accuracy")

# 4. Learning Rate
plot_and_save_metric(df_train, "step", "learning_rate",
                     "Learning Rate Over Steps", "Steps", "Learning Rate", color='purple',
                     filename_prefix="learning_rate")

print("\nAll plots generated and saved. Check the specified output directory.")


print("\nTraining Metrics Sample (First 5 and Last 5):")
print(df_train.head())
print(df_train.tail())

print("\nValidation Metrics Sample (First 5 and Last 5):")
print(df_eval.head())
print(df_eval.tail())