import os
import shutil
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig 


### here the LLM will be trained using the vqa pairs created earlier


# Paths
model_name = "meta-llama/Llama-3.2-1B-Instruct"
data_path = "yolov12/896_anchor/detections/yolov12n/vqa_context_6Q.jsonl"
output_dir = "LLM/Meta-Llama-3.2-1B/Q10_context"

# Remove output_dir if it exists 
if os.path.exists(output_dir):
    print(f"Removing existing output directory: {output_dir}")
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

# Load dataset and split for evaluation
print(f"Loading dataset from: {data_path}")
raw_dataset = load_dataset("json", data_files=data_path, split="train")
dataset = raw_dataset.train_test_split(test_size=0.1, seed=42)
print(f"Dataset loaded. Train size: {len(dataset['train'])}, Test size: {len(dataset['test'])}")

# Load tokenizer
print(f"Loading tokenizer for model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 2048

# Load model with 4-bit quantization
print(f"Loading model {model_name} with 4-bit quantization...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
print("Model loaded.")

# Apply LoRA
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# SFT training config
print("Setting up SFT training configuration...")
sft_config = SFTConfig(
    output_dir=output_dir,
    num_train_epochs=4,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    eval_strategy="epoch",  
    logging_steps=10,
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=False,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    save_total_limit=2,
    report_to="none",
    seed=42,
    max_seq_length=2048,
    dataset_text_field="text", 
)

# Trainer
print("Initializing SFTTrainer...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer, # Use processing_class instead of tokenizer
)

# Train
print("Starting training...")
trainer.train()
print("Training complete.")

# Save model and tokenizer
print(f"Saving fine-tuned model and tokenizer to {output_dir}...")
model.save_pretrained(output_dir) 
tokenizer.save_pretrained(output_dir) 
print("Model and tokenizer saved.")

# Sanity check: Sample predictions
print("\nSample predictions:")
sample = dataset["test"][0]
prompt = sample["text"] if "text" in sample else sample.get("prompt", "")
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))