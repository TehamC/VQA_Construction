# VQA_Construction

This repository contains the full pipeline for constructing a Visual Question Answering (VQA) dataset, training a Language Model (LLM) for VQA tasks, and evaluating its performance. The pipeline covers object detection, VQA data generation, LLM training, inference, and comprehensive evaluation.

---

## Getting Started

To get started with this project, ensure you have the necessary dependencies installed (e.g., Python, Git LFS, and any specific libraries you might need, typically listed in a `requirements.txt` file).

### Pipeline Workflow

Here's an exemplary use case showcasing the full VQA pipeline from data generation to evaluation:

1.  **Object Detection:**
    Detect objects on your images to prepare for VQA data generation.
    ```bash
    python VQA_Pipeline/1_datagen/2_pile_detection.py
    ```

2.  **VQA Data Generation:**
    Create VQA data, leveraging the detected objects to generate questions and answers with relevant context.
    ```bash
    python VQA_Pipeline/1_datagen/3_vqa_generation_context.py
    ```

3.  **LLM Training:**
    Train your Language Model (specifically Llama3_2) using the generated VQA data.
    ```bash
    python VQA_Pipeline/2_train/4_train_Llama3_2.py
    ```
    To monitor the training process, you can visualize the training curves:
    ```bash
    python VQA_Pipeline/2_train/training_curves.py
    ```

4.  **Run a Demo (Inference with Images):**
    Execute a demo to annotate images in a selected directory with responses to 6 predefined prompts using your trained LLM.
    ```bash
    python VQA_Pipeline/3_inference/6_demo_context_6Q.py
    ```

5.  **General LLM Inference (Without Images):**
    Test the performance of your trained LLM in a general context without relying on visual inputs.
    ```bash
    python VQA_Pipeline/3_inference/inference_Llama_context_general.py
    ```

6.  **Evaluate Demo Run:**
    Assess the performance of the LLM based on the results from the demo run.
    ```bash
    python VQA_Pipeline/4_eval/eval_demo.py
    ```

7.  **Test LLM Limits with Synthetic Scenarios:**
    Push the boundaries of your LLM by defining and evaluating synthetic scenarios. This process saves all results to a text file.
    ```bash
    python VQA_Pipeline/4_eval/eval_LLM_limits.py
    ```

8.  **Evaluate Synthetic Scenario Results:**
    Analyze the content of the text file generated from the LLM limit testing.
    ```bash
    python VQA_Pipeline/4_eval/eval_txt.py
    ```
