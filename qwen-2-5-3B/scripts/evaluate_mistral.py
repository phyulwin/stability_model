from pathlib import Path
import json
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Base model and file paths used for evaluation.
BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_DIR = "outputs/qwen25_3b_peft_balanced_fast"
TEST_FILE = "data/test.jsonl"
OUTPUT_FILE = "outputs/qwen25_eval_results_balanced_fast.json"

# Only allow a very short output like 0 or 1.
MAX_NEW_TOKENS = 4


# Extract the predicted label from model output.
def parse_label(text: str) -> int:
    match = re.search(r"\b([01])\b", text.strip())
    if match:
        return int(match.group(1))
    return 0


def main() -> None:
    # Load tokenizer from the saved adapter folder.
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit mode to save GPU memory.
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load the base model, then attach the fine-tuned LoRA adapter.
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    # Load the test dataset from JSONL.
    dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]

    # Store true and predicted labels for scoring.
    y_true = []
    y_pred = []

    # Run inference one test example at a time.
    for example in dataset:
        user_msg = example["messages"][0]["content"]
        true_msg = example["messages"][1]["content"]

        # Build the prompt in chat format.
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

        # Convert the prompt into model input tensors.
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        # Generate a short prediction without updating model weights.
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Keep only the newly generated tokens, not the full prompt.
        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # Convert both true output and predicted output into 0/1 labels.
        y_true.append(parse_label(true_msg))
        y_pred.append(parse_label(decoded))

    # Compute evaluation metrics from all predictions.
    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "samples": len(y_true),
    }

    # Save the results as a JSON file.
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Print the results in the terminal.
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
