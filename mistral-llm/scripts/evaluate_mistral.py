from pathlib import Path
import json
import re

import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_DIR = "outputs/mistral_peft"
TEST_FILE = "data/test.jsonl"
OUTPUT_FILE = "outputs/eval_results.json"
MAX_NEW_TOKENS = 4


def parse_label(text: str) -> int:
    match = re.search(r"\b([01])\b", text.strip())
    if match:
        return int(match.group(1))
    return 0


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    dataset = load_dataset("json", data_files={"test": TEST_FILE})["test"]

    y_true = []
    y_pred = []

    for example in dataset:
        user_msg = example["messages"][0]["content"]
        true_msg = example["messages"][1]["content"]

        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        y_true.append(parse_label(true_msg))
        y_pred.append(parse_label(decoded))

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "samples": len(y_true),
    }

    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
