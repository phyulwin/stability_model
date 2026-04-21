from pathlib import Path
import random

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"
OUTPUT_DIR = "outputs/qwen25_3b_peft_balanced_fast"
MAX_SEQ_LENGTH = 512
MAX_TRAIN_SAMPLES = 4000
MAX_VAL_SAMPLES = 800
TARGET_POS_FRACTION = 0.25
SEED = 42


def render_chat(example: dict, tokenizer) -> dict:
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def tokenize_example(example: dict, tokenizer) -> dict:
    encoded = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        padding=False,
    )
    encoded["labels"] = encoded["input_ids"].copy()
    return encoded


def get_label(example: dict) -> int:
    return int(example["messages"][1]["content"].strip())


def rebalance_records(records: list[dict], max_samples: int, target_pos_fraction: float, seed: int) -> list[dict]:
    rng = random.Random(seed)

    positives = [r for r in records if get_label(r) == 1]
    negatives = [r for r in records if get_label(r) == 0]

    if not positives:
        raise ValueError("No positive examples found in training set.")
    if not negatives:
        raise ValueError("No negative examples found in training set.")

    target_pos = int(max_samples * target_pos_fraction)
    target_neg = max_samples - target_pos

    sampled_pos = [rng.choice(positives) for _ in range(target_pos)]
    sampled_neg = rng.sample(negatives, min(target_neg, len(negatives)))

    balanced = sampled_pos + sampled_neg
    rng.shuffle(balanced)
    return balanced


def sample_validation(records: list[dict], max_samples: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    if len(records) <= max_samples:
        return records
    return rng.sample(records, max_samples)


def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    raw = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

    train_records = [raw["train"][i] for i in range(len(raw["train"]))]
    val_records = [raw["validation"][i] for i in range(len(raw["validation"]))]

    train_records = rebalance_records(
        train_records,
        max_samples=MAX_TRAIN_SAMPLES,
        target_pos_fraction=TARGET_POS_FRACTION,
        seed=SEED,
    )
    val_records = sample_validation(
        val_records,
        max_samples=MAX_VAL_SAMPLES,
        seed=SEED,
    )

    train_dataset = Dataset.from_list(train_records)
    val_dataset = Dataset.from_list(val_records)

    train_dataset = train_dataset.map(
        lambda x: render_chat(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: render_chat(x, tokenizer),
        remove_columns=val_dataset.column_names,
    )

    train_dataset = train_dataset.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=val_dataset.column_names,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=25,
        evaluation_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=20,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
