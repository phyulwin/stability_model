from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"
OUTPUT_DIR = "outputs/mistral_peft_fast"
MAX_SEQ_LENGTH = 512


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


def main() -> None:
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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "validation": VAL_FILE},
    )

    dataset["train"] = dataset["train"].map(
        lambda x: render_chat(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    dataset["validation"] = dataset["validation"].map(
        lambda x: render_chat(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )

    dataset["train"] = dataset["train"].map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    dataset["validation"] = dataset["validation"].map(
        lambda x: tokenize_example(x, tokenizer),
        remove_columns=dataset["validation"].column_names,
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        load_best_model_at_end=False,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_steps=10,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Saved adapter to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
