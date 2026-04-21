# Mistral Fine-Tuning on Windows

This workflow fine-tunes Mistral using `transformers + peft + bitsandbytes` on Windows.

It does **not** use Unsloth.

It uses your existing JSONL files directly:

- [train.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\train.jsonl)
- [val.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\val.jsonl)
- [test.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\test.jsonl)

## Files You Need

- `scripts/finetune_mistral.py`
- `scripts/evaluate_mistral.py`
- `data/train.jsonl`
- `data/val.jsonl`
- `data/test.jsonl`

You do **not** need to rerun `build_jsonl.py` unless you want to rebuild the dataset.

## 1. Create a Clean Environment

```powershell
conda create -n mistral_hf python=3.10 -y
conda activate mistral_hf
```

## 2. Install Packages

```powershell
pip install --upgrade pip
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate peft bitsandbytes datasets scikit-learn pandas sentencepiece protobuf
```

## 3. Verify GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "import bitsandbytes as bnb; print(bnb.__version__)"
```

Expected:

- `torch.cuda.is_available()` -> `True`
- `torch.cuda.device_count()` -> `1`

## 4. Put the JSONL Files in `data/`

Your scripts should point to:

```python
TRAIN_FILE = "data/train.jsonl"
VAL_FILE = "data/val.jsonl"
TEST_FILE = "data/test.jsonl"
```

If your current files are in:

`C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\`

then run the scripts from that project folder, or update the paths in the scripts.

## 5. Train

```powershell
python scripts/finetune_mistral.py
```

This will:

- load `mistralai/Mistral-7B-Instruct-v0.3`
- load the training and validation JSONL files
- fine-tune LoRA adapters
- save the adapter output

Expected output folder:

```text
outputs/mistral_peft
```

## 6. Evaluate

```powershell
python scripts/evaluate_mistral.py
```

This will:

- load the base model
- load the saved adapter
- run on `test.jsonl`
- save metrics

Expected output file:

```text
outputs/eval_results.json
```

## 7. Metrics to Report

At minimum, report:

- accuracy
- precision
- recall
- F1
- confusion matrix

Your professor wants `70% accuracy`, so include that clearly.

## 8. Full Run Order

```powershell
conda activate mistral_hf
python scripts/finetune_mistral.py
python scripts/evaluate_mistral.py
```

## 9. If You Hit Out-of-Memory

Edit `scripts/finetune_mistral.py` and reduce:

```python
MAX_SEQ_LENGTH = 1024
```

to:

```python
MAX_SEQ_LENGTH = 768
```

or:

```python
MAX_SEQ_LENGTH = 512
```

## Summary

Use the existing JSONL files.

Do not use Unsloth.

Train with:

- `transformers`
- `peft`
- `bitsandbytes`

Then evaluate on `test.jsonl`.

