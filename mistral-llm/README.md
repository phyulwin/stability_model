# Qwen 2.5 3B Fine-Tuning

- `Qwen` is a family of large language models developed by `Alibaba Cloud`.
- It includes multiple model sizes, from smaller local models to larger high-capacity models.
- `Qwen2.5-3B-Instruct` is an instruction-tuned version designed to follow prompts and structured tasks.
- It is a decoder-only transformer model, similar in overall LLM style to models like Mistral and Llama.
- The `3B` means it has about `3 billion parameters`, which makes it smaller and faster than `7B` models.
- It supports chat-style prompting, which fits the JSONL `messages` format well for this stability analysis pipeline.
- It is strong at structured output tasks, such as returning short constrained answers like `0` or `1`.
- It is available on Hugging Face, which makes it easy to load with `transformers`, `PEFT`, and `bitsandbytes`.
- Compared with larger models, it is more practical for fine-tuning on consumer GPUs like an `RTX 3060 6GB`.
- In this project, it is being used as a binary classifier by fine-tuning it on pose-window prompts and stability labels.

## 1. Go to the project folder

```powershell
cd C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm
```

## 2. Create and activate the environment

```powershell
conda create -n mistral_hf python=3.10 -y
conda activate mistral_hf
```

## 3. Install dependencies

```powershell
pip install --upgrade pip
pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "numpy<2"
pip install "transformers==4.41.2" "peft==0.11.1" "accelerate==0.31.0" "bitsandbytes==0.43.1" "tokenizers==0.19.1"
pip install datasets scikit-learn pandas sentencepiece protobuf
```

## 4. Check GPU

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Expected:

- `2.3.0+cu121`
- `12.1`
- `True`
- `1`

## 5. Make sure these files exist

- [train.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\train.jsonl)
- [val.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\val.jsonl)
- [test.jsonl](C:\Users\646ca\Downloads\CPP\SeniorProject\stability_model\mistral-llm\data\test.jsonl)

## 6. Train the model

```powershell
python scripts/finetune_mistral.py
```

This uses:

- model: `Qwen/Qwen2.5-3B-Instruct`
- training file: `data/train.jsonl`
- output folder: `outputs/qwen25_3b_peft_fast`

## 7. Evaluate the model

```powershell
python scripts/evaluate_mistral.py
```

This uses:

- test file: `data/test.jsonl`
- adapter folder: `outputs/qwen25_3b_peft_fast`

## 8. Check the result

Evaluation results are saved in:

```text
outputs/qwen25_eval_results.json
```

This file includes:

- accuracy
- precision
- recall
- F1
- confusion matrix

