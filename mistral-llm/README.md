# Qwen 2.5 3B Fine-Tuning

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

