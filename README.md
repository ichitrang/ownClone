# OwnClone

Train a personal "business decision clone" model from your JSON conversations, then chat with it locally.

## What this system does

This project turns conversation-style training data into a supervised fine-tuning dataset and fine-tunes a causal language model (default: Mistral 7B). After training, you can run an interactive CLI chat with your clone persona.

Pipeline:
1. `training/train_data.py`: converts message JSON into JSONL prompt/completion format.
2. `training/train.py`: fine-tunes the base model on the converted dataset.
3. `inference/chat.py`: loads the trained model and runs interactive chat.

---

## Local setup

### Option A: Python on host

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Docker

```bash
docker compose build
docker compose run --rm ownclone bash
```

Inside the container:

```bash
pip install -r requirements.txt
```

---

## Prepare your training data

Input data expected at: `data/founder_decisions_messages_full.json`

Convert it to JSONL:

```bash
python training/train_data.py \
  --input data/founder_decisions_messages_full.json \
  --output data/founder_dataset.jsonl
```

---

## Train your clone model

> Recommended: GPU with enough VRAM for your selected model.

```bash
python training/train.py \
  --model-name mistralai/Mistral-7B-v0.1 \
  --dataset-file data/founder_dataset.jsonl \
  --output-dir founder-model \
  --epochs 3
```

---

## Run interactive chat

```bash
python inference/chat.py --model-path founder-model
```

Type `exit` or `quit` to stop.

---

## Test cases (what to validate)

### 1) Data conversion test cases
- Valid input JSON with list of items and 3 messages each -> should generate one JSONL row per item.
- Missing/invalid input file -> should fail with file error.
- Wrong JSON structure (not list) -> should raise validation error.

### 2) Training test cases
- Missing `data/founder_dataset.jsonl` -> should instruct user to run data conversion first.
- Valid small JSONL dataset -> training starts and writes files to `--output-dir`.
- CPU-only environment -> script should run (slowly) without CUDA.

### 3) Inference test cases
- Missing model path -> should fail fast when loading model/tokenizer.
- Valid trained model path -> should open prompt loop.
- Exit behavior -> typing `exit` or `quit` should end chat cleanly.

---

## Quick verification commands

```bash
python training/train_data.py --input data/founder_decisions_messages_full.json --output data/founder_dataset.jsonl
python -m py_compile training/train_data.py training/train.py inference/chat.py
python training/train.py --help
python inference/chat.py --help
```
