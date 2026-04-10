import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

DEFAULT_MODEL_NAME = "mistralai/Mistral-7B-v0.1"
DEFAULT_DATASET_FILE = "data/founder_dataset.jsonl"
DEFAULT_OUTPUT_DIR = "founder-model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a base model using founder JSONL data")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset-file", default=DEFAULT_DATASET_FILE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_path = Path(args.dataset_file)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {args.dataset_file}. Run training/train_data.py first."
        )

    dataset = load_dataset("json", data_files=str(dataset_path))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Training complete. Saved model to {args.output_dir}")


if __name__ == "__main__":
    main()
