import argparse
import json
from pathlib import Path

DEFAULT_INPUT_FILE = "data/founder_decisions_messages_full.json"
DEFAULT_OUTPUT_FILE = "data/founder_dataset.jsonl"


def format_record(messages: list[dict]) -> dict:
    if len(messages) < 3:
        raise ValueError("Each item must contain at least 3 messages: system, user, assistant")

    system = messages[0].get("content", "").strip()
    user = messages[1].get("content", "").strip()
    assistant = messages[2].get("content", "").strip()

    text = f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant}</s>"""
    return {"text": text}


def convert(input_file: Path, output_file: Path) -> int:
    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of conversation items")

    output_rows = [format_record(item["messages"]) for item in data]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(output_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert chat JSON data to SFT JSONL format")
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, help="Input JSON file path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, help="Output JSONL file path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert(Path(args.input), Path(args.output))
    print(f"Dataset ready! Wrote {count} rows to {args.output}")


if __name__ == "__main__":
    main()
