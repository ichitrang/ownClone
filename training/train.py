import json

INPUT_FILE = "data/founder_decisions_messages_full.json"
OUTPUT_FILE = "data/founder_dataset.jsonl"

def convert():
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)

    output = []

    for item in data:
        messages = item["messages"]
        system = messages[0]["content"]
        user = messages[1]["content"]
        assistant = messages[2]["content"]

        text = f"""<s>[INST] <<SYS>>
{system}
<</SYS>>

{user} [/INST] {assistant}</s>"""

        output.append({"text": text})

    with open(OUTPUT_FILE, "w") as f:
        for row in output:
            f.write(json.dumps(row) + "\n")

    print("Dataset ready!")

if __name__ == "__main__":
    convert()