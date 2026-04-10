import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = "founder-model"
SYSTEM_PROMPT = (
    "You are my business clone. Respond with practical, balanced, calm decisions "
    "in concise format: Decision, Reason, Next step."
)


def build_prompt(user_input: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    return (
        f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"
        f"{user_input.strip()} [/INST]"
    )


def load_model_and_tokenizer(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    return model, tokenizer


def generate_reply(model, tokenizer, user_input: str, max_new_tokens: int = 180) -> str:
    prompt = build_prompt(user_input)
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    with torch.no_grad():
        output = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)
    if "[/INST]" in decoded:
        return decoded.split("[/INST]", 1)[1].replace("</s>", "").strip()
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chat with your fine-tuned founder clone model.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Path to local model directory")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    print("Founder clone chat is ready. Type 'exit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        reply = generate_reply(model, tokenizer, user_input, max_new_tokens=args.max_new_tokens)
        print(f"Clone: {reply}\n")


if __name__ == "__main__":
    main()
