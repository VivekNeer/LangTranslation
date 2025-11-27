import argparse
import torch
from simpletransformers.t5 import T5Model

MODEL_PATH = "outputs/mt5-english-tulu"  # Final trained model (all 10 epochs)
TRANSLATION_PREFIX = "translate english to tulu"

def _format_input(prefix: str, text: str) -> str:
    return f"{prefix}: {text.strip()}"

def main():
    parser = argparse.ArgumentParser(description="Quickly test the English->Tulu translation model.")
    parser.add_argument("text", nargs="*", help="Text to translate")
    parser.add_argument(
        "--max_length",
        type=int,
        default=50,
        help="Maximum output token length (default: 50)",
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search (default: 5)",
    )
    args = parser.parse_args()

    sentence = " ".join(args.text).strip() or "hello my name is vivek"
    if not args.text:
        print("No text supplied; falling back to a default sentence.")

    print(f"Loading model from: {MODEL_PATH}")
    use_cuda = torch.cuda.is_available()
    if not use_cuda:
        print("CUDA not available - loading model on CPU (use_cuda=False).")
    model = T5Model("mt5", MODEL_PATH, use_cuda=use_cuda)

    # Tweak generation args
    model.args.num_beams = args.num_beams
    model.args.max_length = args.max_length

    print("Model loaded successfully.")
    prefixed = _format_input(TRANSLATION_PREFIX, sentence)
    print("-" * 40)
    print(f"Input: {sentence}")
    translated_text = model.predict([prefixed])
    print(f"Output: {translated_text[0]}")
    print("-" * 40)

if __name__ == "__main__":
    main()
