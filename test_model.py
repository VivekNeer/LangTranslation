import argparse
from simpletransformers.t5 import T5Model

MODEL_PATH = "VivekNeer/mt5-english-kannada-tulu"

TRANSLATION_DIRECTIONS = {
    "en-ka": "translate english to kannada",
    "ka-tu": "translate kannada to tulu",
}


def _format_input(prefix: str, text: str) -> str:
    return f"{prefix}: {text.strip()}"


def main():
    parser = argparse.ArgumentParser(description="Quickly test the bilingual translation model.")
    parser.add_argument("text", nargs="*", help="Text to translate")
    parser.add_argument(
        "--direction",
        choices=TRANSLATION_DIRECTIONS.keys(),
        default="en-ka",
        help="Translation direction (default: en-ka)",
    )
    args = parser.parse_args()

    sentence = " ".join(args.text).strip() or "hello my name is vivek"
    if not args.text:
        print("No text supplied; falling back to a default sentence.")

    print(f"Loading model from: {MODEL_PATH}")
    model = T5Model("mt5", MODEL_PATH, use_cuda=False)
    model.args.num_beams = 5
    model.args.max_length = 50
    print("Model loaded successfully.")

    prefixed = _format_input(TRANSLATION_DIRECTIONS[args.direction], sentence)
    print("-" * 40)
    print(f"Input ({args.direction}): {sentence}")
    translated_text = model.predict([prefixed])
    print(f"Output: {translated_text[0]}")
    print("-" * 40)


if __name__ == "__main__":
    main()
