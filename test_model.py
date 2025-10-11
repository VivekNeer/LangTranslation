import sys
from simpletransformers.t5 import T5Model

# --- 1. Load your fully trained Kannada model ---
# This path must match the 'output_dir' from your train_model.py script
# --- CHANGED ---
model_path = "outputs/mt5-kannada-english-100k"

# Define the prefixes the model was trained on
# --- NEW ---
PREFIX_MAP = {
    "en-kn": "translate english to kannada",
    "kn-en": "translate kannada to english",
}

print(f"Loading model from: {model_path}")
# Set use_cuda=True if you have a GPU, otherwise False
model = T5Model("mt5", model_path, use_cuda=False)

# --- 2. Set generation arguments for good quality output ---
model.args.num_beams = 5
model.args.max_length = 50

print("Model loaded successfully.")

# --- 3. Get the direction and input sentence from the command line ---
# --- UPDATED LOGIC ---
if len(sys.argv) < 3:
    print("\nUsage: python test_model.py <direction> \"Your sentence to translate\"")
    print("       <direction> can be 'en-kn' (English->Kannada) or 'kn-en' (Kannada->English)")
    # Use a default sentence and direction if none is provided
    direction = "en-kn"
    input_text = "manas is a good boy."
    print(f"\nNo input provided. Using default direction '{direction}' and sentence: \"{input_text}\"")
else:
    direction = sys.argv[1]
    input_text = " ".join(sys.argv[2:])

# Validate the direction and get the correct prefix
if direction not in PREFIX_MAP:
    print(f"\nError: Invalid direction '{direction}'. Please use 'en-kn' or 'kn-en'.")
    sys.exit(1)

prefix = PREFIX_MAP[direction]
prefixed_input = f"{prefix}: {input_text}"


# --- 4. Translate and print the output ---
print("-" * 30)
print(f"Input to model: {prefixed_input}")

# The model.predict() method handles the translation
# We pass the full string with the prefix to the model
translated_text = model.predict([prefixed_input])

print(f"Output:         {translated_text[0]}")
print("-" * 30)