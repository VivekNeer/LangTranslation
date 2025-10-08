import sys
from simpletransformers.t5 import T5Model

# --- 1. Load your fully trained model ---
# This path should point to the model you trained for several hours
# model_path = "outputs/mt5-sinhalese-english-100k"
model_path = "VivekNeer/mt5-sinhalese-english"

print(f"Loading model from: {model_path}")
model = T5Model("mt5", model_path, use_cuda=False) # Ensure GPU is used if available

# --- 2. Set generation arguments for good quality output ---
model.args.num_beams = 5
model.args.max_length = 50

print("Model loaded successfully.")

# --- 3. Get the input sentence from the command line ---
# Check if a sentence was provided
if len(sys.argv) < 2:
    print("\nUsage: python test_model.py \"Your sentence to translate\"")
    # Use a default sentence if none is provided
    input_text = "hello my name is vivek"
    print(f"\nNo sentence provided. Using default: \"{input_text}\"")
else:
    # Join all arguments to form the sentence
    input_text = " ".join(sys.argv[1:])

# --- 4. Translate and print the output ---
print("-" * 30)
print(f"Input:    {input_text}")

# The model.predict() method handles the translation
translated_text = model.predict([input_text])

print(f"Output:   {translated_text[0]}")
print("-" * 30)
