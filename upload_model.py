from simpletransformers.t5 import T5Model

# 1. DEFINE YOUR MODEL PATHS
# The local path where your trained model is saved
local_model_path = "outputs/mt5-sinhalese-english-100k"

# The name you want for your model on the Hub.
hub_model_name = "VivekNeer/mt5-sinhalese-english"

# 2. LOAD THE LOCAL SIMPLETRANSFORMERS MODEL
print(f"Loading model from {local_model_path}...")
model = T5Model("mt5", local_model_path)

# --- THIS IS THE CORRECT METHOD ---
# 3. Access the underlying Hugging Face objects
hf_model = model.model
hf_tokenizer = model.tokenizer

# 4. Push both the model and the tokenizer to the Hub separately
print(f"Uploading model to {hub_model_name}...")
hf_model.push_to_hub(hub_model_name)

print(f"Uploading tokenizer to {hub_model_name}...")
hf_tokenizer.push_to_hub(hub_model_name)
# --- END OF CORRECTION ---

print("Upload complete! You can now see the model on your Hugging Face profile.")