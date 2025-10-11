import os
from datasets import load_dataset

# --- Configuration ---
LANG_CODE = "kn"  # Kannada
OUTPUT_DIR = f"data/eng-{LANG_CODE}"
NUM_SAMPLES = 105000 # Download a bit more to account for empty lines

# --- Create Directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Download Data ---
print(f"Downloading {LANG_CODE} data from ai4bharat/samanantar...")
# Using streaming=True to avoid downloading the whole dataset
dataset = load_dataset("ai4bharat/samanantar", LANG_CODE, split=f"train[:{NUM_SAMPLES}]")
print("Download complete.")

# --- Split into Train/Test and Save ---
train_size = 100000
test_size = 5000

train_set = dataset.select(range(train_size))
test_set = dataset.select(range(train_size, train_size + test_size))

def save_to_files(dataset_split, split_name):
    src_path = os.path.join(OUTPUT_DIR, f"{split_name}.src")
    trg_path = os.path.join(OUTPUT_DIR, f"{split_name}.trg")
    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(trg_path, "w", encoding="utf-8") as f_trg:
        for item in dataset_split:
            f_src.write(item['src'] + "\n")
            f_trg.write(item['tgt'] + "\n")
    print(f"Saved {split_name} data to {src_path} and {trg_path}")

save_to_files(train_set, "train")
save_to_files(test_set, "test")

print("\nKannada data is ready!")
