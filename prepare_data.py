import pandas as pd

# Define file paths for KANNADA data
# --- CHANGED ---
train_src_file = "data/eng-kn/train.src"
train_trg_file = "data/eng-kn/train.trg"
test_src_file = "data/eng-kn/test.src"
test_trg_file = "data/eng-kn/test.trg"

# Read the data
with open(train_src_file, "r", encoding="utf-8") as f:
    train_src = f.read().split("\n")
with open(train_trg_file, "r", encoding="utf-8") as f:
    train_trg = f.read().split("\n")

with open(test_src_file, "r", encoding="utf-8") as f:
    test_src = f.read().split("\n")
with open(test_trg_file, "r", encoding="utf-8") as f:
    test_trg = f.read().split("\n")

# Create DataFrames with KANNADA prefixes
# --- CHANGED ---
train_data = []
for src, trg in zip(train_src, train_trg):
    if src and trg: # Ensure lines are not empty
        train_data.append(["translate english to kannada", src, trg])
        train_data.append(["translate kannada to english", trg, src])

train_df = pd.DataFrame(train_data, columns=["prefix", "input_text", "target_text"])

eval_data = []
for src, trg in zip(test_src, test_trg):
    if src and trg: # Ensure lines are not empty
        eval_data.append(["translate english to kannada", src, trg])
        eval_data.append(["translate kannada to english", trg, src])

eval_df = pd.DataFrame(eval_data, columns=["prefix", "input_text", "target_text"])

# Save to the same TSV files (they will be overwritten)
train_df.to_csv("data/train.tsv", sep="\t", index=False)
eval_df.to_csv("data/eval.tsv", sep="\t", index=False)

print("Data preparation complete. Files saved to data/train.tsv and data/eval.tsv")