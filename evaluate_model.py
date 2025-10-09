import pandas as pd
from simpletransformers.t5 import T5Model
import sacrebleu

# --- Configuration ---
MODEL_PATH = "outputs/mt5-sinhalese-english-100k"
EVAL_DATA_FILE = "data/eval.tsv"

# --- Load Model and Data ---
print("Loading fine-tuned model...")
model = T5Model("mt5", MODEL_PATH, use_cuda=True)
eval_df = pd.read_csv(EVAL_DATA_FILE, sep="\t").astype(str)

# --- Prepare Data for Evaluation ---
# We want to test both translation directions
to_predict_en_to_si = eval_df[eval_df["prefix"] == "translate english to sinhalese"]["input_text"].tolist()
ground_truth_en_to_si = eval_df[eval_df["prefix"] == "translate english to sinhalese"]["target_text"].tolist()

to_predict_si_to_en = eval_df[eval_df["prefix"] == "translate sinhalese to english"]["input_text"].tolist()
ground_truth_si_to_en = eval_df[eval_df["prefix"] == "translate sinhalese to english"]["target_text"].tolist()

# --- Generate Predictions ---
print("\nGenerating English to Sinhalese predictions...")
predictions_en_to_si = model.predict(to_predict_en_to_si)

print("Generating Sinhalese to English predictions...")
predictions_si_to_en = model.predict(to_predict_si_to_en)

# --- Calculate BLEU Score ---
# Sacrebleu expects a list of ground truths for each prediction
bleu_en_to_si = sacrebleu.corpus_bleu(predictions_en_to_si, [ground_truth_en_to_si])
bleu_si_to_en = sacrebleu.corpus_bleu(predictions_si_to_en, [ground_truth_si_to_en])

print("\n--- Evaluation Results ---")
print(f"English to Sinhalese BLEU Score: {bleu_en_to_si.score:.2f}")
print(f"Sinhalese to English BLEU Score: {bleu_si_to_en.score:.2f}")
print("--------------------------")

# Save scores to a file for the plotting script
with open("bleu_scores.txt", "w") as f:
    f.write(f"en-si,{bleu_en_to_si.score:.2f}\n")
    f.write(f"si-en,{bleu_si_to_en.score:.2f}\n")
print("Scores saved to bleu_scores.txt")
