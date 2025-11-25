import os
import pandas as pd
from simpletransformers.t5 import T5Model
import sacrebleu

# --- Configuration ---
MODEL_PATH = "outputs/mt5-english-kannada-tulu"
EVAL_CSV = os.path.join("data", "combined_translations_test.csv")

TASKS = (
    ("translate english to kannada", "English", "Kannada", "en-ka"),
    ("translate kannada to tulu", "Kannada", "Tulu", "ka-tu"),
)


def _build_prediction_sets(df: pd.DataFrame):
    evaluation_sets = {}
    for prefix, source_col, target_col, key in TASKS:
        task_df = df[[source_col, target_col]].dropna()
        inputs = (
            prefix + ": " + task_df[source_col].astype(str).str.strip()
        ).tolist()
        labels = task_df[target_col].astype(str).str.strip().tolist()
        evaluation_sets[key] = (inputs, labels)
    return evaluation_sets


print("Loading fine-tuned model and evaluation data...")
model = T5Model("mt5", MODEL_PATH, use_cuda=True)
eval_df = pd.read_csv(EVAL_CSV)
eval_sets = _build_prediction_sets(eval_df)

bleu_scores = {}
for key, (inputs, labels) in eval_sets.items():
    direction_label = "English to Kannada" if key == "en-ka" else "Kannada to Tulu"
    print(f"\nGenerating predictions for {direction_label} ({len(inputs)} samples)...")
    preds = model.predict(inputs)
    bleu_scores[key] = sacrebleu.corpus_bleu(preds, [labels])

print("\n--- Evaluation Results ---")
print(f"English to Kannada BLEU Score: {bleu_scores['en-ka'].score:.2f}")
print(f"Kannada to Tulu BLEU Score: {bleu_scores['ka-tu'].score:.2f}")
print("--------------------------")

with open("bleu_scores.txt", "w") as f:
    f.write(f"en-ka,{bleu_scores['en-ka'].score:.2f}\n")
    f.write(f"ka-tu,{bleu_scores['ka-tu'].score:.2f}\n")
print("Scores saved to bleu_scores.txt")
