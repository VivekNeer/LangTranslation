import os
import pandas as pd
import torch
from simpletransformers.t5 import T5Model
import sacrebleu
import logging

logging.basicConfig(level=logging.INFO)
DATA_DIR = "data"
EVAL_CSV = os.path.join(DATA_DIR, "combined_translations_test.csv")

# Evaluate ONLY English -> Tulu
TASK = ("translate english to tulu", "English", "Tulu", "en-tu")
MODEL_PATH = "outputs/mt5-english-tulu"

def _find_col_case_insensitive(df: pd.DataFrame, name: str):
    name_l = name.strip().lower()
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() == name_l:
            return c
    return None

def _build_prediction_set(df: pd.DataFrame):
    prefix, source_col, target_col, key = TASK
    src = _find_col_case_insensitive(df, source_col)
    tgt = _find_col_case_insensitive(df, target_col)
    if src is None or tgt is None:
        raise ValueError(f"Evaluation CSV must contain columns '{source_col}' and '{target_col}'")
    task_df = df[[src, tgt]].dropna()
    inputs = (prefix + ": " + task_df[src].astype(str).str.strip()).tolist()
    labels = task_df[tgt].astype(str).str.strip().tolist()
    return key, (inputs, labels)

print("Loading fine-tuned model and evaluation data...")
use_cuda = torch.cuda.is_available()
if not use_cuda:
    logging.info("CUDA not available -- evaluation will run on CPU (use_cuda=False).")
else:
    logging.info("CUDA is available -- evaluation will run on GPU (use_cuda=True).")

model = T5Model("mt5", MODEL_PATH, use_cuda=use_cuda)

eval_df = pd.read_csv(EVAL_CSV)
key, (inputs, labels) = _build_prediction_set(eval_df)

print(f"\nGenerating predictions for English -> Tulu ({len(inputs)} samples)...")
preds = model.predict(inputs)

# sacrebleu expects list-of-lists for references
bleu = sacrebleu.corpus_bleu(preds, [labels])
print("\n--- Evaluation Results ---")
print(f"English -> Tulu BLEU Score: {bleu.score:.2f}")
print("--------------------------")

with open("bleu_scores.txt", "w") as f:
    f.write(f"{key},{bleu.score:.2f}\n")
print("Scores saved to bleu_scores.txt")
