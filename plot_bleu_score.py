import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---
DIRECTIONS = {
    "en-ka": "English to Kannada",
    "ka-tu": "Kannada to Tulu",
}

# Optional benchmark BLEU scores; set to None if not available
BENCHMARK_SCORES = {
    "English to Kannada": None,
    "Kannada to Tulu": None,
}

# --- Plot Styling ---
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 6)

# --- Load Your Scores ---
try:
    with open("bleu_scores.txt", "r") as f:
        scores = {line.split(',')[0]: float(line.split(',')[1]) for line in f}
except FileNotFoundError:
    print("Error: bleu_scores.txt not found. Please run evaluate_model.py first.")
    raise SystemExit(1)

rows = []
for key, label in DIRECTIONS.items():
    model_score = scores.get(key)
    if model_score is None:
        raise ValueError(f"Missing BLEU score for key '{key}' in bleu_scores.txt")
    rows.append({
        "Translation Direction": label,
        "Model": "Our Fine-Tuned mT5",
        "BLEU Score": model_score,
    })

    benchmark_score = BENCHMARK_SCORES.get(label)
    if benchmark_score is not None:
        rows.append({
            "Translation Direction": label,
            "Model": "Benchmark",
            "BLEU Score": benchmark_score,
        })

df = pd.DataFrame(rows)

# --- Generate Bar Chart ---
print("Generating BLEU Score comparison chart...")
plt.figure()
ax = sns.barplot(data=df, x="Translation Direction", y="BLEU Score", hue="Model", palette="viridis")
plt.title("Translation Performance (BLEU Score)", fontsize=16)
plt.ylabel("BLEU Score (Higher is Better)")
plt.xlabel("")
plt.ylim(0, max(df["BLEU Score"]) * 1.2)

for p in ax.patches:
    ax.annotate(
        f"{p.get_height():.1f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        fontsize=11,
        color="black",
        xytext=(0, 5),
        textcoords="offset points",
    )

plt.tight_layout()
plt.savefig("bleu_score_comparison.png", dpi=300)
print("Saved bleu_score_comparison.png")
