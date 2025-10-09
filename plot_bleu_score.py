import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# --- Configuration ---
# Benchmark scores from the Tatoeba Challenge for comparison
BENCHMARK_SCORES = {
    "English to Sinhalese": 9.3,
    "Sinhalese to English": 23.3,
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
    exit()

# --- Create DataFrame for Plotting ---
data = {
    "Translation Direction": ["English to Sinhalese", "Sinhalese to English", "English to Sinhalese", "Sinhalese to English"],
    "Model": ["Benchmark", "Benchmark", "Our Fine-Tuned mT5", "Our Fine-Tuned mT5"],
    "BLEU Score": [
        BENCHMARK_SCORES["English to Sinhalese"],
        BENCHMARK_SCORES["Sinhalese to English"],
        scores["en-si"],
        scores["si-en"]
    ]
}
df = pd.DataFrame(data)

# --- Generate Bar Chart ---
print("Generating BLEU Score comparison chart...")
plt.figure()
ax = sns.barplot(data=df, x="Translation Direction", y="BLEU Score", hue="Model", palette="viridis")
plt.title("Translation Performance (BLEU Score)", fontsize=16)
plt.ylabel("BLEU Score (Higher is Better)")
plt.xlabel("")
plt.ylim(0, max(df["BLEU Score"]) * 1.2) # Set y-axis limit

# Add score labels on top of bars
for p in ax.patches:
    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=11, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.tight_layout()
plt.savefig("bleu_score_comparison.png", dpi=300)
print("Saved bleu_score_comparison.png")
