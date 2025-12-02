#!/usr/bin/env python3
"""
Comprehensive evaluation with multiple metrics and detailed analysis
"""

import os
import pandas as pd
import torch
from simpletransformers.t5 import T5Model
import sacrebleu
import logging
import json
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)

DATA_DIR = "data"
EVAL_CSV = os.path.join(DATA_DIR, "combined_translations_validation.csv")
MODEL_PATH = "outputs/mt5-english-tulu"
TASK = ("translate english to tulu", "English", "Tulu", "en-tu")

def _find_col_case_insensitive(df: pd.DataFrame, name: str):
    name_l = name.strip().lower()
    for c in df.columns:
        if isinstance(c, str) and c.strip().lower() == name_l:
            return c
    return None

def calculate_exact_match(predictions, references):
    """Calculate exact match accuracy"""
    exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip() == r.strip())
    return (exact_matches / len(predictions)) * 100

def calculate_character_error_rate(predictions, references):
    """Calculate character-level error rate"""
    total_chars = 0
    total_errors = 0
    
    for pred, ref in zip(predictions, references):
        pred_chars = set(pred)
        ref_chars = set(ref)
        errors = len(ref_chars - pred_chars) + len(pred_chars - ref_chars)
        total_errors += errors
        total_chars += len(ref)
    
    return (total_errors / total_chars) * 100 if total_chars > 0 else 0

def calculate_length_ratio(predictions, references):
    """Calculate average length ratio (predicted/reference)"""
    ratios = [len(p) / len(r) if len(r) > 0 else 0 for p, r in zip(predictions, references)]
    return np.mean(ratios), np.std(ratios)

def analyze_by_length_bucket(predictions, references, inputs):
    """Analyze performance by input length buckets"""
    buckets = {
        'short (0-20)': [],
        'medium (21-50)': [],
        'long (51+)': []
    }
    
    for pred, ref, inp in zip(predictions, references, inputs):
        inp_len = len(inp.split())
        
        if inp_len <= 20:
            buckets['short (0-20)'].append((pred, ref))
        elif inp_len <= 50:
            buckets['medium (21-50)'].append((pred, ref))
        else:
            buckets['long (51+)'].append((pred, ref))
    
    results = {}
    for bucket_name, pairs in buckets.items():
        if pairs:
            preds = [p for p, r in pairs]
            refs = [r for p, r in pairs]
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            exact = calculate_exact_match(preds, refs)
            results[bucket_name] = {
                'count': len(pairs),
                'bleu': bleu.score,
                'exact_match': exact
            }
        else:
            results[bucket_name] = {'count': 0, 'bleu': 0, 'exact_match': 0}
    
    return results

def get_sample_translations(predictions, references, inputs, n=20):
    """Get sample translations with quality scores"""
    samples = []
    
    # Get diverse samples based on BLEU scores
    individual_bleus = []
    for pred, ref in zip(predictions, references):
        try:
            bleu = sacrebleu.sentence_bleu(pred, [ref])
            individual_bleus.append(bleu.score)
        except:
            individual_bleus.append(0)
    
    # Sort by BLEU and get diverse samples
    sorted_indices = np.argsort(individual_bleus)
    
    # Get best, worst, and median samples
    indices = []
    indices.extend(sorted_indices[-5:].tolist())  # Top 5
    indices.extend(sorted_indices[:5].tolist())   # Bottom 5
    indices.extend(sorted_indices[len(sorted_indices)//2-5:len(sorted_indices)//2+5].tolist())  # Middle 10
    
    for idx in indices[:n]:
        samples.append({
            'input': inputs[idx].replace("translate english to tulu: ", ""),
            'prediction': predictions[idx],
            'reference': references[idx],
            'bleu': individual_bleus[idx],
            'exact_match': predictions[idx].strip() == references[idx].strip()
        })
    
    return samples

def main():
    print("="*70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("="*70)
    print()
    
    # Load model
    print("Loading fine-tuned model...")
    use_cuda = torch.cuda.is_available()
    model = T5Model("mt5", MODEL_PATH, use_cuda=use_cuda)
    
    # Load data
    print("Loading evaluation data...")
    eval_df = pd.read_csv(EVAL_CSV)
    
    prefix, source_col, target_col, key = TASK
    src = _find_col_case_insensitive(eval_df, source_col)
    tgt = _find_col_case_insensitive(eval_df, target_col)
    
    if src is None or tgt is None:
        raise ValueError(f"CSV must contain columns '{source_col}' and '{target_col}'")
    
    task_df = eval_df[[src, tgt]].dropna()
    inputs = (prefix + ": " + task_df[src].astype(str).str.strip()).tolist()
    references = task_df[tgt].astype(str).str.strip().tolist()
    
    print(f"Evaluating {len(inputs)} samples...")
    print()
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(inputs)
    
    # Calculate metrics
    print("\n" + "="*70)
    print("METRICS SUMMARY")
    print("="*70)
    
    # BLEU Score
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    print(f"📊 BLEU Score: {bleu.score:.2f}")
    
    # Exact Match Accuracy
    exact_match = calculate_exact_match(predictions, references)
    print(f"🎯 Exact Match Accuracy: {exact_match:.2f}%")
    
    # Character Error Rate
    cer = calculate_character_error_rate(predictions, references)
    print(f"📝 Character Error Rate: {cer:.2f}%")
    
    # Length Ratio
    length_mean, length_std = calculate_length_ratio(predictions, references)
    print(f"📏 Length Ratio (pred/ref): {length_mean:.2f} ± {length_std:.2f}")
    
    # Analysis by length
    print("\n" + "="*70)
    print("PERFORMANCE BY INPUT LENGTH")
    print("="*70)
    
    length_analysis = analyze_by_length_bucket(predictions, references, inputs)
    for bucket, metrics in length_analysis.items():
        print(f"\n{bucket}:")
        print(f"  Samples: {metrics['count']}")
        print(f"  BLEU: {metrics['bleu']:.2f}")
        print(f"  Exact Match: {metrics['exact_match']:.2f}%")
    
    # Get sample translations
    print("\n" + "="*70)
    print("SAMPLE TRANSLATIONS")
    print("="*70)
    
    samples = get_sample_translations(predictions, references, inputs, n=20)
    
    print("\n🏆 Best Translations (High BLEU):")
    for i, sample in enumerate([s for s in samples if s['bleu'] > 15][:5], 1):
        print(f"\n{i}. Input: {sample['input']}")
        print(f"   Predicted: {sample['prediction']}")
        print(f"   Reference: {sample['reference']}")
        print(f"   BLEU: {sample['bleu']:.2f} | Exact Match: {sample['exact_match']}")
    
    print("\n⚠️ Worst Translations (Low BLEU):")
    for i, sample in enumerate([s for s in samples if s['bleu'] < 5][:5], 1):
        print(f"\n{i}. Input: {sample['input']}")
        print(f"   Predicted: {sample['prediction']}")
        print(f"   Reference: {sample['reference']}")
        print(f"   BLEU: {sample['bleu']:.2f} | Exact Match: {sample['exact_match']}")
    
    # Save comprehensive results
    results = {
        'overall_metrics': {
            'bleu_score': float(bleu.score),
            'exact_match_accuracy': float(exact_match),
            'character_error_rate': float(cer),
            'length_ratio_mean': float(length_mean),
            'length_ratio_std': float(length_std),
            'total_samples': len(predictions)
        },
        'length_analysis': length_analysis,
        'sample_translations': samples
    }
    
    # Save to JSON
    with open('comprehensive_evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*70)
    print("✅ Evaluation complete!")
    print("📁 Results saved to: comprehensive_evaluation_results.json")
    print("="*70)
    
    return results

if __name__ == "__main__":
    results = main()
