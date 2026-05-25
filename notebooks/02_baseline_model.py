"""
ChronicGuard AI — Notebook 02: Baseline Model Training & Evaluation
Trains TF-IDF + Logistic Regression baseline, evaluates on test set,
generates per-class metrics, confusion matrix, and saves artifacts.

Run: python notebooks/02_baseline_model.py
"""

import sys, csv, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)

print("=" * 60)
print("NOTEBOOK 02 — BASELINE MODEL TRAINING & EVALUATION")
print("=" * 60)

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("\n[1/5] Loading dataset...")
with open("data/synthetic_messages.csv") as f:
    rows = list(csv.DictReader(f))

texts   = [r["message"]    for r in rows]
intents = [r["intent"]     for r in rows]
risks   = [r["risk_level"] for r in rows]

print(f"  Total messages: {len(rows)}")
print(f"  Intent classes: {sorted(set(intents))}")
print(f"  Risk levels:    {sorted(set(risks))}")

# ── 2. Train/test split ────────────────────────────────────────────────────────
print("\n[2/5] Stratified 80/20 train/test split...")
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(texts, risks))

train_texts   = [texts[i]   for i in train_idx]
test_texts    = [texts[i]   for i in test_idx]
train_intents = [intents[i] for i in train_idx]
test_intents  = [intents[i] for i in test_idx]
train_risks   = [risks[i]   for i in train_idx]
test_risks    = [risks[i]   for i in test_idx]

print(f"  Train: {len(train_texts)} | Test: {len(test_texts)}")
from collections import Counter
risk_dist = Counter(train_risks)
print(f"  Train risk dist: {dict(sorted(risk_dist.items()))}")

# ── 3. Train baseline ──────────────────────────────────────────────────────────
print("\n[3/5] Training TF-IDF + Logistic Regression baseline...")
from src.classifier import TFIDFClassifier
clf = TFIDFClassifier()
clf.fit(train_texts, train_intents, train_risks)
print(f"  Intent features: {clf.intent_pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]}")
print(f"  Risk features:   {clf.risk_pipeline.named_steps['tfidf'].get_feature_names_out().shape[0]}")
print(f"  Intent classes:  {list(clf.intent_pipeline.classes_)}")
print(f"  Risk classes:    {list(clf.risk_pipeline.classes_)}")

# ── 4. Evaluate ────────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating on test set...")
pred_intents = list(clf.intent_pipeline.predict(test_texts))
pred_risks   = list(clf.risk_pipeline.predict(test_texts))
risk_proba   = clf.risk_pipeline.predict_proba(test_texts)
risk_classes = list(clf.risk_pipeline.classes_)

RISK_ORDER = ["low", "medium", "high", "urgent"]

# Overall metrics
risk_f1   = f1_score(test_risks,   pred_risks,   average="macro", zero_division=0)
intent_f1 = f1_score(test_intents, pred_intents, average="macro", zero_division=0)
risk_acc  = accuracy_score(test_risks,   pred_risks)
intent_acc= accuracy_score(test_intents, pred_intents)

print(f"\n  {'Metric':<25} {'Risk':>8} {'Intent':>8}")
print(f"  {'-'*42}")
print(f"  {'Macro F1':<25} {risk_f1:>8.3f} {intent_f1:>8.3f}")
print(f"  {'Accuracy':<25} {risk_acc:>8.3f} {intent_acc:>8.3f}")

# Safety metrics
SAFETY_CRITICAL = {"high", "urgent"}
urgent_true = [1 if r == "urgent" else 0 for r in test_risks]
urgent_pred = [1 if r == "urgent" else 0 for r in pred_risks]
tp = sum(1 for t,p in zip(urgent_true, urgent_pred) if t==1 and p==1)
fn = sum(1 for t,p in zip(urgent_true, urgent_pred) if t==1 and p==0)
urgent_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

critical_true = sum(1 for r in test_risks if r in SAFETY_CRITICAL)
critical_fn   = sum(1 for t,p in zip(test_risks, pred_risks)
                    if t in SAFETY_CRITICAL and p not in SAFETY_CRITICAL)
fn_rate = critical_fn / critical_true if critical_true > 0 else 0.0

print(f"\n  {'Safety Metrics':<25}")
print(f"  {'-'*42}")
print(f"  {'Urgent recall':<25} {urgent_recall:>8.3f}  {'✓ PASS' if urgent_recall >= 0.92 else '✗ FAIL'}")
print(f"  {'Critical FN rate':<25} {fn_rate:>8.3f}")
print(f"  {'Safety constraint':<25} {'MET ✓' if urgent_recall >= 0.92 else 'FAILED ✗':>8}")

# Per-class breakdown
print(f"\n  Risk per-class breakdown:")
risk_report = classification_report(
    test_risks, pred_risks,
    labels=RISK_ORDER, output_dict=True, zero_division=0
)
print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print(f"  {'-'*50}")
for cls in RISK_ORDER:
    if cls in risk_report:
        r = risk_report[cls]
        print(f"  {cls:<12} {r['precision']:>10.3f} {r['recall']:>8.3f} "
              f"{r['f1-score']:>8.3f} {r['support']:>8.0f}")

# Confusion matrix
print(f"\n  Risk confusion matrix (rows=true, cols=pred):")
cm = confusion_matrix(test_risks, pred_risks, labels=RISK_ORDER)
header = "         " + "".join(f"{c:>10}" for c in RISK_ORDER)
print(f"  {header}")
for i, cls in enumerate(RISK_ORDER):
    row = f"  {cls:<8} " + "".join(f"{cm[i][j]:>10}" for j in range(len(RISK_ORDER)))
    print(row)

# ── 5. Save artifacts ──────────────────────────────────────────────────────────
print("\n[5/5] Saving artifacts...")
Path("results").mkdir(exist_ok=True)

results = {
    "model": "TF-IDF + Logistic Regression (baseline)",
    "dataset_size": len(rows),
    "train_size": len(train_texts),
    "test_size": len(test_texts),
    "metrics": {
        "risk_macro_f1": round(risk_f1, 4),
        "intent_macro_f1": round(intent_f1, 4),
        "risk_accuracy": round(risk_acc, 4),
        "intent_accuracy": round(intent_acc, 4),
    },
    "safety": {
        "urgent_recall": round(urgent_recall, 4),
        "critical_fn_rate": round(fn_rate, 4),
        "safety_constraint_met": urgent_recall >= 0.92,
        "constraint": "urgent_recall >= 0.92",
    },
    "risk_per_class": {
        cls: {
            "precision": round(risk_report[cls]["precision"], 4),
            "recall": round(risk_report[cls]["recall"], 4),
            "f1": round(risk_report[cls]["f1-score"], 4),
            "support": int(risk_report[cls]["support"]),
        }
        for cls in RISK_ORDER if cls in risk_report
    },
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_labels": RISK_ORDER,
}

with open("results/baseline_model_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save model
import joblib
joblib.dump(clf, "results/baseline_classifier.joblib")

print(f"  results/baseline_model_results.json")
print(f"  results/baseline_classifier.joblib")

print("\n" + "=" * 60)
print("NOTEBOOK 02 COMPLETE")
print("=" * 60)
print(f"  Risk macro-F1:    {risk_f1:.3f}")
print(f"  Intent macro-F1:  {intent_f1:.3f}")
print(f"  Urgent recall:    {urgent_recall:.3f}  {'✓ Safety constraint met' if urgent_recall >= 0.92 else '✗ Constraint failed'}")
print(f"  Critical FN rate: {fn_rate:.3f}")
print("=" * 60)
print("\nNext: Run notebook 03_ablation_study.py to compare with SBERT and ClinicalBERT")
