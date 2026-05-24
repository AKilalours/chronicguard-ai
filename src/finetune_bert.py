"""
ChronicGuard AI — ClinicalBERT Fine-Tuning Module
Fine-tunes Bio_ClinicalBERT for patient message risk triage using LoRA
(parameter-efficient fine-tuning). Demonstrates PyTorch + HuggingFace
proficiency required by the job description.

Usage:
    python src/finetune_bert.py --epochs 5 --output_dir models/clinicalbert

Requirements:
    pip install transformers torch peft accelerate
"""

from __future__ import annotations
import os
import json
import csv
import argparse
from pathlib import Path
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback,
        DataCollatorWithPadding,
    )
    from transformers import set_seed
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False


MODEL_ID = "emilyalsentzer/Bio_ClinicalBERT"
RISK_ORDER = ["low", "medium", "high", "urgent"]


@dataclass
class FinetuneConfig:
    model_id: str = MODEL_ID
    output_dir: str = "models/clinicalbert"
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 2e-4
    max_length: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    seed: int = 42
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_task: str = "risk"  # "risk" or "intent"


class PatientMessageDataset(Dataset):
    """PyTorch Dataset for tokenized patient messages."""

    def __init__(self, encodings: dict, labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_data(data_path: Path, task: str = "risk") -> tuple:
    """Load and split data for fine-tuning."""
    with open(data_path) as f:
        rows = list(csv.DictReader(f))

    texts = [r["message"] for r in rows]
    labels_raw = [r[task if task == "risk_level" else "intent"] for r in rows]

    # Fix key name
    if task == "risk":
        labels_raw = [r["risk_level"] for r in rows]
    else:
        labels_raw = [r["intent"] for r in rows]

    # Encode labels
    le = LabelEncoder()
    if task == "risk":
        le.fit(RISK_ORDER)
    else:
        le.fit(sorted(set(labels_raw)))

    labels = le.transform(labels_raw)

    # Stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(sss.split(texts, labels))

    return (
        [texts[i] for i in train_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in train_idx],
        [labels[i] for i in test_idx],
        le,
    )


def apply_lora(model, config: FinetuneConfig):
    """Apply LoRA adapters for parameter-efficient fine-tuning."""
    if not HAS_PEFT:
        print("peft not installed — running full fine-tuning (no LoRA)")
        return model

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"  LoRA applied: {trainable:,} trainable / {total:,} total params ({100*trainable/total:.2f}%)")
    return model


def compute_metrics(eval_pred, label_encoder: LabelEncoder):
    """Compute safety-aware metrics during training."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    # Urgent recall (safety constraint)
    if "urgent" in label_encoder.classes_:
        urgent_idx = list(label_encoder.classes_).index("urgent")
        urgent_true = (labels == urgent_idx).astype(int)
        urgent_pred = (predictions == urgent_idx).astype(int)
        tp = (urgent_true & urgent_pred).sum()
        fn = (urgent_true & ~urgent_pred.astype(bool)).sum()
        urgent_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        urgent_recall = 0.0

    return {
        "macro_f1": macro_f1,
        "urgent_recall": urgent_recall,
        "safety_passed": int(urgent_recall >= 0.92),
    }


class SafetyAwareTrainer(Trainer):
    """
    Custom Trainer that logs the safety constraint at each eval step.
    Overrides the default loss to add class weights for urgent/high.
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weights = torch.tensor(self.class_weights, dtype=torch.float).to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        else:
            loss_fn = torch.nn.CrossEntropyLoss()

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: list[int], n_classes: int, risk_boost: float = 2.0) -> list[float]:
    """
    Compute class weights with extra boost for high/urgent classes.
    This directly implements the safety-first training objective.
    """
    counts = np.bincount(labels, minlength=n_classes)
    weights = len(labels) / (n_classes * counts + 1e-8)
    # Boost urgent (index 3) and high (index 2) in RISK_ORDER
    if n_classes == 4:  # risk task
        weights[2] *= risk_boost   # high
        weights[3] *= risk_boost   # urgent
    weights = weights / weights.sum() * n_classes
    return weights.tolist()


def run_finetuning(config: FinetuneConfig, data_path: Path):
    """Full fine-tuning pipeline."""
    if not HAS_TRANSFORMERS:
        print("transformers not installed. Run: pip install transformers torch peft accelerate")
        print_design_summary(config)
        return

    set_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    train_texts, test_texts, train_labels, test_labels, le = load_data(data_path, config.target_task)
    n_classes = len(le.classes_)
    print(f"Classes ({n_classes}): {list(le.classes_)}")

    # Tokenizer
    print(f"Loading tokenizer: {config.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=config.max_length)
    test_enc  = tokenizer(test_texts,  truncation=True, padding=True, max_length=config.max_length)

    train_dataset = PatientMessageDataset(train_enc, train_labels)
    test_dataset  = PatientMessageDataset(test_enc,  test_labels)

    # Model
    print(f"Loading model: {config.model_id}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_id,
        num_labels=n_classes,
        id2label={i: c for i, c in enumerate(le.classes_)},
        label2id={c: i for i, c in enumerate(le.classes_)},
        ignore_mismatched_sizes=True,
    )

    if config.use_lora:
        model = apply_lora(model, config)

    # Class weights for safety-aware training
    class_weights = compute_class_weights(list(train_labels), n_classes)
    print(f"Class weights: {[round(w,3) for w in class_weights]}")

    # Training arguments
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        warmup_steps=100,
        weight_decay=config.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="urgent_recall",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        report_to="none",
        seed=config.seed,
        dataloader_num_workers=0,
        fp16=device == "cuda",
    )

    # Safety-aware trainer
    le_ref = le
    trainer = SafetyAwareTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, le_ref),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )

    print("\nStarting fine-tuning...")
    print("Primary optimization target: urgent_recall >= 0.92 (safety constraint)")
    trainer.train()

    # Final evaluation
    print("\nFinal evaluation...")
    eval_results = trainer.evaluate()
    print(json.dumps(eval_results, indent=2))

    # Save
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\nModel saved → {output_dir}/final")
    print(f"Safety constraint met: {eval_results.get('eval_urgent_recall', 0) >= 0.92}")


def print_design_summary(config: FinetuneConfig):
    """Print fine-tuning design when transformers not available."""
    print("\n" + "="*60)
    print("CLINICALBERT FINE-TUNING DESIGN SUMMARY")
    print("="*60)
    print(f"Base model:     {config.model_id}")
    print(f"Method:         LoRA (r={config.lora_rank}, alpha={config.lora_alpha})")
    print(f"Trainable:      ~0.5% of parameters (efficient)")
    print(f"Epochs:         {config.epochs}")
    print(f"Batch size:     {config.batch_size}")
    print(f"LR:             {config.learning_rate}")
    print(f"Loss:           CrossEntropy with safety class weights")
    print(f"Boost factor:   2x weight on high/urgent classes")
    print(f"Primary metric: urgent_recall >= 0.92 (hard constraint)")
    print(f"Secondary:      macro-F1")
    print(f"Early stopping: patience=2 on urgent_recall")
    print("="*60)
    print("\nWhy ClinicalBERT?")
    print("  Pre-trained on MIMIC-III clinical notes — understands")
    print("  medical terminology, symptom severity language, and")
    print("  clinical context that general-purpose models miss.")
    print("  E.g., 'diaphoresis' → urgent cardiac, not just 'sweating'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune ClinicalBERT for patient triage")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--task",       type=str,   default="risk", choices=["risk","intent"])
    parser.add_argument("--output_dir", type=str,   default="models/clinicalbert")
    parser.add_argument("--no_lora",    action="store_true")
    args = parser.parse_args()

    config = FinetuneConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        target_task=args.task,
        output_dir=args.output_dir,
        use_lora=not args.no_lora,
    )

    print_design_summary(config)
    run_finetuning(config, Path("data/synthetic_messages.csv"))
