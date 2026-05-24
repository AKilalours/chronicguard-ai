"""
ChronicGuard AI — Evaluation Framework
Implements the safety-first evaluation methodology described in research_report.md.

Key principle: In chronic care management, false negatives on urgent/high-risk
messages are categorically worse than false positives. Urgent-class recall is
treated as a hard safety constraint, not a trade-off metric.

Usage:
    from src.evaluation import SafetyEvaluator
    evaluator = SafetyEvaluator()
    report = evaluator.evaluate(y_true_risk, y_pred_risk, y_true_intent, y_pred_intent)
    evaluator.save_report(report, Path("results/"))
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

RISK_ORDER = ["low", "medium", "high", "urgent"]
SAFETY_CRITICAL = {"high", "urgent"}


@dataclass
class SafetyMetrics:
    urgent_recall: float
    urgent_precision: float
    high_recall: float
    high_precision: float
    critical_false_negative_rate: float  # FN rate across high+urgent combined
    safety_constraint_met: bool          # urgent_recall >= 0.92

    def to_dict(self) -> dict:
        return {
            "urgent_recall": round(self.urgent_recall, 4),
            "urgent_precision": round(self.urgent_precision, 4),
            "high_recall": round(self.high_recall, 4),
            "high_precision": round(self.high_precision, 4),
            "critical_false_negative_rate": round(self.critical_false_negative_rate, 4),
            "safety_constraint_met": self.safety_constraint_met,
            "safety_constraint": "urgent_recall >= 0.92",
        }


@dataclass
class CalibrationResult:
    mean_predicted_proba: list[float]
    fraction_positives: list[float]
    brier_score: float
    n_bins: int = 10

    def to_dict(self) -> dict:
        return {
            "mean_predicted_proba": [round(x, 4) for x in self.mean_predicted_proba],
            "fraction_positives": [round(x, 4) for x in self.fraction_positives],
            "brier_score": round(self.brier_score, 4),
            "n_bins": self.n_bins,
        }


@dataclass
class EvaluationReport:
    # Classification performance
    risk_macro_f1: float
    risk_accuracy: float
    intent_macro_f1: float
    intent_accuracy: float

    # Per-class breakdown
    risk_per_class: dict = field(default_factory=dict)
    intent_per_class: dict = field(default_factory=dict)

    # Safety metrics
    safety: SafetyMetrics | None = None

    # Confusion matrices
    risk_confusion: list[list[int]] = field(default_factory=list)
    intent_confusion: list[list[int]] = field(default_factory=list)

    # Calibration
    calibration: CalibrationResult | None = None

    # Error analysis
    error_analysis: dict = field(default_factory=dict)

    # Latency
    latency_p50_ms: float = 0.0
    latency_p95_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "performance": {
                "risk_macro_f1": round(self.risk_macro_f1, 4),
                "risk_accuracy": round(self.risk_accuracy, 4),
                "intent_macro_f1": round(self.intent_macro_f1, 4),
                "intent_accuracy": round(self.intent_accuracy, 4),
            },
            "risk_per_class": self.risk_per_class,
            "intent_per_class": self.intent_per_class,
            "safety": self.safety.to_dict() if self.safety else None,
            "risk_confusion": self.risk_confusion,
            "intent_confusion": self.intent_confusion,
            "calibration": self.calibration.to_dict() if self.calibration else None,
            "error_analysis": self.error_analysis,
            "latency": {
                "p50_ms": round(self.latency_p50_ms, 1),
                "p95_ms": round(self.latency_p95_ms, 1),
            },
        }


class SafetyEvaluator:
    """
    Safety-first evaluation for chronic care triage.
    Treats urgent-class recall as a hard safety constraint.
    """

    URGENT_RECALL_THRESHOLD = 0.92

    def evaluate(
        self,
        y_true_risk: list[str],
        y_pred_risk: list[str],
        y_true_intent: list[str],
        y_pred_intent: list[str],
        risk_proba: list[list[float]] | None = None,
        risk_classes: list[str] | None = None,
        latencies_ms: list[float] | None = None,
        messages: list[str] | None = None,
    ) -> EvaluationReport:

        # ── Risk classification ──────────────────────────────────────────────
        risk_labels = sorted(set(y_true_risk), key=lambda x: RISK_ORDER.index(x) if x in RISK_ORDER else 99)
        risk_f1 = f1_score(y_true_risk, y_pred_risk, average="macro", labels=risk_labels, zero_division=0)
        risk_acc = accuracy_score(y_true_risk, y_pred_risk)
        risk_report = classification_report(y_true_risk, y_pred_risk, labels=risk_labels,
                                             output_dict=True, zero_division=0)
        risk_cm = confusion_matrix(y_true_risk, y_pred_risk, labels=risk_labels).tolist()

        # ── Intent classification ────────────────────────────────────────────
        intent_labels = sorted(set(y_true_intent))
        intent_f1 = f1_score(y_true_intent, y_pred_intent, average="macro",
                              labels=intent_labels, zero_division=0)
        intent_acc = accuracy_score(y_true_intent, y_pred_intent)
        intent_report = classification_report(y_true_intent, y_pred_intent,
                                               labels=intent_labels, output_dict=True, zero_division=0)
        intent_cm = confusion_matrix(y_true_intent, y_pred_intent, labels=intent_labels).tolist()

        # ── Safety metrics ───────────────────────────────────────────────────
        safety = self._compute_safety_metrics(y_true_risk, y_pred_risk, risk_labels)

        # ── Calibration ──────────────────────────────────────────────────────
        calibration = None
        if risk_proba is not None and risk_classes is not None:
            calibration = self._compute_calibration(y_true_risk, risk_proba, risk_classes)

        # ── Error analysis ───────────────────────────────────────────────────
        error_analysis = self._error_analysis(
            y_true_risk, y_pred_risk, y_true_intent, y_pred_intent,
            messages, risk_labels
        )

        # ── Latency ──────────────────────────────────────────────────────────
        p50, p95 = 0.0, 0.0
        if latencies_ms:
            p50 = float(np.percentile(latencies_ms, 50))
            p95 = float(np.percentile(latencies_ms, 95))

        # Per-class summaries
        risk_per_class = {
            label: {
                "precision": round(risk_report[label]["precision"], 4),
                "recall": round(risk_report[label]["recall"], 4),
                "f1": round(risk_report[label]["f1-score"], 4),
                "support": risk_report[label]["support"],
            }
            for label in risk_labels if label in risk_report
        }
        intent_per_class = {
            label: {
                "precision": round(intent_report[label]["precision"], 4),
                "recall": round(intent_report[label]["recall"], 4),
                "f1": round(intent_report[label]["f1-score"], 4),
                "support": intent_report[label]["support"],
            }
            for label in intent_labels if label in intent_report
        }

        return EvaluationReport(
            risk_macro_f1=risk_f1,
            risk_accuracy=risk_acc,
            intent_macro_f1=intent_f1,
            intent_accuracy=intent_acc,
            risk_per_class=risk_per_class,
            intent_per_class=intent_per_class,
            safety=safety,
            risk_confusion=risk_cm,
            intent_confusion=intent_cm,
            calibration=calibration,
            error_analysis=error_analysis,
            latency_p50_ms=p50,
            latency_p95_ms=p95,
        )

    def _compute_safety_metrics(
        self,
        y_true: list[str],
        y_pred: list[str],
        labels: list[str],
    ) -> SafetyMetrics:
        def safe_recall(label):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
            return tp / (tp + fn) if (tp + fn) > 0 else 0.0

        def safe_precision(label):
            tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
            return tp / (tp + fp) if (tp + fp) > 0 else 0.0

        urgent_recall = safe_recall("urgent") if "urgent" in labels else 0.0
        urgent_precision = safe_precision("urgent") if "urgent" in labels else 0.0
        high_recall = safe_recall("high") if "high" in labels else 0.0
        high_precision = safe_precision("high") if "high" in labels else 0.0

        # Critical FN rate: across high + urgent combined
        critical_true = sum(1 for t in y_true if t in SAFETY_CRITICAL)
        critical_fn = sum(1 for t, p in zip(y_true, y_pred)
                          if t in SAFETY_CRITICAL and p not in SAFETY_CRITICAL)
        fn_rate = critical_fn / critical_true if critical_true > 0 else 0.0

        return SafetyMetrics(
            urgent_recall=urgent_recall,
            urgent_precision=urgent_precision,
            high_recall=high_recall,
            high_precision=high_precision,
            critical_false_negative_rate=fn_rate,
            safety_constraint_met=urgent_recall >= self.URGENT_RECALL_THRESHOLD,
        )

    def _compute_calibration(
        self,
        y_true: list[str],
        risk_proba: list[list[float]],
        risk_classes: list[str],
    ) -> CalibrationResult:
        # Evaluate calibration for the "urgent" class (binary: urgent vs not)
        if "urgent" not in risk_classes:
            return None
        urgent_idx = risk_classes.index("urgent")
        y_binary = [1 if y == "urgent" else 0 for y in y_true]
        y_prob = [p[urgent_idx] for p in risk_proba]

        if sum(y_binary) == 0:
            return None

        frac_pos, mean_pred = calibration_curve(y_binary, y_prob, n_bins=5, strategy="quantile")
        brier = brier_score_loss(y_binary, y_prob)

        return CalibrationResult(
            mean_predicted_proba=mean_pred.tolist(),
            fraction_positives=frac_pos.tolist(),
            brier_score=brier,
            n_bins=5,
        )

    def _error_analysis(
        self,
        y_true_risk: list[str],
        y_pred_risk: list[str],
        y_true_intent: list[str],
        y_pred_intent: list[str],
        messages: list[str] | None,
        risk_labels: list[str],
    ) -> dict:
        # Most dangerous errors: predicted low/medium when actually high/urgent
        dangerous_errors = []
        common_risk_errors = defaultdict(int)
        common_intent_errors = defaultdict(int)

        for i, (tr, pr, ti, pi) in enumerate(
            zip(y_true_risk, y_pred_risk, y_true_intent, y_pred_intent)
        ):
            if tr != pr:
                key = f"{tr} → {pr}"
                common_risk_errors[key] += 1

                if tr in SAFETY_CRITICAL and pr not in SAFETY_CRITICAL:
                    dangerous_errors.append({
                        "true_risk": tr,
                        "predicted_risk": pr,
                        "true_intent": ti,
                        "predicted_intent": pi,
                        "message": messages[i][:120] + "..." if messages and len(messages[i]) > 120 else (messages[i] if messages else ""),
                    })

            if ti != pi:
                key = f"{ti} → {pi}"
                common_intent_errors[key] += 1

        return {
            "dangerous_misclassifications": dangerous_errors[:10],  # top 10
            "common_risk_errors": dict(sorted(common_risk_errors.items(),
                                               key=lambda x: -x[1])[:10]),
            "common_intent_errors": dict(sorted(common_intent_errors.items(),
                                                 key=lambda x: -x[1])[:10]),
            "total_risk_errors": sum(1 for t, p in zip(y_true_risk, y_pred_risk) if t != p),
            "total_intent_errors": sum(1 for t, p in zip(y_true_intent, y_pred_intent) if t != p),
            "safety_critical_fn_count": len(dangerous_errors),
        }

    def save_report(self, report: EvaluationReport, output_dir: Path):
        output_dir.mkdir(parents=True, exist_ok=True)
        report_dict = report.to_dict()

        # Save JSON
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(report_dict, f, indent=2)

        # Save human-readable error analysis
        with open(output_dir / "error_analysis.md", "w") as f:
            self._write_error_analysis_md(f, report)

        print(f"Evaluation report saved to {output_dir}")
        self._print_summary(report)

    def _print_summary(self, report: EvaluationReport):
        print("\n" + "=" * 55)
        print("CHRONICGUARD AI — EVALUATION SUMMARY")
        print("=" * 55)
        print(f"Risk macro-F1:    {report.risk_macro_f1:.3f}")
        print(f"Risk accuracy:    {report.risk_accuracy:.3f}")
        print(f"Intent macro-F1:  {report.intent_macro_f1:.3f}")
        print(f"Intent accuracy:  {report.intent_accuracy:.3f}")
        if report.safety:
            print(f"\n── SAFETY METRICS ──────────────────────────────────")
            print(f"Urgent recall:    {report.safety.urgent_recall:.3f}  {'✓ PASS' if report.safety.safety_constraint_met else '✗ FAIL'}")
            print(f"High recall:      {report.safety.high_recall:.3f}")
            print(f"Critical FN rate: {report.safety.critical_false_negative_rate:.3f}")
            print(f"Safety constraint met: {report.safety.safety_constraint_met}")
        if report.latency_p95_ms > 0:
            print(f"\n── LATENCY ─────────────────────────────────────────")
            print(f"P50: {report.latency_p50_ms:.0f}ms   P95: {report.latency_p95_ms:.0f}ms")
        print("=" * 55)

    def _write_error_analysis_md(self, f, report: EvaluationReport):
        ea = report.error_analysis
        f.write("# Error Analysis — ChronicGuard AI\n\n")
        f.write("## Overview\n\n")
        f.write(f"- Total risk classification errors: {ea.get('total_risk_errors', 0)}\n")
        f.write(f"- Total intent classification errors: {ea.get('total_intent_errors', 0)}\n")
        f.write(f"- Safety-critical false negatives (high/urgent predicted as low/medium): {ea.get('safety_critical_fn_count', 0)}\n\n")

        f.write("## Safety-Critical Misclassifications\n\n")
        f.write("These are the most dangerous errors: cases where the model predicted low or medium risk\n")
        f.write("for a message that was actually high or urgent. In a production system, these could\n")
        f.write("result in delayed care or missed escalation.\n\n")

        for err in ea.get("dangerous_misclassifications", []):
            f.write(f"- **True: {err['true_risk']} / Predicted: {err['predicted_risk']}**\n")
            f.write(f"  Intent: {err['true_intent']} → {err['predicted_intent']}\n")
            if err.get("message"):
                f.write(f"  Message: \"{err['message']}\"\n\n")

        f.write("## Most Common Risk Errors\n\n")
        for pattern, count in ea.get("common_risk_errors", {}).items():
            f.write(f"- `{pattern}`: {count} occurrences\n")

        f.write("\n## Most Common Intent Errors\n\n")
        for pattern, count in ea.get("common_intent_errors", {}).items():
            f.write(f"- `{pattern}`: {count} occurrences\n")

        f.write("\n## Safety Implications\n\n")
        f.write("The primary concern in this system is the downward misclassification of urgent "
                "and high-risk messages. Upward misclassification (predicting high when actually low) "
                "results in unnecessary care manager workload but does not create patient safety risk. "
                "The model is tuned to minimize downward errors at the cost of precision on the "
                "low and medium classes.\n")
