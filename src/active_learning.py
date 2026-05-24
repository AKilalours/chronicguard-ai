"""
ChronicGuard AI — Active Learning Loop
Care manager corrections are logged and used to retrain the classifier.
Implements uncertainty sampling: prioritizes examples where the model
was most uncertain for human review and retraining.

Usage:
    from src.active_learning import ActiveLearner
    learner = ActiveLearner(classifier)
    learner.log_correction("message text", pred_intent, true_intent, pred_risk, true_risk)
    if learner.should_retrain():
        new_clf = learner.retrain(existing_train_data)
"""

from __future__ import annotations
import json
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Correction:
    message: str
    predicted_intent: str
    corrected_intent: str
    predicted_risk: str
    corrected_risk: str
    intent_confidence: float
    risk_confidence: float
    timestamp: float
    care_manager_id: str = "cm_001"

    @property
    def intent_corrected(self) -> bool:
        return self.predicted_intent != self.corrected_intent

    @property
    def risk_corrected(self) -> bool:
        return self.predicted_risk != self.corrected_risk

    @property
    def was_safety_correction(self) -> bool:
        """True if a high/urgent was downgraded or low/medium was upgraded to high/urgent."""
        safety_critical = {"high", "urgent"}
        return (
            (self.predicted_risk not in safety_critical and self.corrected_risk in safety_critical) or
            (self.predicted_risk in safety_critical and self.corrected_risk not in safety_critical)
        )

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "predicted_intent": self.predicted_intent,
            "corrected_intent": self.corrected_intent,
            "predicted_risk": self.predicted_risk,
            "corrected_risk": self.corrected_risk,
            "intent_confidence": self.intent_confidence,
            "risk_confidence": self.risk_confidence,
            "intent_corrected": self.intent_corrected,
            "risk_corrected": self.risk_corrected,
            "was_safety_correction": self.was_safety_correction,
            "timestamp": self.timestamp,
            "care_manager_id": self.care_manager_id,
        }


@dataclass
class LearningStats:
    total_corrections: int = 0
    intent_corrections: int = 0
    risk_corrections: int = 0
    safety_corrections: int = 0
    retraining_rounds: int = 0
    last_retrain_timestamp: float = 0.0
    avg_confidence_at_correction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_corrections": self.total_corrections,
            "intent_corrections": self.intent_corrections,
            "risk_corrections": self.risk_corrections,
            "safety_corrections": self.safety_corrections,
            "retraining_rounds": self.retraining_rounds,
            "correction_rate": round(
                self.total_corrections / max(self.total_corrections + 100, 1), 3
            ),
            "avg_confidence_at_correction": round(self.avg_confidence_at_correction, 3),
        }


class ActiveLearner:
    """
    Active learning loop for ChronicGuard AI.

    Strategy: Uncertainty sampling
    - Log all care manager corrections
    - Prioritize low-confidence predictions for human review
    - Retrain classifier when correction buffer reaches threshold
    - Weight safety corrections (risk upgrades/downgrades) more heavily

    In production: corrections come from care manager UI feedback.
    In this prototype: corrections can be simulated or manually provided.
    """

    RETRAIN_THRESHOLD = 20       # retrain after N corrections
    UNCERTAINTY_THRESHOLD = 0.75  # flag predictions below this for review
    SAFETY_WEIGHT = 3.0          # safety corrections count 3x toward retraining

    def __init__(self, classifier=None, log_dir: str = "results/active_learning"):
        self.classifier = classifier
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._corrections: list[Correction] = []
        self._stats = LearningStats()
        self._weighted_correction_count = 0.0
        self._load_existing_corrections()

    def _load_existing_corrections(self):
        path = self.log_dir / "corrections.jsonl"
        if path.exists():
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        self._corrections.append(Correction(**d))
                    except Exception:
                        pass
            self._recompute_stats()

    def log_correction(
        self,
        message: str,
        predicted_intent: str,
        corrected_intent: str,
        predicted_risk: str,
        corrected_risk: str,
        intent_confidence: float = 0.5,
        risk_confidence: float = 0.5,
        care_manager_id: str = "cm_001",
    ) -> Correction:
        """Log a care manager correction."""
        correction = Correction(
            message=message,
            predicted_intent=predicted_intent,
            corrected_intent=corrected_intent,
            predicted_risk=predicted_risk,
            corrected_risk=corrected_risk,
            intent_confidence=intent_confidence,
            risk_confidence=risk_confidence,
            timestamp=time.time(),
            care_manager_id=care_manager_id,
        )
        self._corrections.append(correction)

        # Weighted count — safety corrections count more
        weight = self.SAFETY_WEIGHT if correction.was_safety_correction else 1.0
        self._weighted_correction_count += weight

        # Persist
        with open(self.log_dir / "corrections.jsonl", "a") as f:
            f.write(json.dumps(correction.to_dict()) + "\n")

        self._recompute_stats()
        return correction

    def should_retrain(self) -> bool:
        return self._weighted_correction_count >= self.RETRAIN_THRESHOLD

    def get_uncertain_examples(self, threshold: float | None = None) -> list[Correction]:
        """Return corrections where model was most uncertain — best for retraining."""
        t = threshold or self.UNCERTAINTY_THRESHOLD
        return [
            c for c in self._corrections
            if c.intent_confidence < t or c.risk_confidence < t
        ]

    def retrain(self, base_train_data: list[dict]) -> object:
        """
        Retrain classifier incorporating corrections.
        Corrections are weighted by safety importance.
        """
        if self.classifier is None:
            print("No classifier attached — cannot retrain.")
            return None

        # Build augmented training set
        augmented_texts = [r["message"] for r in base_train_data]
        augmented_intents = [r["intent"] for r in base_train_data]
        augmented_risks = [r["risk_level"] for r in base_train_data]

        # Add corrections (with repetition for safety corrections = more weight)
        for correction in self._corrections:
            weight = int(self.SAFETY_WEIGHT) if correction.was_safety_correction else 1
            for _ in range(weight):
                augmented_texts.append(correction.message)
                augmented_intents.append(correction.corrected_intent)
                augmented_risks.append(correction.corrected_risk)

        print(f"Retraining on {len(base_train_data)} base + {len(self._corrections)} corrections "
              f"({len(augmented_texts)} total examples)")

        # Retrain
        self.classifier.fit(augmented_texts, augmented_intents, augmented_risks)
        self._stats.retraining_rounds += 1
        self._stats.last_retrain_timestamp = time.time()
        self._weighted_correction_count = 0.0  # reset counter

        # Save retrain log
        retrain_log = {
            "round": self._stats.retraining_rounds,
            "base_examples": len(base_train_data),
            "correction_examples": len(self._corrections),
            "total_examples": len(augmented_texts),
            "safety_corrections": sum(1 for c in self._corrections if c.was_safety_correction),
            "timestamp": time.time(),
        }
        with open(self.log_dir / "retrain_log.jsonl", "a") as f:
            f.write(json.dumps(retrain_log) + "\n")

        print(f"Retraining complete. Round {self._stats.retraining_rounds}.")
        return self.classifier

    def _recompute_stats(self):
        self._stats.total_corrections = len(self._corrections)
        self._stats.intent_corrections = sum(1 for c in self._corrections if c.intent_corrected)
        self._stats.risk_corrections = sum(1 for c in self._corrections if c.risk_corrected)
        self._stats.safety_corrections = sum(1 for c in self._corrections if c.was_safety_correction)
        if self._corrections:
            avg_conf = np.mean([
                (c.intent_confidence + c.risk_confidence) / 2
                for c in self._corrections
            ])
            self._stats.avg_confidence_at_correction = float(avg_conf)

    def get_stats(self) -> dict:
        stats = self._stats.to_dict()
        stats["weighted_correction_count"] = round(self._weighted_correction_count, 1)
        stats["retrain_threshold"] = self.RETRAIN_THRESHOLD
        stats["progress_to_retrain"] = round(
            min(1.0, self._weighted_correction_count / self.RETRAIN_THRESHOLD), 3
        )
        stats["should_retrain"] = self.should_retrain()
        stats["uncertain_examples"] = len(self.get_uncertain_examples())
        return stats

    def simulate_corrections(self, n: int = 25) -> list[Correction]:
        """Simulate care manager corrections for demo purposes."""
        import random
        random.seed(42)

        scenarios = [
            ("I have chest pain", "symptom_escalation", "symptom_escalation", "medium", "urgent", 0.55, 0.48),
            ("feeling very down lately", "appointment_admin", "crisis", "low", "high", 0.42, 0.38),
            ("my meds ran out", "medication_question", "care_gap", "low", "medium", 0.61, 0.55),
            ("blood sugar 320", "lab_results", "symptom_escalation", "medium", "high", 0.58, 0.52),
            ("I don't feel like going on", "appointment_admin", "crisis", "low", "urgent", 0.35, 0.30),
            ("didn't take warfarin for 3 days", "medication_question", "care_gap", "medium", "high", 0.64, 0.59),
            ("headache and vision blur", "medication_question", "symptom_escalation", "low", "high", 0.51, 0.44),
        ]

        corrections = []
        for i in range(n):
            scenario = random.choice(scenarios)
            msg, pred_i, true_i, pred_r, true_r, i_conf, r_conf = scenario
            c = self.log_correction(
                message=f"{msg} (patient {i+1})",
                predicted_intent=pred_i,
                corrected_intent=true_i,
                predicted_risk=pred_r,
                corrected_risk=true_r,
                intent_confidence=i_conf + random.uniform(-0.1, 0.1),
                risk_confidence=r_conf + random.uniform(-0.1, 0.1),
                care_manager_id=f"cm_{random.randint(1,3):03d}",
            )
            corrections.append(c)

        return corrections


if __name__ == "__main__":
    learner = ActiveLearner()
    print("Simulating 25 care manager corrections...")
    corrections = learner.simulate_corrections(25)

    stats = learner.get_stats()
    print(f"\nActive Learning Stats:")
    print(f"  Total corrections:    {stats['total_corrections']}")
    print(f"  Risk corrections:     {stats['risk_corrections']}")
    print(f"  Safety corrections:   {stats['safety_corrections']} (weighted {ActiveLearner.SAFETY_WEIGHT}x)")
    print(f"  Weighted count:       {stats['weighted_correction_count']}")
    print(f"  Progress to retrain:  {stats['progress_to_retrain']*100:.0f}%")
    print(f"  Should retrain:       {stats['should_retrain']}")
    print(f"  Uncertain examples:   {stats['uncertain_examples']}")
    print(f"  Avg confidence:       {stats['avg_confidence_at_correction']:.3f}")

    import json
    Path("results").mkdir(exist_ok=True)
    with open("results/active_learning_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved → results/active_learning_stats.json")
