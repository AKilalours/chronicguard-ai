"""
ChronicGuard AI — Classifier Module
Implements three classifiers:
  1. TF-IDF + Logistic Regression (baseline)
  2. SentenceTransformer embeddings + Logistic Regression
  3. ClinicalBERT fine-tuned risk triage (load separately via train_bert.py)

Usage:
    from src.classifier import TriageClassifier
    clf = TriageClassifier()
    clf.fit(texts, intent_labels, risk_labels)
    result = clf.predict("I have chest pain and shortness of breath")
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from pathlib import Path
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


RISK_ORDER = ["low", "medium", "high", "urgent"]
SAFETY_CRITICAL_INTENTS = {"crisis", "symptom_escalation"}


@dataclass
class TriagePrediction:
    intent: str
    intent_confidence: float
    risk_level: str
    risk_confidence: float
    requires_human_review: bool
    safety_flag: bool
    rationale: str

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "intent_confidence": round(self.intent_confidence, 3),
            "risk_level": self.risk_level,
            "risk_confidence": round(self.risk_confidence, 3),
            "requires_human_review": self.requires_human_review,
            "safety_flag": self.safety_flag,
            "rationale": self.rationale,
        }


class TFIDFClassifier:
    """Baseline classifier using TF-IDF + Logistic Regression."""

    def __init__(self):
        self.intent_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                
                
            )),
        ])
        self.risk_pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=8000,
                sublinear_tf=True,
                min_df=1,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                
                
            )),
        ])
        self.intent_encoder = LabelEncoder()
        self.risk_encoder = LabelEncoder()
        self.fitted = False

    def fit(self, texts: list[str], intents: list[str], risks: list[str]) -> "TFIDFClassifier":
        self.intent_encoder.fit(intents)
        self.risk_encoder.fit(risks)
        self.intent_pipeline.fit(texts, intents)
        self.risk_pipeline.fit(texts, risks)
        self.fitted = True
        return self

    def predict_proba(self, text: str) -> tuple[dict, dict]:
        intent_proba = self.intent_pipeline.predict_proba([text])[0]
        risk_proba = self.risk_pipeline.predict_proba([text])[0]
        intent_classes = self.intent_pipeline.classes_
        risk_classes = self.risk_pipeline.classes_
        return (
            dict(zip(intent_classes, intent_proba)),
            dict(zip(risk_classes, risk_proba)),
        )

    def evaluate(self, texts: list[str], intents: list[str], risks: list[str]) -> dict:
        intent_preds = self.intent_pipeline.predict(texts)
        risk_preds = self.risk_pipeline.predict(texts)
        return {
            "intent_report": classification_report(intents, intent_preds, output_dict=True),
            "risk_report": classification_report(risks, risk_preds, output_dict=True),
            "intent_confusion": confusion_matrix(intents, intent_preds,
                                                  labels=sorted(set(intents))).tolist(),
            "risk_confusion": confusion_matrix(risks, risk_preds,
                                                labels=RISK_ORDER).tolist(),
        }

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFClassifier":
        return joblib.load(path)


class SBERTClassifier:
    """SentenceTransformer embeddings + Logistic Regression classifier."""

    MODEL_NAME = "all-mpnet-base-v2"

    def __init__(self):
        if not HAS_SBERT:
            raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        self.encoder = SentenceTransformer(self.MODEL_NAME)
        self.intent_clf = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
             
        )
        self.risk_clf = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
             
        )
        self.intent_classes_: list[str] = []
        self.risk_classes_: list[str] = []
        self.fitted = False

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def fit(self, texts: list[str], intents: list[str], risks: list[str]) -> "SBERTClassifier":
        print("Encoding training data with SBERT...")
        embeddings = self._embed(texts)
        self.intent_clf.fit(embeddings, intents)
        self.risk_clf.fit(embeddings, risks)
        self.intent_classes_ = list(self.intent_clf.classes_)
        self.risk_classes_ = list(self.risk_clf.classes_)
        self.fitted = True
        return self

    def predict_proba(self, text: str) -> tuple[dict, dict]:
        emb = self._embed([text])
        intent_proba = self.intent_clf.predict_proba(emb)[0]
        risk_proba = self.risk_clf.predict_proba(emb)[0]
        return (
            dict(zip(self.intent_classes_, intent_proba)),
            dict(zip(self.risk_classes_, risk_proba)),
        )

    def evaluate(self, texts: list[str], intents: list[str], risks: list[str]) -> dict:
        embeddings = self._embed(texts)
        intent_preds = self.intent_clf.predict(embeddings)
        risk_preds = self.risk_clf.predict(embeddings)
        return {
            "intent_report": classification_report(intents, intent_preds, output_dict=True),
            "risk_report": classification_report(risks, risk_preds, output_dict=True),
            "intent_confusion": confusion_matrix(intents, intent_preds,
                                                  labels=sorted(set(intents))).tolist(),
            "risk_confusion": confusion_matrix(risks, risk_preds,
                                                labels=RISK_ORDER).tolist(),
        }

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "SBERTClassifier":
        return joblib.load(path)


class TriageClassifier:
    """
    Orchestrates intent + risk prediction with safety rules.
    Uses SBERT if available, falls back to TF-IDF.
    """

    HUMAN_REVIEW_THRESHOLD = 0.75
    URGENT_INTENTS = {"crisis"}

    def __init__(self, use_sbert: bool = True):
        if use_sbert and HAS_SBERT:
            self._clf = SBERTClassifier()
            self.model_name = "sbert"
        else:
            self._clf = TFIDFClassifier()
            self.model_name = "tfidf"

    def fit(self, texts: list[str], intents: list[str], risks: list[str]) -> "TriageClassifier":
        self._clf.fit(texts, intents, risks)
        return self

    def predict(self, text: str) -> TriagePrediction:
        intent_proba, risk_proba = self._clf.predict_proba(text)

        intent = max(intent_proba, key=intent_proba.get)
        intent_conf = float(intent_proba[intent])

        # Safety override: if crisis signals are present, bump risk
        risk = max(risk_proba, key=risk_proba.get)
        risk_conf = float(risk_proba[risk])

        # Hard safety rules
        safety_flag = False
        if intent in self.URGENT_INTENTS:
            # Crisis always flags for human review regardless of confidence
            safety_flag = True
            if RISK_ORDER.index(risk) < RISK_ORDER.index("high"):
                risk = "high"
                risk_conf = max(risk_conf, 0.6)

        requires_review = (
            risk in ("high", "urgent")
            or intent_conf < self.HUMAN_REVIEW_THRESHOLD
            or risk_conf < self.HUMAN_REVIEW_THRESHOLD
            or safety_flag
        )

        rationale = (
            f"Intent: {intent} (p={intent_conf:.2f}). "
            f"Risk: {risk} (p={risk_conf:.2f}). "
            + ("Safety flag active. " if safety_flag else "")
            + ("Human review required due to low confidence. " if requires_review and not safety_flag else "")
        )

        return TriagePrediction(
            intent=intent,
            intent_confidence=intent_conf,
            risk_level=risk,
            risk_confidence=risk_conf,
            requires_human_review=requires_review,
            safety_flag=safety_flag,
            rationale=rationale,
        )

    def evaluate(self, texts: list[str], intents: list[str], risks: list[str]) -> dict:
        return self._clf.evaluate(texts, intents, risks)

    def save(self, path: Path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TriageClassifier":
        return joblib.load(path)
