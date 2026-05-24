"""
ChronicGuard AI — Main Pipeline Orchestrator
Runs the full triage pipeline: classify → retrieve → draft → evaluate

Usage:
    from src.pipeline import ChronicGuardPipeline
    pipeline = ChronicGuardPipeline()
    pipeline.setup()
    result = pipeline.run("I have chest pain and shortness of breath")
    print(result)
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.classifier import TriageClassifier
from src.retriever import CareRetriever
from src.llm_response import ResponseDrafter


@dataclass
class PipelineResult:
    message: str

    # Triage
    intent: str = ""
    intent_confidence: float = 0.0
    risk_level: str = ""
    risk_confidence: float = 0.0
    safety_flag: bool = False

    # Retrieval
    retrieved_protocols: list[dict] = field(default_factory=list)
    context_window: str = ""

    # Generation
    draft_response: str = ""
    recommended_action: str = ""
    escalation_needed: bool = False
    llm_confidence: float = 0.0
    safety_notes: str = ""

    # HITL
    requires_human_review: bool = True

    # Performance
    total_latency_ms: float = 0.0
    classify_latency_ms: float = 0.0
    retrieve_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "message": self.message,
            "triage": {
                "intent": self.intent,
                "intent_confidence": round(self.intent_confidence, 3),
                "risk_level": self.risk_level,
                "risk_confidence": round(self.risk_confidence, 3),
                "safety_flag": self.safety_flag,
            },
            "retrieval": {
                "num_protocols": len(self.retrieved_protocols),
                "protocols": self.retrieved_protocols,
            },
            "response": {
                "draft": self.draft_response,
                "recommended_action": self.recommended_action,
                "escalation_needed": self.escalation_needed,
                "llm_confidence": round(self.llm_confidence, 3),
                "safety_notes": self.safety_notes,
            },
            "hitl": {
                "requires_human_review": self.requires_human_review,
            },
            "latency": {
                "total_ms": round(self.total_latency_ms, 1),
                "classify_ms": round(self.classify_latency_ms, 1),
                "retrieve_ms": round(self.retrieve_latency_ms, 1),
                "llm_ms": round(self.llm_latency_ms, 1),
            },
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ChronicGuardPipeline:
    """
    Full inference pipeline for ChronicGuard AI.
    Supports batch evaluation and single-message inference.
    """

    def __init__(
        self,
        classifier_path: Path | None = None,
        chroma_dir: str = "./chroma_db",
        use_sbert: bool = True,
        use_reranker: bool = True,
        llm_model: str = "gpt-4o-mini",
    ):
        self.classifier = (
            TriageClassifier.load(classifier_path)
            if classifier_path and classifier_path.exists()
            else TriageClassifier(use_sbert=use_sbert)
        )
        self.retriever = CareRetriever(persist_dir=chroma_dir, use_reranker=use_reranker)
        self.drafter = ResponseDrafter(model=llm_model)
        self._ready = False

    def setup(self, train_data: list[dict] | None = None):
        """
        Initialize the pipeline:
        - Build ChromaDB index
        - Train classifier if train_data is provided
        """
        print("Building care protocol index...")
        self.retriever.build_index()

        if train_data:
            print(f"Training classifier on {len(train_data)} examples...")
            texts = [d["message"] for d in train_data]
            intents = [d["intent"] for d in train_data]
            risks = [d["risk_level"] for d in train_data]
            self.classifier.fit(texts, intents, risks)
            print(f"Classifier ready (model: {self.classifier.model_name})")

        self._ready = True
        print("Pipeline ready.")

    def run(self, message: str) -> PipelineResult:
        """Run the full pipeline on a single patient message."""
        result = PipelineResult(message=message)
        total_start = time.time()

        # ── Step 1: Triage classification ────────────────────────────────────
        t0 = time.time()
        triage = self.classifier.predict(message)
        result.classify_latency_ms = (time.time() - t0) * 1000

        result.intent = triage.intent
        result.intent_confidence = triage.intent_confidence
        result.risk_level = triage.risk_level
        result.risk_confidence = triage.risk_confidence
        result.safety_flag = triage.safety_flag
        result.requires_human_review = triage.requires_human_review

        # ── Step 2: RAG retrieval ─────────────────────────────────────────────
        t0 = time.time()
        retrieval = self.retriever.retrieve(
            query=message,
            intent=result.intent,
            risk_level=result.risk_level,
            n_candidates=6,
            top_k=3,
        )
        result.retrieve_latency_ms = (time.time() - t0) * 1000

        result.retrieved_protocols = [
            {"title": d.title, "category": d.category, "score": round(d.rerank_score, 3)}
            for d in retrieval.documents
        ]
        result.context_window = retrieval.context_window

        # ── Step 3: LLM response drafting ────────────────────────────────────
        t0 = time.time()
        draft = self.drafter.draft(
            message=message,
            intent=result.intent,
            risk_level=result.risk_level,
            context_window=result.context_window,
            requires_human_review=result.requires_human_review,
        )
        result.llm_latency_ms = draft.latency_ms

        result.draft_response = draft.draft_response
        result.recommended_action = draft.recommended_action
        result.escalation_needed = draft.escalation_needed
        result.llm_confidence = draft.confidence
        result.safety_notes = draft.safety_notes
        result.requires_human_review = draft.requires_human_review

        result.total_latency_ms = (time.time() - total_start) * 1000
        return result

    def run_batch(self, messages: list[str]) -> list[PipelineResult]:
        return [self.run(m) for m in messages]
