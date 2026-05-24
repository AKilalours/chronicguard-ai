"""
ChronicGuard AI — FastAPI Backend
Production-oriented REST API for the triage pipeline.

Endpoints:
  POST /triage          - Full pipeline: classify + retrieve + draft
  POST /classify        - Classification only (fast path)
  GET  /health          - Health check
  GET  /protocols       - List indexed care protocols
  POST /evaluate        - Run evaluation on a labeled batch

Run:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations
import os
import json
import time
import csv
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse
import traceback
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import ChronicGuardPipeline
from src.evaluation import SafetyEvaluator


# ── Global pipeline instance ─────────────────────────────────────────────────
pipeline: ChronicGuardPipeline | None = None
evaluator = SafetyEvaluator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Initializing ChronicGuard AI pipeline...")
    pipeline = ChronicGuardPipeline(
        use_sbert=False,  # set True when GPU available
        use_reranker=False,  # set True when cross-encoder available
    )

    # Load training data if available
    data_path = Path("data/synthetic_messages.csv")
    train_data = []
    if data_path.exists():
        with open(data_path) as f:
            reader = csv.DictReader(f)
            train_data = list(reader)
        print(f"Loaded {len(train_data)} training examples.")

    pipeline.setup(train_data=train_data if train_data else None)
    print("API ready.")
    yield
    print("Shutting down.")


app = FastAPI(
    title="ChronicGuard AI",
    description="Safety-first patient message triage for chronic care management.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "traceback": traceback.format_exc()},
    )


# ── Request / Response models ─────────────────────────────────────────────────
class TriageRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=2000,
                          example="I have chest pain and shortness of breath today.")
    include_draft: bool = Field(True, description="Whether to generate LLM draft response")


class ClassifyRequest(BaseModel):
    message: str = Field(..., min_length=5, max_length=2000)


class BatchEvaluateRequest(BaseModel):
    messages: list[str]
    true_intents: list[str]
    true_risks: list[str]


class TriageResponse(BaseModel):
    message: str
    intent: str
    intent_confidence: float
    risk_level: str
    risk_confidence: float
    safety_flag: bool
    requires_human_review: bool
    retrieved_protocols: list[dict]
    draft_response: Optional[str]
    recommended_action: Optional[str]
    escalation_needed: Optional[bool]
    safety_notes: Optional[str]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    pipeline_ready: bool
    model: str
    version: str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        pipeline_ready=pipeline is not None and pipeline._ready,
        model=pipeline.classifier.model_name if pipeline else "none",
        version="0.1.0",
    )


@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    if not pipeline or not pipeline._ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    result = pipeline.run(request.message)

    return TriageResponse(
        message=result.message,
        intent=result.intent,
        intent_confidence=result.intent_confidence,
        risk_level=result.risk_level,
        risk_confidence=result.risk_confidence,
        safety_flag=result.safety_flag,
        requires_human_review=result.requires_human_review,
        retrieved_protocols=result.retrieved_protocols,
        draft_response=result.draft_response if request.include_draft else None,
        recommended_action=result.recommended_action if request.include_draft else None,
        escalation_needed=result.escalation_needed if request.include_draft else None,
        safety_notes=result.safety_notes if request.include_draft else None,
        latency_ms=result.total_latency_ms,
    )


@app.post("/classify")
async def classify(request: ClassifyRequest):
    """Fast path — classification only, no RAG or LLM."""
    if not pipeline or not pipeline._ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    triage = pipeline.classifier.predict(request.message)
    return triage.to_dict()


@app.get("/protocols")
async def list_protocols():
    """List all indexed care protocols."""
    from src.retriever import CARE_PROTOCOLS
    return {
        "count": len(CARE_PROTOCOLS),
        "protocols": [
            {"id": p["id"], "title": p["title"], "category": p["category"]}
            for p in CARE_PROTOCOLS
        ],
    }


@app.post("/evaluate")
async def evaluate_batch(request: BatchEvaluateRequest):
    if not pipeline or not pipeline._ready:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    if not (len(request.messages) == len(request.true_intents) == len(request.true_risks)):
        raise HTTPException(status_code=400, detail="messages, true_intents, true_risks must have equal length")

    # Run predictions
    pred_intents, pred_risks = [], []
    for msg in request.messages:
        triage = pipeline.classifier.predict(msg)
        pred_intents.append(triage.intent)
        pred_risks.append(triage.risk_level)

    report = evaluator.evaluate(
        y_true_risk=request.true_risks,
        y_pred_risk=pred_risks,
        y_true_intent=request.true_intents,
        y_pred_intent=pred_intents,
        messages=request.messages,
    )

    return report.to_dict()
