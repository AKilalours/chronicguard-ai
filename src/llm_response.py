"""
ChronicGuard AI — LLM Response Drafter + HITL Gate
Generates safe, grounded care manager draft responses using retrieved context.
All outputs are drafts for licensed care manager review — never sent autonomously.

Usage:
    from src.llm_response import ResponseDrafter
    drafter = ResponseDrafter()
    output = drafter.draft(message, triage_result, retrieval_result)
"""

from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    HAS_OLLAMA = False


SYSTEM_PROMPT = """You are a clinical care manager assistant supporting a chronic care management platform.

Your role is to draft a response for a licensed care manager to send to a patient.

STRICT RULES:
1. Never provide a medical diagnosis.
2. Never recommend specific medication doses or changes.
3. Always recommend the patient speak with their care team for clinical questions.
4. For urgent or high-risk messages, always include language directing to emergency services (911) or provider.
5. Responses must be warm, empathetic, and clear — written for a patient audience.
6. Use ONLY information from the provided care protocols — do not invent clinical facts.
7. Keep responses concise: 2-4 sentences for low/medium risk; 1-2 direct sentences for urgent.
8. This is a DRAFT for care manager review. Never claim to be a human or a doctor.

Output JSON only with this exact schema:
{
  "draft_response": "string — the message to send to patient",
  "recommended_action": "string — what the care manager should do next",
  "escalation_needed": boolean,
  "confidence": float (0.0 to 1.0),
  "safety_notes": "string — any concerns the care manager should note before sending"
}"""


def _build_user_prompt(
    message: str,
    intent: str,
    risk_level: str,
    context_window: str,
    care_gaps: list[str],
) -> str:
    return f"""Patient message: "{message}"

Classified intent: {intent}
Risk level: {risk_level}
Identified care gaps: {', '.join(care_gaps) if care_gaps else 'none'}

{context_window}

Draft an appropriate care manager response following the system rules.
Output valid JSON only."""


@dataclass
class DraftResponse:
    draft_response: str
    recommended_action: str
    escalation_needed: bool
    confidence: float
    safety_notes: str
    latency_ms: float
    model_used: str
    requires_human_review: bool

    def to_dict(self) -> dict:
        return {
            "draft_response": self.draft_response,
            "recommended_action": self.recommended_action,
            "escalation_needed": self.escalation_needed,
            "confidence": round(self.confidence, 3),
            "safety_notes": self.safety_notes,
            "latency_ms": round(self.latency_ms, 1),
            "model_used": self.model_used,
            "requires_human_review": self.requires_human_review,
        }


class ResponseDrafter:
    """
    LLM response drafter with fallback chain:
    OpenAI GPT-4o-mini → Ollama (local) → rule-based fallback
    """

    HITL_THRESHOLD = 0.75  # confidence below this triggers mandatory human review

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3):
        self.model = model
        self.temperature = temperature
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._client = OpenAI(api_key=api_key) if (HAS_OPENAI and api_key) else None

    def draft(
        self,
        message: str,
        intent: str,
        risk_level: str,
        context_window: str,
        care_gaps: list[str] | None = None,
        requires_human_review: bool = False,
    ) -> DraftResponse:
        care_gaps = care_gaps or []
        user_prompt = _build_user_prompt(message, intent, risk_level, context_window, care_gaps)

        start = time.time()
        raw = self._call_llm(user_prompt)
        latency_ms = (time.time() - start) * 1000

        parsed = self._parse_response(raw, risk_level)
        parsed["latency_ms"] = latency_ms

        # HITL gate: low confidence or high-risk always requires review
        hitl = (
            requires_human_review
            or parsed["confidence"] < self.HITL_THRESHOLD
            or risk_level in ("high", "urgent")
            or intent == "crisis"
        )

        return DraftResponse(
            draft_response=parsed["draft_response"],
            recommended_action=parsed["recommended_action"],
            escalation_needed=parsed["escalation_needed"],
            confidence=parsed["confidence"],
            safety_notes=parsed["safety_notes"],
            latency_ms=latency_ms,
            model_used=self.model,
            requires_human_review=hitl,
        )

    def _call_llm(self, user_prompt: str) -> str:
        if self._client:
            return self._call_openai(user_prompt)
        elif HAS_OLLAMA:
            return self._call_ollama(user_prompt)
        else:
            return self._rule_based_fallback(user_prompt)

    def _call_openai(self, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_ollama(self, user_prompt: str) -> str:
        response = ollama.chat(
            model="llama3",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response["message"]["content"]

    def _rule_based_fallback(self, user_prompt: str) -> str:
        """Deterministic fallback when no LLM is available."""
        return json.dumps({
            "draft_response": (
                "Thank you for reaching out to your care team. "
                "A care manager will review your message and follow up with you shortly. "
                "If you are experiencing a medical emergency, please call 911 immediately."
            ),
            "recommended_action": "Care manager review required",
            "escalation_needed": False,
            "confidence": 0.5,
            "safety_notes": "Rule-based fallback response — LLM not available. Manual review mandatory.",
        })

    def _parse_response(self, raw: str, risk_level: str) -> dict:
        try:
            parsed = json.loads(raw)
            # Validate required fields
            required = ["draft_response", "recommended_action", "escalation_needed",
                        "confidence", "safety_notes"]
            for field in required:
                if field not in parsed:
                    raise ValueError(f"Missing field: {field}")
            parsed["confidence"] = float(parsed["confidence"])
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError):
            # Safe fallback
            return {
                "draft_response": (
                    "We received your message and a care manager will follow up with you soon. "
                    "If this is a medical emergency, please call 911."
                ),
                "recommended_action": "Manual care manager review — parsing failed",
                "escalation_needed": risk_level in ("high", "urgent"),
                "confidence": 0.3,
                "safety_notes": "JSON parsing failed. Full manual review required before sending.",
            }
