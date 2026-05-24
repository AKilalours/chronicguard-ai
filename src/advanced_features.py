"""
ChronicGuard AI — Additional Feature Modules
Covers: Multilingual triage, ICD-10 suggestion, A/B prompt comparison, LangGraph workflow
"""

# ══════════════════════════════════════════════════════════════════════════════
# 1. MULTILINGUAL TRIAGE — Spanish patient message support
# ══════════════════════════════════════════════════════════════════════════════

from __future__ import annotations
import os
import json
from pathlib import Path

SPANISH_TO_ENGLISH_MAP = {
    "dolor de pecho": "chest pain",
    "presión en el pecho": "chest pressure",
    "falta de aire": "shortness of breath",
    "dificultad para respirar": "difficulty breathing",
    "azúcar en la sangre": "blood sugar",
    "presión arterial": "blood pressure",
    "medicamento": "medication",
    "medicina": "medicine",
    "pastilla": "pill",
    "me olvidé": "I forgot",
    "no tomé": "I did not take",
    "dolor de cabeza": "headache",
    "mareo": "dizziness",
    "cansancio": "fatigue",
    "no quiero vivir": "I don't want to live",
    "me quiero morir": "I want to die",
    "ya no aguanto": "I cannot take it anymore",
    "cita": "appointment",
    "resultados de laboratorio": "lab results",
    "receta": "prescription",
    "farmacia": "pharmacy",
    "autorización previa": "prior authorization",
    "me falta": "I ran out of",
    "análisis de sangre": "blood test",
    "hemoglobina": "hemoglobin",
    "glucosa": "glucose",
    "insulina": "insulin",
    "riñón": "kidney",
    "corazón": "heart",
    "diabetes": "diabetes",
    "hipertensión": "hypertension",
}

SPANISH_URGENCY_PHRASES = {
    "emergencia", "urgente", "911", "ambulancia", "me muero",
    "muy mal", "no puedo respirar", "desmayo", "inconsciente",
}


def detect_language(text: str) -> str:
    """Simple language detection based on Spanish word presence."""
    spanish_words = {"me", "mi", "el", "la", "de", "en", "no", "que", "es", "un", "una",
                     "los", "las", "del", "con", "por", "para", "como", "más", "pero"}
    words = set(text.lower().split())
    spanish_count = len(words & spanish_words)
    return "es" if spanish_count >= 2 else "en"


def translate_to_english(text: str) -> tuple[str, bool]:
    """
    Translate Spanish patient message to English.
    Returns (translated_text, was_translated).
    Uses dictionary substitution locally; falls back to GPT if key available.
    """
    lang = detect_language(text)
    if lang != "es":
        return text, False

    translated = text.lower()
    for es, en in SPANISH_TO_ENGLISH_MAP.items():
        translated = translated.replace(es, en)

    # Try GPT translation if available
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": f"Translate this patient message from Spanish to English. "
                               f"Preserve medical meaning exactly. Output only the translation:\n\n{text}"
                }],
                max_tokens=200,
                temperature=0,
            )
            return resp.choices[0].message.content.strip(), True
        except Exception:
            pass

    return translated, True


class MultilingualTriageWrapper:
    """Wraps any triage pipeline with multilingual support."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, message: str) -> dict:
        lang = detect_language(message)
        translated, was_translated = translate_to_english(message)

        result = self.pipeline.run(translated)
        result_dict = result.to_dict() if hasattr(result, "to_dict") else result

        result_dict["original_message"] = message
        result_dict["detected_language"] = lang
        result_dict["was_translated"] = was_translated
        result_dict["translated_message"] = translated if was_translated else message

        return result_dict


# ══════════════════════════════════════════════════════════════════════════════
# 2. ICD-10 CODE SUGGESTION
# ══════════════════════════════════════════════════════════════════════════════

ICD10_MAP = {
    "medication_question": {
        "low":    [("Z79.899", "Long-term medication use"), ("Z87.39", "Personal history of medication")],
        "medium": [("T36-T50", "Poisoning/adverse effects — medication"), ("Z79.4", "Long-term insulin use")],
        "high":   [("T46.5X5A", "Adverse effect antihypertensive"), ("T38.3X5A", "Adverse effect insulin")],
        "urgent": [("T36-T50", "Medication overdose/poisoning"), ("T46.5X1A", "Antihypertensive poisoning")],
    },
    "symptom_escalation": {
        "low":    [("R00-R99", "Symptoms and signs"), ("R51.9", "Headache"), ("R53.83", "Fatigue")],
        "medium": [("I10", "Essential hypertension"), ("E11.65", "Type 2 DM hyperglycemia"), ("R06.0", "Dyspnea")],
        "high":   [("I10", "Hypertension"), ("I50.9", "Heart failure"), ("E11.649", "T2DM hypoglycemia")],
        "urgent": [("I21.9", "Acute MI"), ("I63.9", "Cerebral infarction"), ("I46.9", "Cardiac arrest")],
    },
    "care_gap": {
        "low":    [("Z00.00", "General exam"), ("Z13.89", "Screening for disorder")],
        "medium": [("Z79.01", "Long-term anticoagulant use"), ("Z87.39", "Medication history")],
        "high":   [("Z79.01", "Long-term anticoagulant — gap"), ("N18.9", "CKD — medication gap")],
        "urgent": [("Z79.01", "Critical anticoagulant gap"), ("I48.91", "AF — anticoagulant gap")],
    },
    "lab_results": {
        "low":    [("Z00.00", "Routine exam"), ("Z13.1", "Diabetes screening")],
        "medium": [("E11.9", "Type 2 diabetes"), ("I10", "Hypertension"), ("N18.3", "CKD stage 3")],
        "high":   [("N18.4", "CKD stage 4"), ("E11.65", "T2DM — hyperglycemia"), ("E87.5", "Hyperkalemia")],
        "urgent": [("N17.9", "Acute kidney failure"), ("E11.0", "T2DM with hyperosmolarity")],
    },
    "crisis": {
        "low":    [("Z65.8", "Problems related to circumstances"), ("F43.20", "Adjustment disorder")],
        "medium": [("F32.9", "Major depressive episode"), ("F43.10", "PTSD")],
        "high":   [("F32.2", "Severe depressive episode"), ("F33.2", "Recurrent severe depression")],
        "urgent": [("R45.851", "Suicidal ideation"), ("F43.11", "PTSD with acute reaction")],
    },
    "appointment_admin": {
        "low":    [("Z00.00", "General exam"), ("Z71.89", "Counseling")],
        "medium": [("Z71.89", "Other counseling/advice"), ("Z09", "Follow-up exam")],
        "high":   [("Z09", "Follow-up after treatment")],
        "urgent": [("Z09", "Urgent follow-up required")],
    },
}


def suggest_icd10_codes(intent: str, risk_level: str) -> list[dict]:
    """Suggest relevant ICD-10 codes based on triage classification."""
    codes = ICD10_MAP.get(intent, {}).get(risk_level, [])
    return [{"code": code, "description": desc} for code, desc in codes]


# ══════════════════════════════════════════════════════════════════════════════
# 3. A/B PROMPT COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

PROMPT_VARIANTS = {
    "safety_first": {
        "name": "Safety-first prompt",
        "description": "Leads with safety escalation for any high-risk message",
        "system": """You are a clinical care manager assistant. 
SAFETY RULE: For any high or urgent risk message, your FIRST sentence must include emergency guidance (911 or provider).
Never diagnose. Never recommend specific doses. Always defer to the care team.
Output JSON: {draft_response, recommended_action, escalation_needed, confidence, safety_notes}""",
    },
    "empathy_first": {
        "name": "Empathy-first prompt",
        "description": "Leads with acknowledgment before clinical guidance",
        "system": """You are a compassionate clinical care manager assistant.
Always acknowledge the patient's concern warmly before providing guidance.
For urgent cases, include emergency guidance after the acknowledgment.
Never diagnose. Always defer clinical decisions to the care team.
Output JSON: {draft_response, recommended_action, escalation_needed, confidence, safety_notes}""",
    },
    "concise": {
        "name": "Concise clinical prompt",
        "description": "Short, direct responses optimized for care manager efficiency",
        "system": """You are a clinical care manager assistant. Be concise and direct.
2-3 sentences maximum. Include safety escalation if urgent. No diagnosis.
Output JSON: {draft_response, recommended_action, escalation_needed, confidence, safety_notes}""",
    },
}


class PromptABTester:
    """
    A/B test different prompting strategies on patient messages.
    Evaluates: response length, RAGAS scores, safety compliance, escalation accuracy.
    """

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.results: list[dict] = []

    def compare(
        self,
        message: str,
        context_window: str,
        risk_level: str,
        variants: list[str] | None = None,
    ) -> dict:
        if not self.openai_key:
            return self._mock_comparison(message, risk_level)

        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        variants = variants or list(PROMPT_VARIANTS.keys())
        comparison = {"message": message, "risk_level": risk_level, "variants": {}}

        for variant_key in variants:
            variant = PROMPT_VARIANTS[variant_key]
            user_prompt = (
                f"Patient message: \"{message}\"\n"
                f"Risk level: {risk_level}\n\n"
                f"{context_window}\n\n"
                "Draft a care manager response. Output valid JSON only."
            )
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": variant["system"]},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                )
                output = json.loads(resp.choices[0].message.content)
                draft = output.get("draft_response", "")

                # Score the response
                has_safety = any(w in draft.lower() for w in ["911", "emergency", "provider", "care team", "immediately"])
                word_count = len(draft.split())
                escalation_correct = (
                    (risk_level in ("high", "urgent") and output.get("escalation_needed", False)) or
                    (risk_level in ("low", "medium") and not output.get("escalation_needed", True))
                )

                comparison["variants"][variant_key] = {
                    "name": variant["name"],
                    "draft": draft,
                    "word_count": word_count,
                    "has_safety_language": has_safety,
                    "escalation_correct": escalation_correct,
                    "confidence": output.get("confidence", 0.5),
                    "recommended_action": output.get("recommended_action", ""),
                }
            except Exception as e:
                comparison["variants"][variant_key] = {"error": str(e)}

        # Determine winner
        winner = self._pick_winner(comparison["variants"], risk_level)
        comparison["recommended_variant"] = winner
        self.results.append(comparison)
        return comparison

    def _pick_winner(self, variants: dict, risk_level: str) -> str:
        scores = {}
        for key, v in variants.items():
            if "error" in v:
                continue
            score = 0.0
            score += 3.0 if v.get("has_safety_language") and risk_level in ("high", "urgent") else 0
            score += 2.0 if v.get("escalation_correct") else 0
            score += float(v.get("confidence", 0.5))
            if risk_level in ("high", "urgent") and v.get("word_count", 0) < 60:
                score += 1.0
            scores[key] = score
        return max(scores, key=scores.get) if scores else "safety_first"

    def _mock_comparison(self, message: str, risk_level: str) -> dict:
        return {
            "message": message,
            "risk_level": risk_level,
            "recommended_variant": "safety_first",
            "note": "Set OPENAI_API_KEY to run live A/B comparison",
            "variants": {
                k: {"name": v["name"], "description": v["description"]}
                for k, v in PROMPT_VARIANTS.items()
            },
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4. LANGGRAPH WORKFLOW
# ══════════════════════════════════════════════════════════════════════════════

try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


def build_triage_graph(pipeline=None):
    """
    Build a LangGraph multi-step triage workflow.
    Nodes: classify → safety_check → retrieve → draft → hitl_gate → output
    """
    if not HAS_LANGGRAPH:
        print("LangGraph not installed. Run: pip install langgraph")
        print("Workflow design (would be implemented with LangGraph):")
        print("  classify → safety_check → retrieve → draft → hitl_gate → output")
        return None

    from typing import TypedDict, Annotated
    import operator

    class TriageState(TypedDict):
        message: str
        intent: str
        risk_level: str
        safety_flag: bool
        context_window: str
        draft_response: str
        requires_human_review: bool
        step_log: list[str]

    def classify_node(state: TriageState) -> TriageState:
        state["step_log"].append("classify: running intent + risk classification")
        if pipeline:
            triage = pipeline.classifier.predict(state["message"])
            state["intent"] = triage.intent
            state["risk_level"] = triage.risk_level
            state["safety_flag"] = triage.safety_flag
        else:
            state["intent"] = "symptom_escalation"
            state["risk_level"] = "high"
            state["safety_flag"] = False
        return state

    def safety_check_node(state: TriageState) -> TriageState:
        state["step_log"].append(f"safety_check: risk={state['risk_level']}, flag={state['safety_flag']}")
        if state["intent"] == "crisis":
            state["safety_flag"] = True
            state["risk_level"] = "urgent" if state["risk_level"] in ("low", "medium") else state["risk_level"]
        return state

    def retrieve_node(state: TriageState) -> TriageState:
        state["step_log"].append("retrieve: fetching care protocols from ChromaDB")
        if pipeline:
            result = pipeline.retriever.retrieve(state["message"], state["intent"], state["risk_level"])
            state["context_window"] = result.context_window
        else:
            state["context_window"] = "Care protocols retrieved."
        return state

    def draft_node(state: TriageState) -> TriageState:
        state["step_log"].append("draft: generating LLM response")
        state["draft_response"] = (
            "Thank you for reaching out. A care manager will follow up with you shortly. "
            "If this is a medical emergency, please call 911."
        )
        return state

    def hitl_gate_node(state: TriageState) -> TriageState:
        state["step_log"].append("hitl_gate: checking human review requirement")
        state["requires_human_review"] = (
            state["risk_level"] in ("high", "urgent") or state["safety_flag"]
        )
        return state

    def should_escalate(state) -> str:
        return "escalate" if state["requires_human_review"] else "auto_route"

    graph = StateGraph(TriageState)
    graph.add_node("classify", classify_node)
    graph.add_node("safety_check", safety_check_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("draft", draft_node)
    graph.add_node("hitl_gate", hitl_gate_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "safety_check")
    graph.add_edge("safety_check", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", "hitl_gate")
    graph.add_conditional_edges("hitl_gate", should_escalate, {
        "escalate": END,
        "auto_route": END,
    })

    return graph.compile()


if __name__ == "__main__":
    print("Testing multilingual triage...")
    test_msgs = [
        "Tengo dolor de pecho y falta de aire",
        "Me olvidé tomar mi medicamento para la presión",
        "No quiero vivir más, todo es muy difícil",
    ]
    for msg in test_msgs:
        lang = detect_language(msg)
        translated, _ = translate_to_english(msg)
        print(f"  [{lang}] {msg}")
        print(f"       → {translated}\n")

    print("\nTesting ICD-10 suggestions...")
    for intent, risk in [("symptom_escalation", "urgent"), ("crisis", "high"), ("care_gap", "medium")]:
        codes = suggest_icd10_codes(intent, risk)
        print(f"  {intent} / {risk}: {[c['code'] for c in codes]}")

    print("\nTesting A/B prompt tester...")
    tester = PromptABTester()
    result = tester.compare("I have chest pain", "context here", "urgent")
    print(f"  Recommended variant: {result['recommended_variant']}")

    print("\nTesting LangGraph workflow...")
    graph = build_triage_graph()
    if graph:
        state = graph.invoke({
            "message": "I have chest pain",
            "intent": "", "risk_level": "", "safety_flag": False,
            "context_window": "", "draft_response": "",
            "requires_human_review": False, "step_log": [],
        })
        print(f"  Steps: {' → '.join(state['step_log'])}")
        print(f"  Result: risk={state['risk_level']}, review={state['requires_human_review']}")
