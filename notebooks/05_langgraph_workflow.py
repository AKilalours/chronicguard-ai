"""
ChronicGuard AI — LangGraph Agentic Triage Workflow
Demonstrates the full pipeline as a LangGraph state machine.
Run: python notebooks/05_langgraph_workflow.py
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("CHRONICGUARD AI — LANGGRAPH AGENTIC WORKFLOW")
print("=" * 60)

# ── Try LangGraph first ────────────────────────────────────────────────────────
try:
    from langgraph.graph import StateGraph, END
    from typing import TypedDict
    HAS_LANGGRAPH = True
    print("LangGraph available — running full graph\n")
except ImportError:
    HAS_LANGGRAPH = False
    print("LangGraph not installed — showing design + manual trace\n")

# ── Load pipeline ──────────────────────────────────────────────────────────────
import csv
from src.pipeline import ChronicGuardPipeline

pipeline = ChronicGuardPipeline(use_sbert=False, use_reranker=False)
with open("data/synthetic_messages.csv") as f:
    train_data = list(csv.DictReader(f))
pipeline.setup(train_data=train_data)
print(f"Pipeline ready ({len(train_data)} training examples)\n")

# ── State definition ───────────────────────────────────────────────────────────
TEST_MESSAGES = [
    "I have chest pain and shortness of breath",
    "I don't want to be here anymore",
    "Can I reschedule my diabetes appointment?",
    "My blood sugar has been over 300 for two days",
]

if HAS_LANGGRAPH:
    class TriageState(TypedDict):
        message: str
        intent: str
        risk_level: str
        safety_flag: bool
        n_protocols: int
        draft_preview: str
        requires_human_review: bool
        routing: str
        step_log: list

    def node_classify(state: TriageState) -> TriageState:
        triage = pipeline.classifier.predict(state["message"])
        state["intent"] = triage.intent
        state["risk_level"] = triage.risk_level
        state["safety_flag"] = triage.safety_flag
        state["step_log"].append(
            f"classify → intent={triage.intent}, risk={triage.risk_level}, "
            f"conf={triage.intent_confidence:.2f}"
        )
        return state

    def node_safety_check(state: TriageState) -> TriageState:
        if state["intent"] == "crisis" or state["risk_level"] == "urgent":
            state["safety_flag"] = True
        state["step_log"].append(
            f"safety_check → flag={state['safety_flag']}"
        )
        return state

    def node_retrieve(state: TriageState) -> TriageState:
        result = pipeline.retriever.retrieve(
            state["message"], state["intent"], state["risk_level"]
        )
        state["n_protocols"] = len(result.documents)
        state["step_log"].append(
            f"retrieve → {state['n_protocols']} protocols from ChromaDB"
        )
        return state

    def node_draft(state: TriageState) -> TriageState:
        if state["risk_level"] in ("urgent", "high") or state["safety_flag"]:
            state["draft_preview"] = "ESCALATE: Contact care manager immediately."
        else:
            state["draft_preview"] = "Route to care manager within standard SLA."
        state["step_log"].append(f"draft → '{state['draft_preview'][:50]}'")
        return state

    def node_hitl_gate(state: TriageState) -> TriageState:
        state["requires_human_review"] = (
            state["risk_level"] in ("high", "urgent") or state["safety_flag"]
        )
        state["routing"] = "escalate" if state["requires_human_review"] else "auto_route"
        state["step_log"].append(
            f"hitl_gate → review={state['requires_human_review']}, routing={state['routing']}"
        )
        return state

    def route_decision(state) -> str:
        return state["routing"]

    graph = StateGraph(TriageState)
    graph.add_node("classify",     node_classify)
    graph.add_node("safety_check", node_safety_check)
    graph.add_node("retrieve",     node_retrieve)
    graph.add_node("draft",        node_draft)
    graph.add_node("hitl_gate",    node_hitl_gate)

    graph.set_entry_point("classify")
    graph.add_edge("classify",     "safety_check")
    graph.add_edge("safety_check", "retrieve")
    graph.add_edge("retrieve",     "draft")
    graph.add_edge("draft",        "hitl_gate")
    graph.add_conditional_edges("hitl_gate", route_decision, {
        "escalate":   END,
        "auto_route": END,
    })
    app = graph.compile()

    print("LangGraph workflow compiled:")
    print("  classify → safety_check → retrieve → draft → hitl_gate → [END]\n")

    results = []
    for msg in TEST_MESSAGES:
        print(f"Message: \"{msg}\"")
        state = app.invoke({
            "message": msg,
            "intent": "", "risk_level": "", "safety_flag": False,
            "n_protocols": 0, "draft_preview": "",
            "requires_human_review": False, "routing": "",
            "step_log": [],
        })
        print(f"  Steps: {' → '.join(s.split(' → ')[0] for s in state['step_log'])}")
        print(f"  Result: risk={state['risk_level']}, review={state['requires_human_review']}, routing={state['routing']}")
        print(f"  Draft: {state['draft_preview']}\n")
        results.append({
            "message": msg,
            "intent": state["intent"],
            "risk_level": state["risk_level"],
            "safety_flag": state["safety_flag"],
            "requires_human_review": state["requires_human_review"],
            "routing": state["routing"],
            "n_protocols": state["n_protocols"],
            "step_log": state["step_log"],
        })

else:
    # Manual trace without LangGraph
    results = []
    for msg in TEST_MESSAGES:
        print(f"Message: \"{msg}\"")
        result = pipeline.run(msg)
        steps = ["classify", "safety_check", "retrieve", "draft", "hitl_gate"]
        routing = "escalate" if result.requires_human_review else "auto_route"
        print(f"  Steps: {' → '.join(steps)}")
        print(f"  Result: risk={result.risk_level}, review={result.requires_human_review}, routing={routing}")
        print(f"  Protocols: {len(result.retrieved_protocols)}\n")
        results.append({
            "message": msg,
            "intent": result.intent,
            "risk_level": result.risk_level,
            "safety_flag": result.safety_flag,
            "requires_human_review": result.requires_human_review,
            "routing": routing,
            "n_protocols": len(result.retrieved_protocols),
        })

# ── Save ───────────────────────────────────────────────────────────────────────
Path("results").mkdir(exist_ok=True)
with open("results/langgraph_workflow.json", "w") as f:
    json.dump({
        "workflow": "classify → safety_check → retrieve → draft → hitl_gate",
        "nodes": 5,
        "conditional_edges": 1,
        "has_langgraph": HAS_LANGGRAPH,
        "test_results": results,
    }, f, indent=2)

print("=" * 60)
print(f"Workflow: classify → safety_check → retrieve → draft → hitl_gate")
print(f"Results saved → results/langgraph_workflow.json")
print("=" * 60)
