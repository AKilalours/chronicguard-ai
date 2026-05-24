"""
ChronicGuard AI — Interactive Demo Dashboard
Run: streamlit run dashboard/app.py
"""

import sys
import csv
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="ChronicGuard AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.risk-urgent  { background:#fee2e2; border-left:4px solid #dc2626; padding:12px; border-radius:6px; }
.risk-high    { background:#fef3c7; border-left:4px solid #d97706; padding:12px; border-radius:6px; }
.risk-medium  { background:#eff6ff; border-left:4px solid #3b82f6; padding:12px; border-radius:6px; }
.risk-low     { background:#f0fdf4; border-left:4px solid #22c55e; padding:12px; border-radius:6px; }
.hitl-badge   { background:#7c3aed; color:white; padding:3px 10px; border-radius:12px; font-size:12px; }
.metric-card  { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:16px; }
</style>
""", unsafe_allow_html=True)


# ── Pipeline setup ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading ChronicGuard AI pipeline...")
def load_pipeline():
    from src.pipeline import ChronicGuardPipeline
    p = ChronicGuardPipeline(use_sbert=False, use_reranker=False)
    data_path = Path("data/synthetic_messages.csv")
    train_data = []
    if data_path.exists():
        with open(data_path) as f:
            train_data = list(csv.DictReader(f))
    p.setup(train_data=train_data if train_data else None)
    return p

@st.cache_resource(show_spinner=False)
def load_evaluator():
    from src.evaluation import SafetyEvaluator
    return SafetyEvaluator()


RISK_COLORS = {"urgent": "🔴", "high": "🟠", "medium": "🔵", "low": "🟢"}
INTENT_LABELS = {
    "medication_question": "💊 Medication",
    "symptom_escalation": "🩺 Symptom",
    "appointment_admin": "📅 Admin",
    "lab_results": "🧪 Lab Results",
    "care_gap": "⚠️ Care Gap",
    "crisis": "🆘 Crisis",
}

EXAMPLE_MESSAGES = [
    "I have chest pain and shortness of breath today. It started this morning.",
    "I forgot to take my blood pressure medicine for two days. Should I double the dose?",
    "My blood sugar has been over 300 for two days and I feel very thirsty and tired.",
    "I've been out of my blood thinner for five days. The prior authorization was denied.",
    "I don't want to be here anymore. I've been thinking about hurting myself.",
    "Can I reschedule my diabetes follow-up appointment next Tuesday?",
    "No one called me after my lab results came in last week. My A1C was 8.2.",
    "I haven't had an eye exam in two years. Is that something the program can help with?",
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/badge/ChronicGuard-AI-blue?style=flat-square", width=180)
    st.markdown("### ChronicGuard AI")
    st.markdown("Safety-first patient message triage for chronic care management.")
    st.divider()
    st.markdown("**Model stack**")
    st.markdown("- TF-IDF + Logistic Regression (baseline)\n- ChromaDB vector store (RAG)\n- GPT-4o-mini (drafting)")
    st.divider()
    st.markdown("**Safety constraint**")
    st.markdown("`urgent_recall ≥ 0.92` — hard requirement, not a trade-off.")
    st.divider()
    page = st.radio("View", ["Live Demo", "Batch Evaluation", "System Info"])


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("🏥 ChronicGuard AI")
st.caption("Chronic care management copilot · Research prototype · All outputs require licensed care manager review")
st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")

pipeline = load_pipeline()
evaluator = load_evaluator()


# ════════════════════════════════════════════════════════════════════════════════
if page == "Live Demo":

    st.subheader("Patient Message Triage")

    col1, col2 = st.columns([2, 1])
    with col1:
        example = st.selectbox("Load an example message", ["— type your own —"] + EXAMPLE_MESSAGES)
        default_text = "" if example.startswith("—") else example
        message = st.text_area("Patient message", value=default_text, height=100,
                                placeholder="Enter a patient message to triage...")

    with col2:
        st.markdown("&nbsp;")
        run = st.button("▶ Run Triage Pipeline", type="primary", use_container_width=True)
        include_draft = st.checkbox("Generate LLM draft response", value=True)
        st.caption("Draft responses require an OpenAI API key (OPENAI_API_KEY env var)")

    if run and message.strip():
        with st.spinner("Running pipeline..."):
            start = time.time()
            result = pipeline.run(message.strip())
            elapsed = (time.time() - start) * 1000

        # ── Triage output ────────────────────────────────────────────────────
        st.divider()
        risk = result.risk_level
        risk_class = f"risk-{risk}"
        risk_icon = RISK_COLORS.get(risk, "⚪")
        intent_label = INTENT_LABELS.get(result.intent, result.intent)

        st.markdown(f"""
        <div class="{risk_class}">
          <strong>{risk_icon} Risk Level: {risk.upper()}</strong> &nbsp;|&nbsp;
          Intent: {intent_label} &nbsp;|&nbsp;
          {'<span class="hitl-badge">👤 HUMAN REVIEW REQUIRED</span>' if result.requires_human_review else '✅ Auto-routable'}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Risk", risk.title(), help="Predicted risk level")
        c2.metric("Intent confidence", f"{result.intent_confidence:.0%}")
        c3.metric("Risk confidence", f"{result.risk_confidence:.0%}")
        c4.metric("Latency", f"{elapsed:.0f}ms")

        if result.safety_flag:
            st.error("⚠️ Safety flag active — crisis or urgent intent detected. Immediate escalation protocol applies.")

        # ── Retrieved protocols ───────────────────────────────────────────────
        with st.expander(f"📚 Retrieved care protocols ({len(result.retrieved_protocols)})"):
            for p in result.retrieved_protocols:
                st.markdown(f"**{p['title']}** — score: `{p['score']:.3f}`")

        # ── Draft response ────────────────────────────────────────────────────
        if include_draft and result.draft_response:
            st.subheader("📝 Draft Response (for care manager review)")
            st.info(result.draft_response)
            if result.safety_notes:
                st.warning(f"**Safety notes:** {result.safety_notes}")
            st.markdown(f"**Recommended action:** {result.recommended_action}")
            if result.escalation_needed:
                st.error("🚨 Clinical escalation recommended")

        # ── Raw JSON ─────────────────────────────────────────────────────────
        with st.expander("🔍 Raw pipeline output (JSON)"):
            st.json(result.to_dict())

    elif run:
        st.warning("Please enter a message.")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Batch Evaluation":
    st.subheader("Evaluation Framework")
    st.markdown(
        "Run the safety-first evaluation suite on the synthetic dataset. "
        "Urgent-class recall ≥ 0.92 is a hard safety constraint."
    )

    data_path = Path("data/synthetic_messages.csv")
    if not data_path.exists():
        st.error("Dataset not found. Run `python data/generate_data.py` first.")
    else:
        with open(data_path) as f:
            rows = list(csv.DictReader(f))

        st.info(f"Dataset: {len(rows)} labeled messages across 6 intents × 4 risk levels")

        if st.button("▶ Run Evaluation Suite", type="primary"):
            with st.spinner("Running classification on full dataset..."):
                texts = [r["message"] for r in rows]
                true_intents = [r["intent"] for r in rows]
                true_risks = [r["risk_level"] for r in rows]

                pred_intents, pred_risks = [], []
                for msg in texts:
                    t = pipeline.classifier.predict(msg)
                    pred_intents.append(t.intent)
                    pred_risks.append(t.risk_level)

                report = evaluator.evaluate(
                    y_true_risk=true_risks,
                    y_pred_risk=pred_risks,
                    y_true_intent=true_intents,
                    y_pred_intent=pred_intents,
                    messages=texts,
                )

            # ── Safety metrics ────────────────────────────────────────────────
            st.subheader("Safety Metrics")
            s = report.safety
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Urgent recall", f"{s.urgent_recall:.0%}",
                      delta="✓ PASS" if s.safety_constraint_met else "✗ FAIL")
            c2.metric("High recall", f"{s.high_recall:.0%}")
            c3.metric("Critical FN rate", f"{s.critical_false_negative_rate:.1%}")
            c4.metric("Safety constraint", "Met ✓" if s.safety_constraint_met else "Failed ✗")

            # ── Performance ───────────────────────────────────────────────────
            st.subheader("Classification Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Risk macro-F1", f"{report.risk_macro_f1:.3f}")
            c2.metric("Risk accuracy", f"{report.risk_accuracy:.3f}")
            c3.metric("Intent macro-F1", f"{report.intent_macro_f1:.3f}")
            c4.metric("Intent accuracy", f"{report.intent_accuracy:.3f}")

            # ── Per-class breakdown ───────────────────────────────────────────
            st.subheader("Risk — Per-Class Breakdown")
            import pandas as pd
            risk_df = pd.DataFrame(report.risk_per_class).T
            st.dataframe(risk_df.style.highlight_max(axis=0, color="#bbf7d0")
                                      .highlight_min(axis=0, color="#fee2e2"), use_container_width=True)

            # ── Error analysis ────────────────────────────────────────────────
            st.subheader("Safety-Critical Misclassifications")
            dangerous = report.error_analysis.get("dangerous_misclassifications", [])
            if dangerous:
                st.error(f"{len(dangerous)} cases where urgent/high was predicted as low/medium")
                for err in dangerous[:5]:
                    st.markdown(f"- **{err['true_risk']} → {err['predicted_risk']}** | `{err['message'][:80]}...`")
            else:
                st.success("No safety-critical false negatives detected.")

            # ── Save ──────────────────────────────────────────────────────────
            output_dir = Path("results")
            evaluator.save_report(report, output_dir)
            st.success(f"Full report saved to `{output_dir}/`")

            with st.expander("Full JSON report"):
                st.json(report.to_dict())


# ════════════════════════════════════════════════════════════════════════════════
elif page == "System Info":
    st.subheader("System Architecture")
    st.markdown("""
**Pipeline stages:**

1. **Triage Classifier** — TF-IDF + Logistic Regression (baseline) / SentenceTransformer + LR (production)
   - Intent classification: 6 categories
   - Risk classification: 4 levels (low / medium / high / urgent)
   - Safety overrides: crisis intent always triggers HITL; confidence < 0.75 triggers HITL

2. **RAG Retriever** — ChromaDB + MMR + Cross-Encoder Reranker
   - 10 care protocol documents indexed
   - Maximal Marginal Relevance for diversity
   - Cross-encoder reranking for precision

3. **LLM Response Drafter** — GPT-4o-mini with structured output
   - JSON-constrained output (draft, action, escalation, confidence, safety notes)
   - Grounded to retrieved care protocols only
   - Never diagnoses; always defers clinical decisions to provider

4. **HITL Gate** — Human-in-the-loop review trigger
   - Confidence < 0.75 → human review
   - Risk level high/urgent → human review
   - Intent = crisis → always human review

5. **Evaluation Framework** — Safety-first evaluation
   - Primary metric: urgent-class recall ≥ 0.92 (hard constraint)
   - Secondary: macro-F1, accuracy, RAGAS (faithfulness, relevance)
   - Error analysis focused on safety-critical false negatives
""")

    st.subheader("Care Protocols Indexed")
    from src.retriever import CARE_PROTOCOLS
    for p in CARE_PROTOCOLS:
        st.markdown(f"- **{p['id']}** — {p['title']}")


# ════════════════════════════════════════════════════════════════════════════════
# PATCH: Add Conversational AI tab to sidebar
# Add "Conversational AI" to the page radio in the sidebar and handle it below.
# Since we can't easily patch the radio, this adds a standalone chat page.
# Run as: streamlit run dashboard/chat.py for the chat-only view.
