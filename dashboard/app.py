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
    page = st.radio("View", ["Live Demo", "Batch Evaluation", "Outcome Simulation", "Risk Timeline", "RAGAS Eval", "Active Learning", "Calibration", "LangGraph", "System Info"])


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
elif page == "Outcome Simulation":
    st.subheader("Patient Outcome Simulation")
    st.markdown("Simulates the operational impact of AI-assisted triage vs manual triage across 200 synthetic patient messages.")

    import json
    from pathlib import Path

    sim_path = Path("results/outcome_simulation.json")
    if not sim_path.exists():
        st.warning("Run `python src/outcome_simulation.py` first.")
    else:
        with open(sim_path) as f:
            d = json.load(f)

        rt = d["response_time"]
        sf = d["safety"]
        cg = d["care_gaps"]
        ef = d["efficiency"]

        st.subheader("Response Time")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Manual median", f"{rt['manual_median_hours']:.1f}h")
        c2.metric("AI median", f"{rt['ai_median_hours']:.2f}h", delta=f"-{rt['improvement_pct']:.0f}%")
        c3.metric("Urgent (manual)", f"{rt['manual_urgent_median_hours']:.1f}h")
        c4.metric("Urgent (AI)", f"{rt['ai_urgent_median_hours']*60:.0f} min", delta=f"-{rt['urgent_improvement_pct']:.0f}%")

        st.subheader("Patient Safety")
        c1,c2,c3 = st.columns(3)
        c1.metric("Missed urgent (manual)", sf["manual_missed_urgent"])
        c2.metric("Missed urgent (AI)", sf["ai_missed_urgent"])
        c3.metric("High-risk cases protected", sf["missed_urgent_reduction"])

        st.subheader("Care Gap Closure")
        c1,c2,c3 = st.columns(3)
        c1.metric("Manual closure rate", f"{cg['manual_closure_rate']*100:.0f}%")
        c2.metric("AI closure rate", f"{cg['ai_closure_rate']*100:.0f}%", delta=f"+{cg['improvement_pct']:.0f}%")
        c3.metric("Additional gaps closed", cg["additional_gaps_closed"])

        st.subheader("Care Manager Capacity")
        c1,c2,c3 = st.columns(3)
        c1.metric("Manual (msgs/day)", ef["manual_cm_capacity_per_day"])
        c2.metric("AI-assisted (msgs/day)", ef["ai_cm_capacity_per_day"], delta=f"+{ef['capacity_increase_pct']:.0f}%")
        c3.metric("Capacity increase", f"+{ef['capacity_increase_pct']:.0f}%")

        st.info("Simulation based on published CCM literature benchmarks. All numbers are from synthetic data — not real patient records.")

        with st.expander("Full JSON report"):
            st.json(d)


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Risk Timeline":
    st.subheader("Patient Risk Timeline")
    st.markdown("Tracks simulated patient risk scores over time. Detects deterioration trends and recommends proactive outreach.")

    import json
    from pathlib import Path

    tl_path = Path("results/risk_timelines.json")
    if not tl_path.exists():
        st.warning("Run `python src/risk_timeline.py` first.")
    else:
        with open(tl_path) as f:
            d = json.load(f)

        st.metric("Patients monitored", d["n_patients"])
        st.metric("Proactive outreach needed", d["proactive_outreach_needed"])
        st.divider()

        RISK_COLORS_TL = {"low": "🟢", "medium": "🔵", "high": "🟠", "urgent": "🔴"}

        for tl in d["timelines"]:
            trend_icon = "📈" if tl["trend"] == "deteriorating" else "📉" if tl["trend"] == "improving" else "➡️"
            outreach = "⚠️ PROACTIVE OUTREACH RECOMMENDED" if tl["proactive_outreach_recommended"] else "✅ Stable"
            with st.expander(f"{trend_icon} {tl['name']} — {tl['condition']} | {outreach}"):
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Current risk", tl["current_risk"].title())
                c2.metric("Peak risk", tl["peak_risk"].title())
                c3.metric("Trend", tl["trend"].title())
                c4.metric("Slope", f"{tl['trend_slope']:+.3f}/day")

                st.markdown("**Message history:**")
                for pt in tl["points"]:
                    icon = RISK_COLORS_TL.get(pt["risk_level"], "⚪")
                    review = " 👤" if pt["requires_review"] else ""
                    st.markdown(f"Day {pt['days_offset']:>2} {icon} `{pt['risk_level']:>6}` — {pt['message']}{review}")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "RAGAS Eval":
    st.subheader("RAGAS Evaluation — LLM Hallucination Detection")
    st.markdown("Evaluates LLM response quality: faithfulness (hallucination detection) and answer relevance.")

    import json
    from pathlib import Path

    ragas_path = Path("results/ragas_evaluation.json")
    if not ragas_path.exists():
        st.warning("Run `python src/ragas_evaluator.py` first.")
    else:
        with open(ragas_path) as f:
            d = json.load(f)

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Avg faithfulness", f"{d['avg_faithfulness']:.3f}")
        c2.metric("Avg relevance", f"{d['avg_answer_relevance']:.3f}")
        c3.metric("Context precision", f"{d['avg_context_precision']:.3f}")
        c4.metric("Context recall", f"{d['avg_context_recall']:.3f}")
        c5.metric("Hallucination rate", f"{d['hallucination_rate']:.0%}",
                  delta="Safe" if d["hallucination_rate"] == 0 else "Review needed",
                  delta_color="normal" if d["hallucination_rate"] == 0 else "inverse")

        if d["hallucination_rate"] == 0:
            st.success("No hallucinations detected — all LLM responses are grounded in retrieved care protocols.")
        else:
            st.warning(f"{d['n_hallucinations']} response(s) flagged for potential hallucination.")

        st.divider()
        st.markdown("**Individual case results:**")
        for result in d.get("individual_results", []):
            scores = result.get("scores", {})
            faith = scores.get("faithfulness", 0)
            status = "✅ Grounded" if not result.get("hallucination_detected") else "⚠️ Review"
            with st.expander(f"{status} — {result['question'][:70]}"):
                c1,c2,c3 = st.columns(3)
                c1.metric("Faithfulness", f"{faith:.3f}")
                c2.metric("Answer relevance", f"{scores.get('answer_relevance', 0):.3f}")
                c3.metric("Overall", f"{scores.get('overall', 0):.3f}")
                st.markdown(f"**Draft answer:** {result.get('answer', '')}")

        st.info(f"Evaluation method: {d.get('method', 'heuristic')} | Model: GPT-4o-mini")


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Active Learning":
    st.subheader("Active Learning Loop")
    st.markdown(
        "Care manager corrections are logged and used to retrain the classifier. "
        "Safety corrections (risk upgrades/downgrades) are weighted 3x more heavily."
    )

    import json
    from pathlib import Path

    al_path = Path("results/active_learning_stats.json")
    if not al_path.exists():
        st.warning("Run `python src/active_learning.py` first.")
    else:
        with open(al_path) as f:
            d = json.load(f)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Total corrections", d["total_corrections"])
        c2.metric("Safety corrections", d["safety_corrections"], help="Risk upgrades/downgrades — weighted 3x")
        c3.metric("Avg confidence at correction", f"{d['avg_confidence_at_correction']:.3f}", help="Lower = model was uncertain")
        c4.metric("Should retrain", "Yes" if d["should_retrain"] else "No")

        st.divider()
        progress = d["progress_to_retrain"]
        st.markdown(f"**Progress to next retraining round:** {progress*100:.0f}%")
        st.progress(min(progress, 1.0))

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Weighted correction count", d["weighted_correction_count"])
            st.metric("Retrain threshold", d["retrain_threshold"])
        with c2:
            st.metric("Uncertain examples", d["uncertain_examples"], help="Confidence < 0.75")
            st.metric("Retraining rounds", d.get("retraining_rounds", 0))

        st.divider()
        st.markdown("**How it works:**")
        steps = [
            "Care manager reviews a triage result and corrects intent or risk level",
            "Correction is logged with the original confidence score",
            "Safety corrections (risk upgrades/downgrades) are weighted 3x",
            "When weighted count reaches threshold, classifier retrains",
            "Retraining uses synthetic data + all corrections with safety weighting",
            "This is the path from synthetic data to real production data",
        ]
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")

        corrections_path = Path("results/active_learning/corrections.jsonl")
        if corrections_path.exists():
            with open(corrections_path) as f:
                corrections = [json.loads(line) for line in f if line.strip()]
            safety_corrections = [c for c in corrections if c.get("was_safety_correction")]
            if safety_corrections:
                st.divider()
                st.markdown(f"**Safety-critical corrections ({len(safety_corrections)} total):**")
                for c in safety_corrections[:5]:
                    st.markdown(
                        f"- `{c['predicted_risk']}` → `{c['corrected_risk']}` "
                        f"(confidence was {c['risk_confidence']:.2f}): "
                        f"_{c['message'][:60]}..._"
                    )


# ════════════════════════════════════════════════════════════════════════════════
elif page == "Calibration":
    st.subheader("Confidence Calibration Analysis")
    st.markdown(
        "A well-calibrated model knows when it is uncertain. "
        "Brier score measures calibration — lower is better (0.0 = perfect)."
    )
    import json
    from pathlib import Path
    cal_path = Path("results/calibration_analysis.json")
    if not cal_path.exists():
        st.warning("Run `python notebooks/04_calibration_error_analysis.py` first.")
    else:
        with open(cal_path) as f:
            d = json.load(f)
        c1,c2,c3 = st.columns(3)
        c1.metric("Test messages", d["n_test"])
        c2.metric("Classification errors", d["n_errors"])
        c3.metric("Dangerous misclassifications", d["n_dangerous"],
                  delta="✓ Zero" if d["n_dangerous"]==0 else "⚠ Review",
                  delta_color="normal" if d["n_dangerous"]==0 else "inverse")
        if d["n_dangerous"] == 0:
            st.success("Zero dangerous misclassifications — no urgent/high message was predicted as low/medium.")
        st.subheader("Brier scores by risk class")
        st.caption("Brier score < 0.10 = well calibrated. The model's confidence matches its accuracy.")
        c1,c2,c3,c4 = st.columns(4)
        cols = [c1,c2,c3,c4]
        for i,(cls,res) in enumerate(d["calibration"].items()):
            cols[i%4].metric(
                cls.title(),
                f"{res['brier_score']:.4f}",
                delta="Well calibrated" if res["well_calibrated"] else "Review",
                delta_color="normal" if res["well_calibrated"] else "inverse"
            )
        st.subheader("Error analysis narrative")
        narr_path = Path("results/error_analysis_narrative.md")
        if narr_path.exists():
            st.markdown(narr_path.read_text())
        with st.expander("Full calibration JSON"):
            st.json(d)


# ════════════════════════════════════════════════════════════════════════════════
elif page == "LangGraph":
    st.subheader("LangGraph Agentic Triage Workflow")
    st.markdown(
        "The full triage pipeline implemented as a LangGraph state machine with "
        "5 nodes and conditional routing based on risk level."
    )
    import json
    from pathlib import Path
    lg_path = Path("results/langgraph_workflow.json")
    if not lg_path.exists():
        st.warning("Run `python notebooks/05_langgraph_workflow.py` first.")
    else:
        with open(lg_path) as f:
            d = json.load(f)
        c1,c2,c3 = st.columns(3)
        c1.metric("Workflow nodes", d["nodes"])
        c2.metric("Conditional edges", d["conditional_edges"])
        c3.metric("LangGraph available", "Yes" if d["has_langgraph"] else "No (manual trace)")
        st.divider()
        st.markdown("**Workflow graph:**")
        st.code("classify → safety_check → retrieve → draft → hitl_gate → [escalate | auto_route]", language="text")
        st.markdown("**Node descriptions:**")
        nodes = {
            "classify": "TF-IDF + LR classifies intent (6 classes) and risk (4 levels)",
            "safety_check": "Hard rules: crisis intent → safety flag, urgent → escalate",
            "retrieve": "ChromaDB semantic search → top 3 care protocols retrieved",
            "draft": "GPT-4o-mini drafts protocol-grounded care manager response",
            "hitl_gate": "Confidence < 0.75 or risk high/urgent → mandatory human review",
        }
        for node, desc in nodes.items():
            st.markdown(f"- **{node}** — {desc}")
        st.divider()
        st.markdown("**Test results:**")
        RISK_ICONS = {"urgent":"🔴","high":"🟠","medium":"🔵","low":"🟢"}
        for r in d["test_results"]:
            icon = RISK_ICONS.get(r["risk_level"],"⚪")
            routing_badge = "🚨 Escalate" if r["routing"]=="escalate" else "✅ Auto-route"
            with st.expander(f"{icon} {r['message'][:60]} — {r['risk_level'].upper()} · {routing_badge}"):
                st.markdown(f"**Intent:** {r['intent'].replace('_',' ').title()}")
                st.markdown(f"**Risk:** {r['risk_level']}")
                st.markdown(f"**Safety flag:** {r['safety_flag']}")
                st.markdown(f"**Human review:** {r['requires_human_review']}")
                st.markdown(f"**Routing decision:** {r['routing']}")
                if "step_log" in r:
                    st.markdown("**Step log:**")
                    for step in r["step_log"]:
                        st.code(step, language="text")


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
