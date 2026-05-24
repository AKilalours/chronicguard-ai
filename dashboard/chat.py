"""
ChronicGuard AI — Conversational Chat Interface
Multi-turn dialogue demonstrating conversational AI capability.
Run: streamlit run dashboard/chat.py
"""

import sys, csv, os, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

st.set_page_config(
    page_title="ChronicGuard AI — Chat",
    page_icon="💬",
    layout="wide",
)

st.markdown("""
<style>
.user-msg {
    background: #1d4ed8; color: white;
    padding: 10px 16px; border-radius: 12px 12px 2px 12px;
    margin: 4px 0 4px 20%; font-size: 14px;
}
.bot-msg {
    background: #f1f5f9; color: #1e293b;
    padding: 10px 16px; border-radius: 12px 12px 12px 2px;
    margin: 4px 20% 4px 0; font-size: 14px;
}
.risk-pill {
    display:inline-block; padding: 2px 10px;
    border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 6px;
}
.risk-urgent { background:#fee2e2; color:#991b1b; }
.risk-high   { background:#fef3c7; color:#92400e; }
.risk-medium { background:#dbeafe; color:#1e40af; }
.risk-low    { background:#dcfce7; color:#166534; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading pipeline...")
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


pipeline = load_pipeline()

RISK_ICONS = {"urgent": "🔴", "high": "🟠", "medium": "🔵", "low": "🟢"}

# System context for multi-turn conversation
SYSTEM_CONTEXT = """You are ChronicGuard AI, a care management assistant.
You help care managers understand patient messages and triage priorities.
You remember the conversation history and can answer follow-up questions.
Always be warm, clear, and safety-conscious. Never diagnose or prescribe.
If asked about a specific patient message, reference the triage results provided."""


def build_conversation_response(user_input: str, triage_result, history: list) -> str:
    """Build a conversational response incorporating triage context."""
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        intent = triage_result.intent if triage_result else "unknown"
        risk = triage_result.risk_level if triage_result else "unknown"
        icon = {"urgent":"🔴","high":"🟠","medium":"🔵","low":"🟢"}.get(risk,"⚪")
        action = "⚠️ This requires immediate human review — escalate to care manager." if risk in ("urgent","high") else "This can be handled within standard SLA timeframes."
        return f"{icon} Classified as **{intent.replace(chr(95),' ')}** with **{risk}** risk. {action}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)

        messages = [{"role": "system", "content": SYSTEM_CONTEXT}]
        for h in history[-6:]:  # last 6 turns for context window
            messages.append({"role": h["role"], "content": h["content"]})

        if triage_result:
            context = (
                f"[TRIAGE RESULT] Intent: {triage_result.intent}, "
                f"Risk: {triage_result.risk_level}, "
                f"Safety flag: {triage_result.safety_flag}, "
                f"Human review needed: {triage_result.requires_human_review}. "
                f"Protocols retrieved: {len(triage_result.retrieved_protocols)}."
            )
            messages[-1]["content"] = context + "\n\nUser: " + user_input
        else:
            messages.append({"role": "user", "content": user_input})

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300,
            temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"I analyzed the triage result. The key concern here is the **{triage_result.risk_level if triage_result else 'unknown'} risk level**. Would you like me to explain the recommended action?"


# ── UI ────────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.title("💬 ChronicGuard AI — Conversational Triage")
    st.caption("Multi-turn dialogue · Type a patient message to triage, then ask follow-up questions")
st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
st.markdown("<p style='font-size:13px;color:#64748b;margin-top:-8px'>By <strong>Akila Lourdes Miriyala Francis</strong></p>", unsafe_allow_html=True)

with col2:
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_triage" not in st.session_state:
    st.session_state.last_triage = None

# Chat history display
chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="user-msg">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-msg">{msg["content"]}</div>', unsafe_allow_html=True)
            if msg.get("triage"):
                t = msg["triage"]
                risk = t["risk_level"]
                icon = RISK_ICONS.get(risk, "⚪")
                st.markdown(
                    f'<span class="risk-pill risk-{risk}">{icon} {risk.upper()}</span>'
                    f'<span style="font-size:12px;color:#64748b">{t["intent"].replace("_"," ")} · '
                    f'Intent conf: {t["intent_confidence"]:.0%} · '
                    f'{"👤 Human review required" if t["requires_human_review"] else "✅ Routable"}</span>',
                    unsafe_allow_html=True,
                )

st.divider()

# Input
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "Message",
        placeholder="Type a patient message or follow-up question...",
        label_visibility="collapsed",
        key="chat_input",
    )
with col_btn:
    send = st.button("Send", type="primary", use_container_width=True)

# Quick examples
st.markdown("**Quick examples:**")
ex_cols = st.columns(3)
examples = [
    "I have chest pain and shortness of breath",
    "I forgot my blood pressure meds for 2 days",
    "I don't want to be here anymore",
]
for i, (col, ex) in enumerate(zip(ex_cols, examples)):
    if col.button(ex[:35] + "...", key=f"ex_{i}"):
        user_input = ex
        send = True

if send and user_input.strip():
    msg = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": msg})

    with st.spinner("Analyzing..."):
        # Run full triage pipeline
        try:
            result = pipeline.run(msg)
            st.session_state.last_triage = result

            # Build conversational response
            conv_response = build_conversation_response(
                msg, result, st.session_state.messages
            )

            triage_summary = {
                "intent": result.intent,
                "risk_level": result.risk_level,
                "intent_confidence": result.intent_confidence,
                "risk_confidence": result.risk_confidence,
                "requires_human_review": result.requires_human_review,
            }

            st.session_state.messages.append({
                "role": "assistant",
                "content": conv_response,
                "triage": triage_summary,
            })

        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I encountered an issue analyzing that message: {str(e)[:100]}. Please check the pipeline setup.",
            })

    st.rerun()

# Right panel — last triage detail
if st.session_state.last_triage:
    with st.sidebar:
        st.markdown("### Last triage result")
        r = st.session_state.last_triage
        risk = r.risk_level
        icon = RISK_ICONS.get(risk, "⚪")
        st.markdown(f"**Risk:** {icon} {risk.upper()}")
        st.markdown(f"**Intent:** {r.intent.replace('_',' ').title()}")
        st.markdown(f"**Intent confidence:** {r.intent_confidence:.0%}")
        st.markdown(f"**Risk confidence:** {r.risk_confidence:.0%}")
        if r.safety_flag:
            st.error("⚠️ Safety flag active")
        if r.requires_human_review:
            st.warning("👤 Human review required")
        else:
            st.success("✅ Auto-routable")
        st.divider()
        st.markdown("**Retrieved protocols:**")
        for p in r.retrieved_protocols:
            st.markdown(f"- {p['title'][:45]}...")
        st.divider()
        st.markdown("**Latency:**")
        st.markdown(f"- Classify: {r.classify_latency_ms:.0f}ms")
        st.markdown(f"- Retrieve: {r.retrieve_latency_ms:.0f}ms")
        st.markdown(f"- Total: {r.total_latency_ms:.0f}ms")
