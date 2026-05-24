"""
ChronicGuard AI — Conversational Chat + Voice Interface
Multi-turn dialogue with triage results inline.
Voice input: care managers can speak patient messages instead of typing.
Run: streamlit run dashboard/chat.py --server.port 8502
"""

import sys, csv, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import streamlit.components.v1 as components

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
.voice-btn {
    background: #7c3aed; color: white; border: none;
    padding: 8px 16px; border-radius: 8px; cursor: pointer;
    font-size: 13px; font-weight: 500;
}
.voice-btn.recording { background: #dc2626; animation: pulse 1s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.6} }
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
SYSTEM_CONTEXT = """You are ChronicGuard AI, a care management assistant.
You help care managers understand patient messages and triage priorities.
Always be warm, clear, and safety-conscious. Never diagnose or prescribe."""


def build_conversation_response(user_input, triage_result, history):
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_key:
        intent = triage_result.intent if triage_result else "unknown"
        risk = triage_result.risk_level if triage_result else "unknown"
        icon = RISK_ICONS.get(risk, "⚪")
        action = "Immediate human review required." if risk in ("urgent","high") else "Handles within standard SLA."
        return f"{icon} Classified as **{intent.replace('_',' ')}** with **{risk}** risk. {action}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        messages = [{"role": "system", "content": SYSTEM_CONTEXT}]
        for h in history[-6:]:
            messages.append({"role": h["role"], "content": h["content"]})
        if triage_result:
            context = (
                f"[TRIAGE] Intent: {triage_result.intent}, Risk: {triage_result.risk_level}, "
                f"Safety flag: {triage_result.safety_flag}, Human review: {triage_result.requires_human_review}."
            )
            messages[-1]["content"] = context + "\n\nUser: " + user_input
        else:
            messages.append({"role": "user", "content": user_input})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=300, temperature=0.4,
        )
        return resp.choices[0].message.content
    except Exception as e:
        risk = triage_result.risk_level if triage_result else "unknown"
        return f"Triage complete. Risk level: **{risk}**. {'Escalate immediately.' if risk in ('urgent','high') else 'Route to care manager.'}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChronicGuard AI")
    st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
    st.divider()

    # Outcome simulation stats
    sim_path = Path("results/outcome_simulation.json")
    if sim_path.exists():
        with open(sim_path) as f:
            sim = json.load(f)
        st.markdown("### Outcome simulation")
        st.metric("Urgent response",
                  f"{sim['response_time']['ai_urgent_median_hours']*60:.0f} min",
                  delta=f"vs {sim['response_time']['manual_urgent_median_hours']:.1f}h manual")
        st.metric("Care gap closure",
                  f"{sim['care_gaps']['ai_closure_rate']*100:.0f}%",
                  delta=f"+{sim['care_gaps']['improvement_pct']:.0f}%")
        st.metric("CM capacity",
                  f"{sim['efficiency']['ai_cm_capacity_per_day']} msg/day",
                  delta=f"+{sim['efficiency']['capacity_increase_pct']:.0f}%")
        st.divider()

    # Risk timeline alerts
    tl_path = Path("results/risk_timelines.json")
    if tl_path.exists():
        with open(tl_path) as f:
            tl = json.load(f)
        if tl["proactive_outreach_needed"] > 0:
            st.warning(f"⚠️ {tl['proactive_outreach_needed']} patient(s) need proactive outreach")
        st.markdown("### Risk timeline")
        for t in tl["timelines"]:
            icon = "📈" if t["trend"] == "deteriorating" else "📉" if t["trend"] == "improving" else "➡️"
            st.markdown(f"{icon} **{t['name'].split('—')[0].strip()}**: {t['current_risk']} ({t['trend']})")
        st.divider()

    # RAGAS stats
    ragas_path = Path("results/ragas_evaluation.json")
    if ragas_path.exists():
        with open(ragas_path) as f:
            ragas = json.load(f)
        st.markdown("### RAGAS evaluation")
        st.metric("Faithfulness", f"{ragas['avg_faithfulness']:.3f}")
        st.metric("Hallucination rate", f"{ragas['hallucination_rate']:.0%}",
                  delta="Safe" if ragas["hallucination_rate"] == 0 else "Review")
        st.divider()

    # Last triage result
    if "last_triage" in st.session_state and st.session_state.last_triage:
        r = st.session_state.last_triage
        risk = r.risk_level
        icon = RISK_ICONS.get(risk, "⚪")
        st.markdown("### Last triage result")
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


# ── Main ──────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([2, 1])
with col1:
    st.title("💬 ChronicGuard AI — Conversational Triage")
    st.caption("Multi-turn dialogue · Triage results inline · Voice input supported")
    st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
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
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

# ── Voice Input Section ───────────────────────────────────────────────────────
st.markdown("### 🎤 Voice Input")
st.caption("Speak a patient message — useful when on a call with a patient")

# Web Speech API via HTML component
voice_html = """
<div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
  <button id="voiceBtn" onclick="toggleRecording()"
    style="background:#7c3aed;color:white;border:none;padding:8px 18px;
    border-radius:8px;cursor:pointer;font-size:13px;font-weight:500">
    🎤 Start Recording
  </button>
  <span id="status" style="font-size:13px;color:#64748b">Click to speak</span>
</div>
<div id="transcript" style="background:#f8fafc;border:1px solid #e2e8f0;
  border-radius:8px;padding:10px;min-height:40px;font-size:13px;color:#1e293b;
  margin-bottom:8px"></div>
<button onclick="useTranscript()"
  style="background:#0f6e56;color:white;border:none;padding:6px 14px;
  border-radius:6px;cursor:pointer;font-size:12px">
  Use this text
</button>

<script>
let recognition = null;
let isRecording = false;
let finalTranscript = '';

function toggleRecording() {
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    document.getElementById('status').textContent = 'Voice not supported in this browser. Use Chrome.';
    return;
  }
  if (isRecording) {
    recognition.stop();
  } else {
    startRecording();
  }
}

function startRecording() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = 'en-US';

  recognition.onstart = () => {
    isRecording = true;
    finalTranscript = '';
    document.getElementById('voiceBtn').textContent = '⏹ Stop Recording';
    document.getElementById('voiceBtn').style.background = '#dc2626';
    document.getElementById('status').textContent = 'Listening...';
  };

  recognition.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + ' ';
      } else {
        interim += event.results[i][0].transcript;
      }
    }
    document.getElementById('transcript').textContent = finalTranscript + interim;
  };

  recognition.onerror = (e) => {
    document.getElementById('status').textContent = 'Error: ' + e.error + '. Try Chrome browser.';
    stopRecording();
  };

  recognition.onend = () => { stopRecording(); };
  recognition.start();
}

function stopRecording() {
  isRecording = false;
  document.getElementById('voiceBtn').textContent = '🎤 Start Recording';
  document.getElementById('voiceBtn').style.background = '#7c3aed';
  document.getElementById('status').textContent = finalTranscript ? 'Click "Use this text" to triage' : 'Click to speak';
}

function useTranscript() {
  const text = document.getElementById('transcript').textContent.trim();
  if (text) {
    window.parent.postMessage({type: 'streamlit:setComponentValue', value: text}, '*');
  }
}
</script>
"""

voice_result = components.html(voice_html, height=140)

# Voice text input fallback
voice_input = st.text_input(
    "Or paste transcribed text here:",
    value=st.session_state.voice_text,
    placeholder="Voice transcript will appear here, or type manually...",
    key="voice_input_field",
)

if voice_input and st.button("🔍 Triage voice message", type="primary"):
    st.session_state.voice_text = voice_input
    user_input = voice_input
    send = True
else:
    send = False
    user_input = ""

st.divider()

# ── Chat Interface ─────────────────────────────────────────────────────────────
st.markdown("### 💬 Chat")

# Display history
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

# Text input
col_input, col_btn = st.columns([5, 1])
with col_input:
    text_input = st.text_input(
        "Message", placeholder="Type a patient message or follow-up question...",
        label_visibility="collapsed", key="chat_input",
    )
with col_btn:
    send_text = st.button("Send", type="primary", use_container_width=True)

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

if send_text and text_input.strip():
    user_input = text_input.strip()
    send = True

if send and user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Analyzing..."):
        try:
            result = pipeline.run(user_input)
            st.session_state.last_triage = result
            conv_response = build_conversation_response(
                user_input, result, st.session_state.messages
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": conv_response,
                "triage": {
                    "intent": result.intent,
                    "risk_level": result.risk_level,
                    "intent_confidence": result.intent_confidence,
                    "risk_confidence": result.risk_confidence,
                    "requires_human_review": result.requires_human_review,
                },
            })
        except Exception as e:
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"Pipeline error: {str(e)[:100]}",
            })
    st.rerun()

# Why this page is useful
with st.expander("ℹ️ Why use the conversational interface?"):
    st.markdown("""
**Care managers** handle dozens of patient messages per shift. This interface lets them:

- **Speak** instead of type — dictate a patient's message while on the phone
- **Ask follow-up questions** — "What should I do first?" after a triage result
- **See risk inline** — every message shows risk level, intent, and HITL flag immediately
- **Multi-turn context** — the system remembers the conversation and can reference prior messages
- **Real-time sidebar** — outcome simulation numbers, risk timeline alerts, and RAGAS scores always visible

Compared to the main dashboard (which is for running one message at a time), the chat interface is designed for **continuous workflow** — a care manager can process an entire patient interaction without leaving the page.
    """)
