"""
ChronicGuard AI — Conversational Chat + Voice I/O
Voice input + voice output with on/off toggle.
Professional UI with glassmorphism design.
"""
import sys, csv, os, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ChronicGuard AI — Voice Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 24px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 20px 60px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(124,58,237,0.15) 0%, transparent 50%),
                radial-gradient(circle at 70% 50%, rgba(15,110,86,0.15) 0%, transparent 50%);
    pointer-events: none;
}
.main-title {
    font-size: 36px;
    font-weight: 700;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-subtitle {
    font-size: 14px;
    color: rgba(255,255,255,0.6);
    margin: 6px 0 0 0;
}
.authors {
    font-size: 12px;
    color: rgba(255,255,255,0.4);
    margin-top: 4px;
}

.voice-panel {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.chat-bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6);
    color: white;
    padding: 12px 18px;
    border-radius: 18px 18px 4px 18px;
    margin: 8px 0 8px 25%;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(29,78,216,0.4);
}
.chat-bubble-bot {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.1);
    color: #e2e8f0;
    padding: 12px 18px;
    border-radius: 18px 18px 18px 4px;
    margin: 8px 25% 8px 0;
    font-size: 14px;
    line-height: 1.5;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}
.risk-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    margin: 4px 2px;
}
.badge-urgent { background: rgba(220,38,38,0.2); color: #fca5a5; border: 1px solid rgba(220,38,38,0.4); }
.badge-high   { background: rgba(217,119,6,0.2);  color: #fcd34d; border: 1px solid rgba(217,119,6,0.4); }
.badge-medium { background: rgba(59,130,246,0.2); color: #93c5fd; border: 1px solid rgba(59,130,246,0.4); }
.badge-low    { background: rgba(34,197,94,0.2);  color: #86efac; border: 1px solid rgba(34,197,94,0.4); }

.stat-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
    backdrop-filter: blur(5px);
    transition: all 0.2s;
}
.stat-label { font-size: 11px; color: rgba(255,255,255,0.4); margin-bottom: 4px; }
.stat-value { font-size: 20px; font-weight: 600; color: white; }
.stat-delta { font-size: 11px; color: #86efac; }

.section-header {
    font-size: 13px;
    font-weight: 600;
    color: rgba(255,255,255,0.5);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 16px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading ChronicGuard AI...")
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
RISK_ICONS = {"urgent":"🔴","high":"🟠","medium":"🔵","low":"🟢"}


def build_response(user_input, triage_result, history):
    openai_key = os.getenv("OPENAI_API_KEY","").strip()
    if not openai_key:
        risk = triage_result.risk_level if triage_result else "unknown"
        intent = triage_result.intent if triage_result else "unknown"
        icon = RISK_ICONS.get(risk,"⚪")
        action = "This requires immediate escalation." if risk in ("urgent","high") else "This can be routed within standard SLA."
        return f"{icon} Classified as **{intent.replace('_',' ')}** with **{risk}** risk. {action}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        msgs = [{"role":"system","content":"You are ChronicGuard AI, a care management assistant. Be concise, warm, safety-conscious. Never diagnose."}]
        for h in history[-6:]:
            msgs.append({"role":h["role"],"content":h["content"]})
        if triage_result:
            ctx = f"[TRIAGE] Intent:{triage_result.intent}, Risk:{triage_result.risk_level}, Safety:{triage_result.safety_flag}, Review:{triage_result.requires_human_review}"
            msgs[-1]["content"] = ctx + "\nUser: " + user_input
        else:
            msgs.append({"role":"user","content":user_input})
        resp = client.chat.completions.create(model="gpt-4o-mini",messages=msgs,max_tokens=250,temperature=0.4)
        return resp.choices[0].message.content
    except Exception:
        risk = triage_result.risk_level if triage_result else "unknown"
        return f"Triage complete. Risk: **{risk}**. {'Escalate immediately.' if risk in ('urgent','high') else 'Route to care manager.'}"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="section-header">ChronicGuard AI</p>', unsafe_allow_html=True)
    st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")

    voice_output_on = st.toggle("🔊 Voice replies", value=True,
                                help="AI reads responses aloud using text-to-speech")
    voice_speed = st.slider("Speech speed", 0.5, 2.0, 1.0, 0.1) if voice_output_on else 1.0
    voice_pitch = st.slider("Voice pitch", 0.0, 2.0, 1.0, 0.1) if voice_output_on else 1.0

    st.divider()

    sim_path = Path("results/outcome_simulation.json")
    if sim_path.exists():
        with open(sim_path) as f:
            sim = json.load(f)
        st.markdown('<p class="section-header">Outcome simulation</p>', unsafe_allow_html=True)
        st.metric("Urgent response", f"{sim['response_time']['ai_urgent_median_hours']*60:.0f} min",
                  delta=f"vs {sim['response_time']['manual_urgent_median_hours']:.1f}h manual")
        st.metric("Care gap closure", f"{sim['care_gaps']['ai_closure_rate']*100:.0f}%",
                  delta=f"+{sim['care_gaps']['improvement_pct']:.0f}%")
        st.metric("CM capacity", f"{sim['efficiency']['ai_cm_capacity_per_day']} msg/day",
                  delta=f"+{sim['efficiency']['capacity_increase_pct']:.0f}%")
        st.divider()

    tl_path = Path("results/risk_timelines.json")
    if tl_path.exists():
        with open(tl_path) as f:
            tl = json.load(f)
        st.markdown('<p class="section-header">Risk timeline</p>', unsafe_allow_html=True)
        if tl["proactive_outreach_needed"] > 0:
            st.warning(f"⚠️ {tl['proactive_outreach_needed']} patient(s) need outreach")
        for t in tl["timelines"]:
            icon = "📈" if t["trend"]=="deteriorating" else "📉" if t["trend"]=="improving" else "➡️"
            st.caption(f"{icon} {t['name'].split('—')[0].strip()}: {t['current_risk']}")
        st.divider()

    ragas_path = Path("results/ragas_evaluation.json")
    if ragas_path.exists():
        with open(ragas_path) as f:
            ragas = json.load(f)
        st.markdown('<p class="section-header">RAGAS evaluation</p>', unsafe_allow_html=True)
        st.metric("Faithfulness", f"{ragas['avg_faithfulness']:.3f}")
        st.metric("Hallucination rate", f"{ragas['hallucination_rate']:.0%}",
                  delta="Safe" if ragas["hallucination_rate"]==0 else "Review")
        st.divider()

    if st.session_state.get("last_triage"):
        r = st.session_state.last_triage
        st.markdown('<p class="section-header">Last triage</p>', unsafe_allow_html=True)
        risk = r.risk_level
        st.markdown(f"{RISK_ICONS.get(risk,'⚪')} **{risk.upper()}** · {r.intent.replace('_',' ').title()}")
        st.caption(f"Intent: {r.intent_confidence:.0%} · Risk: {r.risk_confidence:.0%}")
        if r.safety_flag:
            st.error("⚠️ Safety flag")
        if r.requires_human_review:
            st.warning("👤 Human review")
        else:
            st.success("✅ Auto-routable")
        st.caption(f"⚡ {r.total_latency_ms:.0f}ms total")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <p class="main-title">💬 ChronicGuard AI</p>
    <p class="main-subtitle">Conversational Triage · Voice Input + Voice Replies · Safety-First</p>
    <p class="authors">Built by Akila Lourdes Miriyala Francis &amp; Akilan Manivannan</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([4,1])
with col2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.rerun()

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_triage" not in st.session_state:
    st.session_state.last_triage = None

# ── Voice I/O Component ───────────────────────────────────────────────────────
voice_html = f"""
<div style="font-family:'Inter',sans-serif;padding:4px 0">
  <div style="display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap">
    <button id="voiceBtn" onclick="toggleRecording()"
      style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:white;border:none;
      padding:10px 20px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:600;
      box-shadow:0 4px 15px rgba(124,58,237,0.4);transition:all 0.2s;display:flex;align-items:center;gap:6px">
      🎤 Start Recording
    </button>
    <div id="waveform" style="display:none;align-items:center;gap:3px;height:30px">
      {''.join([f'<div style="width:3px;height:{4+i%5*4}px;background:#7c3aed;border-radius:2px;animation:wave 0.8s ease-in-out {i*0.1}s infinite alternate"></div>' for i in range(12)])}
    </div>
    <span id="status" style="font-size:12px;color:rgba(255,255,255,0.5)">Click to speak a patient message</span>
  </div>
  <div id="transcript" style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
    border-radius:10px;padding:12px;min-height:44px;font-size:13px;color:#e2e8f0;
    margin-bottom:10px;line-height:1.5;transition:all 0.3s"></div>
  <button onclick="useTranscript()"
    style="background:linear-gradient(135deg,#0f6e56,#1d9e75);color:white;border:none;
    padding:8px 18px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600;
    box-shadow:0 4px 12px rgba(15,110,86,0.4)">
    ✓ Use this text
  </button>
</div>
<style>
@keyframes wave {{0%{{transform:scaleY(0.5)}}100%{{transform:scaleY(1.5)}}}}
</style>
<script>
let recognition=null,isRecording=false,finalTranscript='';
function toggleRecording(){{
  if(!('webkitSpeechRecognition'in window)&&!('SpeechRecognition'in window)){{
    document.getElementById('status').textContent='Use Chrome for voice input';return;
  }}
  isRecording?recognition.stop():startRecording();
}}
function startRecording(){{
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  recognition=new SR();
  recognition.continuous=true;recognition.interimResults=true;recognition.lang='en-US';
  recognition.onstart=()=>{{
    isRecording=true;finalTranscript='';
    document.getElementById('voiceBtn').innerHTML='⏹ Stop Recording';
    document.getElementById('voiceBtn').style.background='linear-gradient(135deg,#dc2626,#b91c1c)';
    document.getElementById('waveform').style.display='flex';
    document.getElementById('status').textContent='Listening... speak now';
  }};
  recognition.onresult=(e)=>{{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){{
      e.results[i].isFinal?finalTranscript+=e.results[i][0].transcript+' ':interim+=e.results[i][0].transcript;
    }}
    document.getElementById('transcript').textContent=finalTranscript+interim;
  }};
  recognition.onerror=(e)=>{{document.getElementById('status').textContent='Error: '+e.error;stopRecording();}};
  recognition.onend=()=>stopRecording();
  recognition.start();
}}
function stopRecording(){{
  isRecording=false;
  document.getElementById('voiceBtn').innerHTML='🎤 Start Recording';
  document.getElementById('voiceBtn').style.background='linear-gradient(135deg,#7c3aed,#5b21b6)';
  document.getElementById('waveform').style.display='none';
  document.getElementById('status').textContent=finalTranscript?'Click "Use this text" to triage':'Ready';
}}
function useTranscript(){{
  const text=document.getElementById('transcript').textContent.trim();
  if(text)window.parent.postMessage({{type:'streamlit:setComponentValue',value:text}},'*');
}}
// TTS function — called from Streamlit via postMessage
window.addEventListener('message',(e)=>{{
  if(e.data&&e.data.type==='speak'){{
    const utt=new SpeechSynthesisUtterance(e.data.text);
    utt.rate={voice_speed};utt.pitch={voice_pitch};
    utt.voice=speechSynthesis.getVoices().find(v=>v.lang==='en-US'&&v.name.includes('Female'))||null;
    speechSynthesis.cancel();speechSynthesis.speak(utt);
  }}
  if(e.data&&e.data.type==='stop_speech'){{
    speechSynthesis.cancel();
  }}
}});
</script>
"""

voice_result = components.html(voice_html, height=160)

col_v1, col_v2 = st.columns([4,1])
with col_v1:
    voice_input = st.text_input("Or type here:", placeholder="Voice transcript or manual input...",
                                 label_visibility="collapsed", key="voice_field")
with col_v2:
    triage_voice = st.button("Triage", type="primary", use_container_width=True)

# ── TTS Output Component ──────────────────────────────────────────────────────
tts_placeholder = st.empty()

def speak_text(text: str):
    """Inject TTS into the voice component."""
    if voice_output_on:
        clean = text.replace("**","").replace("*","").replace("#","").replace("`","")[:300]
        tts_html = f"""
        <script>
        (function(){{
            const utt=new SpeechSynthesisUtterance({json.dumps(clean)});
            utt.rate={voice_speed};utt.pitch={voice_pitch};
            utt.voice=speechSynthesis.getVoices().find(v=>v.lang==='en-US')||null;
            speechSynthesis.cancel();
            setTimeout(()=>speechSynthesis.speak(utt),200);
        }})();
        </script>
        """
        tts_placeholder.components.v1.html(tts_html, height=0)

# ── Chat history ──────────────────────────────────────────────────────────────
st.divider()
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-bubble-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bubble-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("triage"):
            t = msg["triage"]
            risk = t["risk_level"]
            badge_class = f"badge-{risk}"
            icon = RISK_ICONS.get(risk,"⚪")
            review_text = "👤 Human review required" if t["requires_human_review"] else "✅ Auto-routable"
            st.markdown(
                f'<span class="risk-badge {badge_class}">{icon} {risk.upper()}</span>'
                f'<span class="risk-badge" style="background:rgba(255,255,255,0.05);color:rgba(255,255,255,0.6);border:1px solid rgba(255,255,255,0.1)">'
                f'{t["intent"].replace("_"," ")} · {t["intent_confidence"]:.0%} · {review_text}</span>',
                unsafe_allow_html=True,
            )

# ── Input row ─────────────────────────────────────────────────────────────────
st.divider()
col_i, col_b = st.columns([5,1])
with col_i:
    text_input = st.text_input("Message", placeholder="Type or use voice above...",
                               label_visibility="collapsed", key="chat_msg")
with col_b:
    send_btn = st.button("Send", type="primary", use_container_width=True)

# Quick examples
ec = st.columns(4)
examples = [
    "I have chest pain and shortness of breath",
    "I don't want to be here anymore",
    "I ran out of my blood thinner for 5 days",
    "Can I reschedule my appointment?",
]
selected = None
for i,(col,ex) in enumerate(zip(ec,examples)):
    if col.button(ex[:22]+"…", key=f"ex{i}"):
        selected = ex

# ── Process input ─────────────────────────────────────────────────────────────
user_input = None
if send_btn and text_input.strip():
    user_input = text_input.strip()
elif triage_voice and voice_input.strip():
    user_input = voice_input.strip()
elif selected:
    user_input = selected

if user_input:
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.spinner("Analyzing..."):
        try:
            result = pipeline.run(user_input)
            st.session_state.last_triage = result
            response = build_response(user_input, result, st.session_state.messages)
            st.session_state.messages.append({
                "role":"assistant","content":response,
                "triage":{
                    "intent":result.intent,
                    "risk_level":result.risk_level,
                    "intent_confidence":result.intent_confidence,
                    "risk_confidence":result.risk_confidence,
                    "requires_human_review":result.requires_human_review,
                },
            })
            # Speak the response
            if voice_output_on:
                speak_prefix = {
                    "urgent": "Alert. Urgent message detected. ",
                    "high":   "High risk message detected. ",
                    "medium": "",
                    "low":    "",
                }.get(result.risk_level, "")
                speak_text(speak_prefix + response)
        except Exception as e:
            err = f"Pipeline error: {str(e)[:80]}"
            st.session_state.messages.append({"role":"assistant","content":err})
    st.rerun()

# Stop speech button
if voice_output_on and st.session_state.messages:
    if st.button("🔇 Stop speaking"):
        components.html("<script>speechSynthesis.cancel();</script>", height=0)
