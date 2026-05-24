"""
ChronicGuard AI — Conversational Chat + Voice I/O
Professional glassmorphism UI with Phamily branding.
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
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.main-header{
    background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    border-radius:20px;padding:28px 36px;margin-bottom:20px;
    border:1px solid rgba(255,255,255,0.1);
    box-shadow:0 20px 60px rgba(0,0,0,0.5),inset 0 1px 0 rgba(255,255,255,0.1);
}
.phamily-badge{
    display:inline-flex;align-items:center;gap:8px;
    background:white;border-radius:10px;padding:6px 14px;margin-bottom:12px;
}
.main-title{font-size:30px;font-weight:700;color:white;margin:0;letter-spacing:-0.5px;}
.main-sub{font-size:13px;color:rgba(255,255,255,0.55);margin:5px 0 2px 0;}
.authors{font-size:11px;color:rgba(255,255,255,0.35);margin:0;}
.chat-user{
    background:linear-gradient(135deg,#1d4ed8,#3b82f6);color:white;
    padding:12px 18px;border-radius:18px 18px 4px 18px;
    margin:8px 0 8px 20%;font-size:14px;line-height:1.5;
    box-shadow:0 4px 15px rgba(29,78,216,0.35);
}
.chat-bot{
    background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
    color:#e2e8f0;padding:12px 18px;border-radius:18px 18px 18px 4px;
    margin:8px 20% 8px 0;font-size:14px;line-height:1.5;
}
.rbadge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
    border-radius:16px;font-size:11px;font-weight:600;margin:3px 2px;}
.b-urgent{background:rgba(220,38,38,.2);color:#fca5a5;border:1px solid rgba(220,38,38,.4);}
.b-high{background:rgba(217,119,6,.2);color:#fcd34d;border:1px solid rgba(217,119,6,.4);}
.b-medium{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.4);}
.b-low{background:rgba(34,197,94,.2);color:#86efac;border:1px solid rgba(34,197,94,.4);}
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
        action = "Immediate escalation required." if risk in ("urgent","high") else "Route within standard SLA."
        return f"{icon} Classified as **{intent.replace('_',' ')}** with **{risk}** risk. {action}"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        msgs = [{"role":"system","content":"You are ChronicGuard AI, a care management assistant. Be concise, warm, safety-conscious. Never diagnose or prescribe."}]
        for h in history[-6:]:
            msgs.append({"role":h["role"],"content":h["content"]})
        if triage_result:
            ctx = f"[TRIAGE] Intent:{triage_result.intent}, Risk:{triage_result.risk_level}, Safety:{triage_result.safety_flag}"
            msgs[-1]["content"] = ctx + "\nUser: " + user_input
        else:
            msgs.append({"role":"user","content":user_input})
        resp = client.chat.completions.create(model="gpt-4o-mini",messages=msgs,max_tokens=250,temperature=0.4)
        return resp.choices[0].message.content
    except Exception:
        risk = triage_result.risk_level if triage_result else "unknown"
        return f"Risk: **{risk}**. {'Escalate immediately.' if risk in ('urgent','high') else 'Route to care manager.'}"

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_triage" not in st.session_state:
    st.session_state.last_triage = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChronicGuard AI")
    st.caption("Built by **Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
    voice_on = st.toggle("🔊 Voice replies", value=True)
    voice_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0
    voice_pitch = st.slider("Pitch", 0.0, 2.0, 1.0, 0.1) if voice_on else 1.0
    if st.button("🔇 Stop speaking", use_container_width=True):
        components.html("<script>speechSynthesis.cancel();</script>", height=0)
    st.divider()

    sim_path = Path("results/outcome_simulation.json")
    if sim_path.exists():
        with open(sim_path) as f:
            sim = json.load(f)
        st.markdown("**Outcome simulation**")
        st.metric("Urgent response", f"{sim['response_time']['ai_urgent_median_hours']*60:.0f} min",
                  delta=f"vs {sim['response_time']['manual_urgent_median_hours']:.1f}h")
        st.metric("Care gap closure", f"{sim['care_gaps']['ai_closure_rate']*100:.0f}%",
                  delta=f"+{sim['care_gaps']['improvement_pct']:.0f}%")
        st.metric("CM capacity", f"{sim['efficiency']['ai_cm_capacity_per_day']} msg/day",
                  delta=f"+{sim['efficiency']['capacity_increase_pct']:.0f}%")
        st.divider()

    tl_path = Path("results/risk_timelines.json")
    if tl_path.exists():
        with open(tl_path) as f:
            tl = json.load(f)
        st.markdown("**Risk timeline**")
        if tl["proactive_outreach_needed"] > 0:
            st.warning(f"⚠️ {tl['proactive_outreach_needed']} need outreach")
        for t in tl["timelines"]:
            icon = "📈" if t["trend"]=="deteriorating" else "📉" if t["trend"]=="improving" else "➡️"
            st.caption(f"{icon} {t['name'].split('—')[0].strip()}: {t['current_risk']}")
        st.divider()

    ragas_path = Path("results/ragas_evaluation.json")
    if ragas_path.exists():
        with open(ragas_path) as f:
            ragas = json.load(f)
        st.markdown("**RAGAS evaluation**")
        st.metric("Faithfulness", f"{ragas['avg_faithfulness']:.3f}")
        st.metric("Hallucination rate", f"{ragas['hallucination_rate']:.0%}",
                  delta="Safe" if ragas["hallucination_rate"]==0 else "Review")
        st.divider()

    if st.session_state.last_triage:
        r = st.session_state.last_triage
        st.markdown("**Last triage**")
        risk = r.risk_level
        st.markdown(f"{RISK_ICONS.get(risk,'⚪')} **{risk.upper()}** · {r.intent.replace('_',' ').title()}")
        if r.safety_flag: st.error("⚠️ Safety flag")
        if r.requires_human_review: st.warning("👤 Human review")
        else: st.success("✅ Auto-routable")
        st.caption(f"⚡ {r.total_latency_ms:.0f}ms")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <div class="phamily-badge">
    <svg width="32" height="26" viewBox="0 0 100 70">
      <circle cx="50" cy="10" r="8" fill="#6366f1"/>
      <circle cx="28" cy="18" r="7" fill="#6366f1"/>
      <circle cx="72" cy="18" r="7" fill="#6366f1"/>
      <rect x="43" y="19" width="14" height="24" rx="7" fill="#6366f1"/>
      <rect x="21" y="26" width="13" height="22" rx="6.5" fill="#6366f1"/>
      <rect x="66" y="26" width="13" height="22" rx="6.5" fill="#6366f1"/>
      <path d="M5 58 Q50 72 95 58" stroke="#10b981" stroke-width="5" fill="none" stroke-linecap="round"/>
    </svg>
    <span style="color:#6366f1;font-size:16px;font-weight:700">Phamily</span>
  </div>
  <p class="main-title">💬 ChronicGuard AI — Conversational Triage</p>
  <p class="main-sub">Voice Input · Voice Replies · Safety-First · Real-time Risk Classification</p>
  <p class="authors">Built by Akila Lourdes Miriyala Francis &amp; Akilan Manivannan</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5,1])
with col2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.rerun()

# ── Voice section ─────────────────────────────────────────────────────────────
st.markdown("### 🎤 Voice Input")
st.caption("Speak a patient message — the transcript appears below. Then click **Triage Voice Message**.")

VOICE_KEY = "voice_transcript_v2"
if VOICE_KEY not in st.session_state:
    st.session_state[VOICE_KEY] = ""

# Voice component — writes transcript to a hidden text area via JS
voice_component = components.html("""
<div style="font-family:Inter,sans-serif">
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
    <button id="vBtn" onclick="toggleRec()"
      style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:white;border:none;
      padding:10px 20px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:600;
      box-shadow:0 4px 15px rgba(124,58,237,0.4);letter-spacing:0.2px">
      🎤 Start Recording
    </button>
    <div id="wave" style="display:none;gap:3px;align-items:center">
      <div style="width:3px;height:8px;background:#7c3aed;border-radius:2px;animation:w 0.6s ease infinite alternate"></div>
      <div style="width:3px;height:14px;background:#7c3aed;border-radius:2px;animation:w 0.6s ease 0.1s infinite alternate"></div>
      <div style="width:3px;height:20px;background:#7c3aed;border-radius:2px;animation:w 0.6s ease 0.2s infinite alternate"></div>
      <div style="width:3px;height:14px;background:#7c3aed;border-radius:2px;animation:w 0.6s ease 0.3s infinite alternate"></div>
      <div style="width:3px;height:8px;background:#7c3aed;border-radius:2px;animation:w 0.6s ease 0.4s infinite alternate"></div>
    </div>
    <span id="status" style="font-size:12px;color:#94a3b8">Use Chrome · Click to begin</span>
  </div>
  <textarea id="tbox" readonly rows="2"
    style="width:100%;background:#1e1b4b;border:1px solid #4338ca;border-radius:10px;
    padding:10px 14px;font-size:13px;color:#e0e7ff;resize:none;box-sizing:border-box;
    font-family:Inter,sans-serif;outline:none"
    placeholder="Your spoken words appear here..."></textarea>
  <div style="display:flex;gap:8px;margin-top:8px">
    <button onclick="copyText()"
      style="background:linear-gradient(135deg,#0f6e56,#1d9e75);color:white;border:none;
      padding:8px 16px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600;
      box-shadow:0 3px 10px rgba(15,110,86,0.4)">
      📋 Copy transcript
    </button>
    <span id="copied" style="font-size:12px;color:#86efac;display:none;align-self:center">Copied!</span>
  </div>
</div>
<style>
@keyframes w{0%{transform:scaleY(0.4)}100%{transform:scaleY(1.4)}}
</style>
<script>
let rec=null,going=false,final_t='';
function toggleRec(){
  if(!('webkitSpeechRecognition'in window||'SpeechRecognition'in window)){
    document.getElementById('status').textContent='Use Chrome for voice input';return;
  }
  going?rec.stop():startRec();
}
function startRec(){
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  rec=new SR();rec.continuous=true;rec.interimResults=true;rec.lang='en-US';
  rec.onstart=()=>{
    going=true;final_t='';
    document.getElementById('vBtn').innerHTML='⏹ Stop Recording';
    document.getElementById('vBtn').style.background='linear-gradient(135deg,#dc2626,#b91c1c)';
    document.getElementById('wave').style.display='flex';
    document.getElementById('status').textContent='Listening...';
  };
  rec.onresult=(e)=>{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      e.results[i].isFinal?final_t+=e.results[i][0].transcript+' ':interim+=e.results[i][0].transcript;
    }
    document.getElementById('tbox').value=final_t+interim;
  };
  rec.onerror=(e)=>{document.getElementById('status').textContent='Error: '+e.error;stopRec();};
  rec.onend=stopRec;
  rec.start();
}
function stopRec(){
  going=false;
  document.getElementById('vBtn').innerHTML='🎤 Start Recording';
  document.getElementById('vBtn').style.background='linear-gradient(135deg,#7c3aed,#5b21b6)';
  document.getElementById('wave').style.display='none';
  document.getElementById('status').textContent=final_t?'Done — copy and paste below':'Ready';
}
function copyText(){
  const t=document.getElementById('tbox').value.trim();
  if(!t)return;
  navigator.clipboard.writeText(t).then(()=>{
    document.getElementById('copied').style.display='inline';
    setTimeout(()=>document.getElementById('copied').style.display='none',2000);
  });
}
</script>
""", height=190)

# Input that the user pastes transcript into
col_v1, col_v2 = st.columns([5,1])
with col_v1:
    voice_text = st.text_input(
        "Paste transcript or type message:",
        placeholder="Paste copied transcript here, or type directly...",
        key="voice_paste_field",
    )
with col_v2:
    triage_voice = st.button("Triage ▶", type="primary", use_container_width=True)

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-bot">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("triage"):
            t = msg["triage"]
            risk = t["risk_level"]
            icon = RISK_ICONS.get(risk,"⚪")
            review = "👤 Human review" if t["requires_human_review"] else "✅ Auto-routable"
            st.markdown(
                f'<span class="rbadge b-{risk}">{icon} {risk.upper()}</span>'
                f'<span class="rbadge" style="background:rgba(255,255,255,0.05);color:rgba(255,255,255,0.55);border:1px solid rgba(255,255,255,0.1)">'
                f'{t["intent"].replace("_"," ")} · {t["intent_confidence"]:.0%} · {review}</span>',
                unsafe_allow_html=True,
            )

# ── Chat input ────────────────────────────────────────────────────────────────
st.divider()
col_i, col_b = st.columns([5,1])
with col_i:
    text_msg = st.text_input("Message", placeholder="Type a message or follow-up question...",
                              label_visibility="collapsed", key="chat_msg")
with col_b:
    send_btn = st.button("Send", type="primary", use_container_width=True)

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

# ── Process ───────────────────────────────────────────────────────────────────
user_input = None
if triage_voice and voice_text.strip():
    user_input = voice_text.strip()
elif send_btn and text_msg.strip():
    user_input = text_msg.strip()
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
                    "intent":result.intent,"risk_level":result.risk_level,
                    "intent_confidence":result.intent_confidence,
                    "risk_confidence":result.risk_confidence,
                    "requires_human_review":result.requires_human_review,
                },
            })
            # TTS
            if voice_on:
                prefix = {"urgent":"Alert. Urgent message. ","high":"High risk message. "}.get(result.risk_level,"")
                clean = (prefix + response).replace("**","").replace("*","").replace("`","")[:300]
                tts_js = f"""
                <script>
                (function(){{
                  const u=new SpeechSynthesisUtterance({json.dumps(clean)});
                  u.rate={voice_speed};u.pitch={voice_pitch};
                  speechSynthesis.cancel();
                  setTimeout(()=>speechSynthesis.speak(u),300);
                }})();
                </script>"""
                components.html(tts_js, height=0)
        except Exception as e:
            st.session_state.messages.append({"role":"assistant","content":f"Error: {str(e)[:80]}"})
    st.rerun()
