"""
ChronicGuard AI — Conversational Chat + Voice I/O
Auto-triage on voice stop. TTS replies. Phamily branded.
"""
import sys, csv, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ChronicGuard AI — Voice Chat", page_icon="💬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.hdr{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:20px;
  padding:28px 36px;margin-bottom:20px;border:1px solid rgba(255,255,255,0.1);
  box-shadow:0 20px 60px rgba(0,0,0,0.5);}
.chat-u{background:linear-gradient(135deg,#1d4ed8,#3b82f6);color:white;
  padding:12px 18px;border-radius:18px 18px 4px 18px;margin:8px 0 8px 20%;
  font-size:14px;line-height:1.5;box-shadow:0 4px 15px rgba(29,78,216,0.3);}
.chat-b{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
  color:#e2e8f0;padding:12px 18px;border-radius:18px 18px 18px 4px;
  margin:8px 20% 8px 0;font-size:14px;line-height:1.5;}
.rb{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:16px;font-size:11px;font-weight:600;margin:3px 2px;}
.b-urgent{background:rgba(220,38,38,.2);color:#fca5a5;border:1px solid rgba(220,38,38,.4);}
.b-high{background:rgba(217,119,6,.2);color:#fcd34d;border:1px solid rgba(217,119,6,.4);}
.b-medium{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.4);}
.b-low{background:rgba(34,197,94,.2);color:#86efac;border:1px solid rgba(34,197,94,.4);}
.speaking-indicator{display:inline-flex;align-items:center;gap:6px;
  background:rgba(124,58,237,0.2);border:1px solid rgba(124,58,237,0.4);
  color:#c4b5fd;padding:4px 12px;border-radius:12px;font-size:12px;}
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
RISK_ICONS = {"urgent":"🔴","high":"🟠","medium":"🔵","low":"🟢"}

def build_response(user_input, triage_result, history):
    openai_key = os.getenv("OPENAI_API_KEY","").strip()
    if not openai_key:
        if triage_result:
            risk = triage_result.risk_level
            intent = triage_result.intent
            icon = RISK_ICONS.get(risk,"⚪")
            action = "Immediate escalation required — do not auto-route." if risk in ("urgent","high") else "Route within standard SLA."
            gaps = ", ".join(triage_result.retrieved_protocols[:2]) if triage_result.retrieved_protocols else "standard protocols"
            return f"{icon} **{intent.replace('_',' ').title()}** — Risk: **{risk.upper()}**\n\n{action}\n\nRelevant protocols retrieved: {len(triage_result.retrieved_protocols)}."
        return "Please set an OpenAI API key for full conversational responses."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        system = (
            "You are ChronicGuard AI, a clinical care management assistant for Phamily. "
            "You help care managers triage patient messages, understand risk levels, and decide next actions. "
            "You can answer general questions about chronic care management, CCM protocols, care gaps, "
            "medication adherence, patient safety, and clinical workflows. "
            "Be concise, warm, accurate, and safety-conscious. Never diagnose or prescribe. "
            "For clinical questions beyond your scope, advise consulting a licensed provider. "
            "Always base clinical answers on standard CCM program guidelines."
        )
        msgs = [{"role":"system","content":system}]
        for h in history[-8:]:
            msgs.append({"role":h["role"],"content":h["content"]})
        if triage_result:
            ctx = (
                f"[LATEST TRIAGE RESULT] "
                f"Intent: {triage_result.intent}, Risk: {triage_result.risk_level}, "
                f"Intent confidence: {triage_result.intent_confidence:.2f}, "
                f"Risk confidence: {triage_result.risk_confidence:.2f}, "
                f"Safety flag: {triage_result.safety_flag}, "
                f"Human review required: {triage_result.requires_human_review}, "
                f"Protocols retrieved: {len(triage_result.retrieved_protocols)}. "
                f"[USER MESSAGE]: {user_input}"
            )
            msgs.append({"role":"user","content":ctx})
        else:
            msgs.append({"role":"user","content":user_input})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=msgs, max_tokens=350, temperature=0.4
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"API error: {str(e)[:60]}. Check your OpenAI key."

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in [("messages",[]),("last_triage",None),("is_speaking",False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChronicGuard AI")
    st.caption("**Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
    st.divider()

    voice_on = st.toggle("🔊 Voice replies", value=True,
                          help="AI reads every response aloud automatically")
    voice_speed = st.slider("Speech speed", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0
    voice_pitch = st.slider("Voice pitch", 0.0, 2.0, 1.1, 0.1) if voice_on else 1.0

    if st.button("🔇 Stop speaking now", use_container_width=True, type="secondary"):
        components.html("<script>speechSynthesis.cancel();</script>", height=0)
        st.session_state.is_speaking = False

    st.divider()

    sim_path = Path("results/outcome_simulation.json")
    if sim_path.exists():
        with open(sim_path) as f: sim = json.load(f)
        st.markdown("**Outcome simulation**")
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

    tl_path = Path("results/risk_timelines.json")
    if tl_path.exists():
        with open(tl_path) as f: tl = json.load(f)
        st.markdown("**Risk timeline**")
        if tl["proactive_outreach_needed"] > 0:
            st.warning(f"⚠️ {tl['proactive_outreach_needed']} patient(s) need outreach")
        for t in tl["timelines"]:
            icon = "📈" if t["trend"]=="deteriorating" else "📉" if t["trend"]=="improving" else "➡️"
            st.caption(f"{icon} {t['name'].split('—')[0].strip()}: {t['current_risk']}")
        st.divider()

    ragas_path = Path("results/ragas_evaluation.json")
    if ragas_path.exists():
        with open(ragas_path) as f: ragas = json.load(f)
        st.markdown("**RAGAS**")
        st.metric("Faithfulness", f"{ragas['avg_faithfulness']:.3f}")
        st.metric("Hallucination", f"{ragas['hallucination_rate']:.0%}",
                  delta="Safe" if ragas["hallucination_rate"]==0 else "Review")
        st.divider()

    if st.session_state.last_triage:
        r = st.session_state.last_triage
        st.markdown("**Last triage**")
        risk = r.risk_level
        st.markdown(f"{RISK_ICONS.get(risk,'⚪')} **{risk.upper()}** · {r.intent.replace('_',' ').title()}")
        st.caption(f"Confidence: {r.intent_confidence:.0%} intent · {r.risk_confidence:.0%} risk")
        if r.safety_flag: st.error("⚠️ Safety flag active")
        if r.requires_human_review: st.warning("👤 Human review required")
        else: st.success("✅ Auto-routable")
        st.caption(f"⚡ {r.total_latency_ms:.0f}ms")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div style="margin-bottom:10px">
    <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);
      color:#a5b4fc;font-size:11px;font-weight:600;padding:4px 12px;border-radius:6px;
      letter-spacing:0.8px">BUILT FOR JAAN HEALTH / PHAMILY CCM</span>
  </div>
  <p style="font-size:28px;font-weight:700;color:white;margin:0">💬 Conversational Triage</p>
  <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:5px 0 2px">
    Voice Input · Auto-Triage · Voice Replies · Safety-First</p>
  <p style="font-size:11px;color:rgba(255,255,255,0.3);margin:0">
    Built by Akila Lourdes Miriyala Francis &amp; Akilan Manivannan</p>
</div>
""", unsafe_allow_html=True)

col_top, col_clear = st.columns([5,1])
with col_clear:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.rerun()

# ── VOICE INPUT ────────────────────────────────────────────────────────────────
st.markdown("### 🎤 Voice Input — Auto Triage")
st.caption("Speak → transcript auto-fills → auto-triages. No copy-paste needed.")

if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""
if "voice_auto_send" not in st.session_state:
    st.session_state.voice_auto_send = False

# Check query params for voice transcript (set by JS)
qp = st.query_params
if "vt" in qp and qp["vt"].strip():
    incoming = qp["vt"].strip()
    if incoming != st.session_state.get("last_voice_qp", ""):
        st.session_state.voice_text = incoming
        st.session_state.voice_auto_send = True
        st.session_state.last_voice_qp = incoming
        st.query_params.clear()

components.html(f"""
<div style="font-family:Inter,sans-serif;padding:2px 0">
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px;flex-wrap:wrap">
    <button id="vBtn" onclick="toggleRec()"
      style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:white;border:none;
      padding:10px 22px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:600;
      box-shadow:0 4px 15px rgba(124,58,237,0.4)">
      🎤 Start Recording
    </button>
    <div id="wave" style="display:none;align-items:center;gap:3px">
      <div style="width:3px;height:6px;background:#a78bfa;border-radius:2px;animation:wv 0.5s ease infinite alternate"></div>
      <div style="width:3px;height:14px;background:#a78bfa;border-radius:2px;animation:wv 0.5s 0.1s ease infinite alternate"></div>
      <div style="width:3px;height:22px;background:#a78bfa;border-radius:2px;animation:wv 0.5s 0.2s ease infinite alternate"></div>
      <div style="width:3px;height:14px;background:#a78bfa;border-radius:2px;animation:wv 0.5s 0.3s ease infinite alternate"></div>
      <div style="width:3px;height:6px;background:#a78bfa;border-radius:2px;animation:wv 0.5s 0.4s ease infinite alternate"></div>
      <span style="color:#a78bfa;font-size:12px;margin-left:6px">Listening...</span>
    </div>
    <span id="st" style="font-size:12px;color:#94a3b8">Use Chrome · click to begin</span>
  </div>
  <div id="box" style="background:#1e1b4b;border:1px solid #4338ca;border-radius:10px;
    padding:10px 14px;min-height:40px;font-size:13px;color:#c4b5fd;line-height:1.5">
    <span style="opacity:0.4">Spoken words appear here automatically...</span>
  </div>
</div>
<style>@keyframes wv{{0%{{transform:scaleY(0.3)}}100%{{transform:scaleY(1.6)}}}}</style>
<script>
let rec=null,going=false,final_t='';
function toggleRec(){{
  if(!('webkitSpeechRecognition'in window||'SpeechRecognition'in window)){{
    document.getElementById('st').textContent='Use Chrome for voice input';return;
  }}
  going?rec.stop():startRec();
}}
function startRec(){{
  const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  rec=new SR();rec.continuous=true;rec.interimResults=true;rec.lang='en-US';
  rec.onstart=()=>{{
    going=true;final_t='';
    document.getElementById('vBtn').innerHTML='⏹ Stop Recording';
    document.getElementById('vBtn').style.background='linear-gradient(135deg,#dc2626,#b91c1c)';
    document.getElementById('wave').style.display='flex';
    document.getElementById('st').textContent='Listening...';
    document.getElementById('box').innerHTML='<span style="color:#a78bfa">Listening...</span>';
  }};
  rec.onresult=(e)=>{{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){{
      e.results[i].isFinal?final_t+=e.results[i][0].transcript+' ':interim+=e.results[i][0].transcript;
    }}
    document.getElementById('box').textContent=final_t+interim;
  }};
  rec.onerror=(e)=>{{document.getElementById('st').textContent='Error: '+e.error+'. Try Chrome.';stopRec();}};
  rec.onend=()=>stopRec();
  rec.start();
}}
function stopRec(){{
  going=false;
  document.getElementById('vBtn').innerHTML='🎤 Start Recording';
  document.getElementById('vBtn').style.background='linear-gradient(135deg,#7c3aed,#5b21b6)';
  document.getElementById('wave').style.display='none';
  const text=final_t.trim();
  if(text){{
    document.getElementById('box').textContent=text;
    document.getElementById('st').textContent='Submitting...';
    // Navigate parent to inject query param — triggers Streamlit rerun
    const url=new URL(window.parent.location.href);
    url.searchParams.set('vt',text);
    window.parent.location.href=url.toString();
  }} else {{
    document.getElementById('st').textContent='No speech detected. Try again.';
  }}
}}
</script>
""", height=140)

# ── Input section ─────────────────────────────────────────────────────────────
st.divider()
col_i, col_s = st.columns([5,1])
with col_i:
    manual_msg = st.text_input(
        "Or type any message or question:",
        placeholder="Patient triage, protocol questions, follow-ups, general CCM questions...",
        label_visibility="collapsed",
        key="manual_msg",
    )
with col_s:
    send_btn = st.button("Send", type="primary", use_container_width=True)

send_voice_btn = False
voice_transcript = st.session_state.voice_text

# Quick examples
st.markdown("**Quick examples:**")
ec = st.columns(4)
examples = [
    ("🫀", "I have chest pain and shortness of breath"),
    ("🆘", "I don't want to be here anymore"),
    ("💊", "I ran out of my blood thinner for 5 days"),
    ("📅", "Can I reschedule my appointment?"),
]
selected = None
for i,(col,(emoji,ex)) in enumerate(zip(ec,examples)):
    if col.button(f"{emoji} {ex[:20]}…", key=f"ex{i}"):
        selected = ex

# ── Speaking indicator ────────────────────────────────────────────────────────
speak_slot = st.empty()
if st.session_state.is_speaking and voice_on:
    speak_slot.markdown(
        '<div class="speaking-indicator">🔊 Speaking... <button onclick="speechSynthesis.cancel()" '
        'style="background:transparent;border:1px solid #c4b5fd;color:#c4b5fd;border-radius:4px;'
        'padding:1px 6px;cursor:pointer;font-size:10px">Stop</button></div>',
        unsafe_allow_html=True
    )

# ── Chat history ──────────────────────────────────────────────────────────────
st.divider()
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-u">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-b">{msg["content"]}</div>', unsafe_allow_html=True)
        if msg.get("triage"):
            t = msg["triage"]
            risk = t["risk_level"]
            icon = RISK_ICONS.get(risk,"⚪")
            review = "👤 Human review" if t["requires_human_review"] else "✅ Auto-routable"
            st.markdown(
                f'<span class="rb b-{risk}">{icon} {risk.upper()}</span>'
                f'<span class="rb" style="background:rgba(255,255,255,0.04);'
                f'color:rgba(255,255,255,0.5);border:1px solid rgba(255,255,255,0.1)">'
                f'{t["intent"].replace("_"," ")} · {t["intent_confidence"]:.0%} · {review}</span>',
                unsafe_allow_html=True,
            )

# ── Process input ─────────────────────────────────────────────────────────────
user_input = None
# Auto-send from voice query param
if st.session_state.voice_auto_send and st.session_state.voice_text.strip():
    user_input = st.session_state.voice_text.strip()
    st.session_state.voice_text = ""
    st.session_state.voice_auto_send = False
elif send_btn and manual_msg.strip():
    user_input = manual_msg.strip()
elif selected:
    user_input = selected

if user_input and not st.session_state.get("processing", False):
    st.session_state.processing = True
    st.session_state.messages.append({"role":"user","content":user_input})
    with st.spinner("Analyzing..."):
        try:
            # Run triage pipeline
            result = pipeline.run(user_input)
            st.session_state.last_triage = result
            # Build response
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
            # TTS — inject after response
            if voice_on:
                prefix = {
                    "urgent": "Alert. Urgent message detected. ",
                    "high":   "High risk. ",
                }.get(result.risk_level, "")
                clean = (prefix + response).replace("**","").replace("*","").replace("`","")[:400]
                tts = f"""<script>
(function(){{
  window.speechSynthesis.cancel();
  const u=new SpeechSynthesisUtterance({json.dumps(clean)});
  u.rate={voice_speed};u.pitch={voice_pitch};u.lang='en-US';
  u.onstart=()=>{{try{{window.parent.postMessage({{type:'speaking',value:true}},'*')}}catch(e){{}}}};
  u.onend=()=>{{try{{window.parent.postMessage({{type:'speaking',value:false}},'*')}}catch(e){{}}}};
  setTimeout(()=>window.speechSynthesis.speak(u),400);
}})();
</script>"""
                components.html(tts, height=0)
                st.session_state.is_speaking = True
        except Exception as e:
            st.session_state.messages.append({
                "role":"assistant",
                "content":f"Pipeline error: {str(e)[:100]}"
            })
    st.session_state.processing = False
    st.rerun()

# ── Suggestions box ───────────────────────────────────────────────────────────
with st.expander("💡 What else can you ask?"):
    st.markdown("""
**Patient triage:**
- "I have chest pain and shortness of breath"
- "My blood sugar has been over 300 for two days"
- "I don't want to be here anymore"

**Protocol questions:**
- "What should I do for a patient with missed blood pressure meds?"
- "What is the protocol for a prior authorization denial?"
- "When should I escalate to the provider?"

**Follow-up questions after triage:**
- "What should the care manager do first?"
- "Why was this flagged as urgent?"
- "What care gaps does this message suggest?"

**General CCM questions:**
- "What is a care gap?"
- "What does urgent recall mean in this system?"
- "How does the HITL gate work?"
- "What is the 988 crisis line?"
- "Explain the RAGAS evaluation results"
""")
