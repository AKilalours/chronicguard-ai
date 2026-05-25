"""
ChronicGuard AI — Conversational Chat
Voice input via browser + auto-triage via Streamlit form submission
"""
import sys, csv, os, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="ChronicGuard AI — Chat", page_icon="💬", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.hdr{background:linear-gradient(135deg,#0f0c29,#302b63,#24243e);border-radius:16px;
  padding:24px 32px;margin-bottom:16px;border:1px solid rgba(255,255,255,0.1);}
.chat-u{background:linear-gradient(135deg,#1d4ed8,#3b82f6);color:white;
  padding:12px 18px;border-radius:18px 18px 4px 18px;
  margin:8px 0 8px 20%;font-size:14px;line-height:1.5;}
.chat-b{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);
  color:#e2e8f0;padding:12px 18px;border-radius:18px 18px 18px 4px;
  margin:8px 20% 8px 0;font-size:14px;line-height:1.5;}
.rb{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;
  border-radius:16px;font-size:11px;font-weight:600;margin:3px 2px;}
.b-urgent{background:rgba(220,38,38,.2);color:#fca5a5;border:1px solid rgba(220,38,38,.4);}
.b-high{background:rgba(217,119,6,.2);color:#fcd34d;border:1px solid rgba(217,119,6,.4);}
.b-medium{background:rgba(59,130,246,.2);color:#93c5fd;border:1px solid rgba(59,130,246,.4);}
.b-low{background:rgba(34,197,94,.2);color:#86efac;border:1px solid rgba(34,197,94,.4);}
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
            action = "Immediate escalation required." if risk in ("urgent","high") else "Route within standard SLA."
            return f"{icon} **{intent.replace('_',' ').title()}** — Risk: **{risk.upper()}**. {action}"
        return "Set OPENAI_API_KEY for full responses."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        system = (
            "You are ChronicGuard AI, a clinical care management assistant for Phamily/Jaan Health. "
            "Answer questions about patient triage, CCM protocols, care gaps, medication adherence, "
            "clinical workflows, and this AI system. Be concise, warm, accurate, safety-conscious. "
            "Never diagnose or prescribe."
        )
        msgs = [{"role":"system","content":system}]
        for h in history[-8:]:
            msgs.append({"role":h["role"],"content":h["content"]})
        if triage_result:
            ctx = (f"[TRIAGE] Intent:{triage_result.intent}, Risk:{triage_result.risk_level}, "
                   f"Safety:{triage_result.safety_flag}, Review:{triage_result.requires_human_review}")
            msgs.append({"role":"user","content":ctx+"\nUser: "+user_input})
        else:
            msgs.append({"role":"user","content":user_input})
        resp = client.chat.completions.create(
            model="gpt-4o-mini", messages=msgs, max_tokens=300, temperature=0.4)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)[:60]}"

def speak(text, speed=1.0):
    clean = text.replace("**","").replace("*","").replace("`","")[:400]
    components.html(f"""
<script>
(function(){{
  function doSpeak(){{
    window.speechSynthesis.cancel();
    const u=new SpeechSynthesisUtterance({json.dumps(clean)});
    u.rate={speed};u.pitch=1.0;u.lang='en-US';
    window.speechSynthesis.speak(u);
  }}
  if(window.speechSynthesis.getVoices().length>0){{doSpeak();}}
  else{{window.speechSynthesis.onvoiceschanged=doSpeak;setTimeout(doSpeak,600);}}
}})();
</script>
<div style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);
  border-radius:8px;padding:4px 12px;display:inline-flex;align-items:center;gap:6px;
  font-size:11px;color:#a78bfa;font-family:Inter,sans-serif">
  🔊 Speaking response...
  <button onclick="window.speechSynthesis.cancel();this.parentElement.style.display='none'"
    style="background:rgba(220,38,38,0.3);border:none;color:#fca5a5;border-radius:4px;
    padding:1px 6px;cursor:pointer;font-size:10px">Stop</button>
</div>
""", height=40)

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in [("messages",[]),("last_triage",None),("processing",False)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChronicGuard AI")
    st.caption("**Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
    voice_on = st.toggle("🔊 Voice replies", value=True)
    voice_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0
    if st.button("🔇 Stop speaking", use_container_width=True):
        components.html("<script>window.speechSynthesis.cancel();</script>", height=0)
    st.divider()
    sim_path = Path("results/outcome_simulation.json")
    if sim_path.exists():
        with open(sim_path) as f: sim = json.load(f)
        st.markdown("**Outcome simulation**")
        st.metric("Urgent response", f"{sim['response_time']['ai_urgent_median_hours']*60:.0f} min",
                  delta=f"vs {sim['response_time']['manual_urgent_median_hours']:.1f}h")
        st.metric("Care gaps", f"{sim['care_gaps']['ai_closure_rate']*100:.0f}%",
                  delta=f"+{sim['care_gaps']['improvement_pct']:.0f}%")
        st.metric("CM capacity", f"{sim['efficiency']['ai_cm_capacity_per_day']} msg/day",
                  delta=f"+{sim['efficiency']['capacity_increase_pct']:.0f}%")
        st.divider()
    if st.session_state.last_triage:
        r = st.session_state.last_triage
        st.markdown("**Last triage**")
        st.markdown(f"{RISK_ICONS.get(r.risk_level,'⚪')} **{r.risk_level.upper()}** · {r.intent.replace('_',' ').title()}")
        if r.safety_flag: st.error("⚠️ Safety flag")
        if r.requires_human_review: st.warning("👤 Human review")
        else: st.success("✅ Auto-routable")
        st.caption(f"⚡ {r.total_latency_ms:.0f}ms")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hdr">
  <div style="margin-bottom:10px">
    <span style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.4);
      color:#a5b4fc;font-size:11px;font-weight:600;padding:4px 12px;border-radius:6px">
      BUILT FOR JAAN HEALTH / PHAMILY CCM
    </span>
  </div>
  <p style="font-size:26px;font-weight:700;color:white;margin:0">💬 Conversational Triage</p>
  <p style="font-size:13px;color:rgba(255,255,255,0.5);margin:4px 0 2px">
    Voice Input · Voice Replies · Safety-First · Real-time Risk Classification</p>
  <p style="font-size:11px;color:rgba(255,255,255,0.3);margin:0">
    Built by Akila Lourdes Miriyala Francis &amp; Akilan Manivannan</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([5,1])
with col2:
    if st.button("🗑 Clear", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_triage = None
        st.rerun()

# ── VOICE INPUT — using st.form for reliable submission ───────────────────────
st.markdown("### 🎤 Voice Input")

# The voice component ONLY shows the transcript visually
# Streamlit can't be auto-triggered from JS — so we use a form
components.html("""
<div style="font-family:Inter,sans-serif;margin-bottom:4px">
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
    <button id="vBtn" onclick="toggleRec()"
      style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:white;border:none;
      padding:9px 20px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:600;
      box-shadow:0 4px 12px rgba(124,58,237,0.4)">
      🎤 Start Recording
    </button>
    <div id="wave" style="display:none;align-items:center;gap:2px">
      <div style="width:3px;height:8px;background:#a78bfa;border-radius:2px;animation:wv .5s ease infinite alternate"></div>
      <div style="width:3px;height:16px;background:#a78bfa;border-radius:2px;animation:wv .5s .1s ease infinite alternate"></div>
      <div style="width:3px;height:22px;background:#a78bfa;border-radius:2px;animation:wv .5s .2s ease infinite alternate"></div>
      <div style="width:3px;height:16px;background:#a78bfa;border-radius:2px;animation:wv .5s .3s ease infinite alternate"></div>
      <div style="width:3px;height:8px;background:#a78bfa;border-radius:2px;animation:wv .5s .4s ease infinite alternate"></div>
    </div>
    <span id="st" style="font-size:12px;color:#94a3b8">Chrome only · speak and stop</span>
  </div>
  <div id="box" style="background:#1e1b4b;border:1.5px solid #4338ca;border-radius:10px;
    padding:10px 14px;min-height:42px;font-size:13px;color:#c4b5fd;line-height:1.5">
    <span id="ph" style="opacity:0.4">Your spoken words appear here...</span>
    <span id="tr"></span>
  </div>
  <p style="font-size:11px;color:#64748b;margin:6px 0 0">
    After stopping, the transcript auto-fills the input below. Then press Enter or click Send.
  </p>
</div>
<style>@keyframes wv{0%{transform:scaleY(0.3)}100%{transform:scaleY(1.6)}}</style>
<script>
let rec=null,going=false,final_t='';
function toggleRec(){
  if(!('webkitSpeechRecognition'in window||'SpeechRecognition'in window)){
    document.getElementById('st').textContent='Use Chrome for voice input';return;
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
    document.getElementById('ph').style.display='none';
    document.getElementById('tr').textContent='Listening...';
    document.getElementById('st').textContent='Speaking...';
  };
  rec.onresult=(e)=>{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      e.results[i].isFinal?final_t+=e.results[i][0].transcript+' ':interim+=e.results[i][0].transcript;
    }
    document.getElementById('tr').textContent=final_t+interim;
  };
  rec.onerror=(e)=>{document.getElementById('st').textContent='Error: '+e.error;stopRec();};
  rec.onend=stopRec;
  rec.start();
}
function stopRec(){
  going=false;
  document.getElementById('vBtn').innerHTML='🎤 Start Recording';
  document.getElementById('vBtn').style.background='linear-gradient(135deg,#7c3aed,#5b21b6)';
  document.getElementById('wave').style.display='none';
  const text=final_t.trim();
  if(text){
    document.getElementById('tr').textContent=text;
    document.getElementById('st').textContent='Done! Filling input below...';
    // Fill ALL text inputs in parent frame
    try{
      const allInputs=[...window.parent.document.querySelectorAll('input[type=text],input:not([type])')];
      const target=allInputs.filter(i=>i.offsetParent!==null).pop();
      if(target){
        Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(target,text);
        target.dispatchEvent(new Event('input',{bubbles:true}));
        target.focus();
        document.getElementById('st').textContent='✓ Filled! Press Enter to send';
      }
    }catch(e){
      document.getElementById('st').textContent='Done! Type or paste: '+text.substring(0,30)+'...';
    }
  } else {
    document.getElementById('st').textContent='Nothing heard. Try again.';
    document.getElementById('ph').style.display='inline';
    document.getElementById('tr').textContent='';
  }
}
</script>
""", height=160)

# ── Single input box — works for both voice and text ─────────────────────────
st.divider()

with st.form(key="msg_form", clear_on_submit=True):
    fc1, fc2 = st.columns([6,1])
    with fc1:
        msg_input = st.text_input(
            "Message",
            placeholder="Speak above (auto-fills here) or type any question...",
            label_visibility="collapsed",
            key="msg_field",
        )
    with fc2:
        submitted = st.form_submit_button("Send ▶", type="primary", use_container_width=True)

# Quick examples outside form
st.markdown("**Quick examples:**")
ec = st.columns(4)
examples = [
    ("🫀","I have chest pain and shortness of breath"),
    ("🆘","I don't want to be here anymore"),
    ("💊","I ran out of my blood thinner for 5 days"),
    ("📅","What is a care gap?"),
]
selected = None
for i,(col,(emoji,ex)) in enumerate(zip(ec,examples)):
    if col.button(f"{emoji} {ex[:20]}…", key=f"ex{i}"):
        selected = ex

# ── TTS slot — always visible, ensures script executes ───────────────────────
tts_slot = st.empty()

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

with st.expander("💡 What can you ask?"):
    st.markdown("""
**Triage:** "chest pain", "blood sugar 320", "don't want to be here", "ran out of blood thinner"

**Follow-ups:** "What should care manager do first?", "Why urgent?", "What protocols apply?"

**CCM questions:** "What is a care gap?", "What is prior authorization?", "When to call 911?"

**System:** "How does HITL work?", "What is urgent recall?", "Explain RAGAS", "What is LoRA?"
    """)

# ── Process ───────────────────────────────────────────────────────────────────
user_input = None
if submitted and msg_input.strip():
    user_input = msg_input.strip()
elif selected:
    user_input = selected

if user_input and not st.session_state.processing:
    st.session_state.processing = True
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
            if voice_on:
                prefix = {"urgent":"Alert. Urgent. ","high":"High risk. "}.get(result.risk_level,"")
                speak(prefix + response, voice_speed)
        except Exception as e:
            st.session_state.messages.append({
                "role":"assistant","content":f"Error: {str(e)[:80]}"})
    st.session_state.processing = False
    st.rerun()
