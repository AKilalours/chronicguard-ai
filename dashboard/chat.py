"""
ChronicGuard AI — Conversational Chat + Voice
Simple reliable voice: speak → see transcript → click Send Voice
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
        return "Set OPENAI_API_KEY for full conversational responses."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        system = (
            "You are ChronicGuard AI, a clinical care management assistant for Phamily/Jaan Health. "
            "Answer questions about patient triage, CCM protocols, care gaps, medication adherence, "
            "clinical workflows, and this AI system. Be concise, warm, accurate, safety-conscious. "
            "Never diagnose or prescribe. For clinical questions beyond scope, advise consulting a provider."
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

# ── Session state ─────────────────────────────────────────────────────────────
for k,v in [("messages",[]),("last_triage",None),("pending_voice","")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ChronicGuard AI")
    st.caption("**Akila Lourdes Miriyala Francis** & **Akilan Manivannan**")
    voice_on = st.toggle("🔊 Voice replies", value=True)
    voice_speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1) if voice_on else 1.0
    if st.button("🔇 Stop speaking", use_container_width=True):
        components.html("<script>speechSynthesis.cancel();</script>", height=0)
    st.divider()
    for path, label, keys in [
        ("results/outcome_simulation.json", "Outcome simulation",
         [("Urgent response","response_time","ai_urgent_median_hours",60,"min",
           "manual_urgent_median_hours","h manual"),
          ("Care gap closure","care_gaps","ai_closure_rate",100,"%",
           "improvement_pct","% gain"),
          ("CM capacity","efficiency","ai_cm_capacity_per_day",1,"msg/day",
           "capacity_increase_pct","% gain")]),
    ]:
        p = Path(path)
        if p.exists():
            with open(p) as f: d = json.load(f)
            st.markdown(f"**{label}**")
            st.metric("Urgent response",
                      f"{d['response_time']['ai_urgent_median_hours']*60:.0f} min",
                      delta=f"vs {d['response_time']['manual_urgent_median_hours']:.1f}h")
            st.metric("Care gaps",
                      f"{d['care_gaps']['ai_closure_rate']*100:.0f}%",
                      delta=f"+{d['care_gaps']['improvement_pct']:.0f}%")
            st.metric("CM capacity",
                      f"{d['efficiency']['ai_cm_capacity_per_day']} msg/day",
                      delta=f"+{d['efficiency']['capacity_increase_pct']:.0f}%")
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
    Voice Input · Auto-Triage · Voice Replies · Safety-First</p>
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

# ── VOICE INPUT ───────────────────────────────────────────────────────────────
st.markdown("### 🎤 Voice Input")
st.caption("Click Start → speak → Stop. Transcript appears in the box. Click **Send Voice**.")

components.html("""
<div style="font-family:Inter,sans-serif">
  <div style="display:flex;gap:10px;align-items:center;margin-bottom:10px">
    <button id="vBtn" onclick="toggleRec()"
      style="background:linear-gradient(135deg,#7c3aed,#5b21b6);color:white;border:none;
      padding:10px 20px;border-radius:10px;cursor:pointer;font-size:13px;font-weight:600">
      🎤 Start Recording
    </button>
    <div id="wave" style="display:none;align-items:center;gap:3px">
      <div style="width:3px;height:8px;background:#a78bfa;border-radius:2px;animation:wv .5s ease infinite alternate"></div>
      <div style="width:3px;height:16px;background:#a78bfa;border-radius:2px;animation:wv .5s .1s ease infinite alternate"></div>
      <div style="width:3px;height:22px;background:#a78bfa;border-radius:2px;animation:wv .5s .2s ease infinite alternate"></div>
      <div style="width:3px;height:16px;background:#a78bfa;border-radius:2px;animation:wv .5s .3s ease infinite alternate"></div>
      <div style="width:3px;height:8px;background:#a78bfa;border-radius:2px;animation:wv .5s .4s ease infinite alternate"></div>
      <span style="color:#a78bfa;font-size:12px;margin-left:6px">Listening...</span>
    </div>
    <span id="st" style="font-size:12px;color:#94a3b8">Use Chrome · click to begin</span>
  </div>
  <div id="box" style="background:#1e1b4b;border:1.5px solid #4338ca;border-radius:10px;
    padding:10px 14px;min-height:44px;font-size:13px;color:#c4b5fd;
    line-height:1.5;margin-bottom:8px;cursor:text"
    onclick="selectAll(this)">
    <span id="placeholder" style="opacity:0.4">Spoken words appear here...</span>
    <span id="transcript" style="display:none"></span>
  </div>
  <div style="display:flex;gap:8px">
    <button onclick="copyToClipboard()"
      style="background:linear-gradient(135deg,#0f6e56,#1d9e75);color:white;border:none;
      padding:8px 16px;border-radius:8px;cursor:pointer;font-size:12px;font-weight:600">
      📋 Copy text
    </button>
    <span id="copied" style="font-size:12px;color:#86efac;align-self:center;display:none">
      ✓ Copied! Now paste into the box below and click Send Voice
    </span>
  </div>
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
    document.getElementById('vBtn').innerHTML='⏹ Stop';
    document.getElementById('vBtn').style.background='linear-gradient(135deg,#dc2626,#b91c1c)';
    document.getElementById('wave').style.display='flex';
    document.getElementById('placeholder').style.display='none';
    document.getElementById('transcript').style.display='inline';
    document.getElementById('transcript').textContent='Listening...';
  };
  rec.onresult=(e)=>{
    let interim='';
    for(let i=e.resultIndex;i<e.results.length;i++){
      e.results[i].isFinal?final_t+=e.results[i][0].transcript+' ':interim+=e.results[i][0].transcript;
    }
    document.getElementById('transcript').textContent=final_t+interim;
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
  document.getElementById('transcript').textContent=text||'(nothing heard)';
  document.getElementById('st').textContent=text?'Click "Copy text" then paste below':'Nothing heard';
}
function copyToClipboard(){
  const text=document.getElementById('transcript').textContent.trim();
  if(!text||text==='(nothing heard)')return;
  navigator.clipboard.writeText(text).then(()=>{
    document.getElementById('copied').style.display='inline';
    setTimeout(()=>document.getElementById('copied').style.display='none',4000);
  });
}
function selectAll(el){window.getSelection().selectAllChildren(el);}
</script>
""", height=175)

# ── Input ─────────────────────────────────────────────────────────────────────
st.divider()
vc1, vc2 = st.columns([5,1])
with vc1:
    voice_input = st.text_input(
        "Voice transcript:",
        placeholder="Paste transcript here (Cmd+V) then click Send Voice →",
        key="voice_box",
    )
with vc2:
    send_voice = st.button("Send Voice", type="primary", use_container_width=True)

c1, c2 = st.columns([5,1])
with c1:
    text_input = st.text_input(
        "Message:",
        placeholder="Or type any question — triage, protocols, CCM, system questions...",
        label_visibility="collapsed",
        key="text_box",
    )
with c2:
    send_text = st.button("Send", type="primary", use_container_width=True)

# Quick examples
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

# What else you can ask
with st.expander("💡 What else can you ask?"):
    st.markdown("""
**Patient triage:** "I have chest pain", "My blood sugar is 320", "I don't want to be here anymore", "I ran out of my blood thinner"

**Follow-ups after triage:** "What should the care manager do first?", "Why was this urgent?", "What protocols apply?"

**General CCM:** "What is a care gap?", "What is the 988 crisis line?", "When should I call 911?", "What is prior authorization?"

**About this system:** "How does HITL work?", "What is urgent recall?", "Explain RAGAS", "How does RAG retrieval work?", "What is LoRA fine-tuning?"
    """)

# ── Process ───────────────────────────────────────────────────────────────────
user_input = None
if send_voice and voice_input.strip():
    user_input = voice_input.strip()
elif send_text and text_input.strip():
    user_input = text_input.strip()
elif selected:
    user_input = selected

if user_input and not st.session_state.get("processing"):
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
            # TTS
            if voice_on:
                prefix = {"urgent":"Alert. Urgent. ","high":"High risk. "}.get(result.risk_level,"")
                clean = (prefix+response).replace("**","").replace("*","").replace("`","")[:350]
                components.html(f"""<script>
(function(){{
  const u=new SpeechSynthesisUtterance({json.dumps(clean)});
  u.rate={voice_speed};u.pitch=1.0;u.lang='en-US';
  speechSynthesis.cancel();
  setTimeout(()=>speechSynthesis.speak(u),300);
}})();
</script>""", height=0)
        except Exception as e:
            st.session_state.messages.append({
                "role":"assistant","content":f"Error: {str(e)[:80]}"})
    st.session_state.processing = False
    st.rerun()
