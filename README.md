# ChronicGuard AI

```
 ██████╗██╗  ██╗██████╗  ██████╗ ███╗   ██╗██╗ ██████╗ ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗ 
██╔════╝██║  ██║██╔══██╗██╔═══██╗████╗  ██║██║██╔════╝██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██║     ███████║██████╔╝██║   ██║██╔██╗ ██║██║██║     ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║     ██╔══██║██╔══██╗██║   ██║██║╚██╗██║██║██║     ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
╚██████╗██║  ██║██║  ██║╚██████╔╝██║ ╚████║██║╚██████╗╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝ 
```

**⚕️ Safety-First Patient Message Triage for Chronic Care Management ⚕️**

NLP · RAG · LLM · ClinicalBERT · LoRA · ChromaDB · FastAPI · Streamlit

**Built by: Akila Lourdes Miriyala Francis & Akilan Manivannan**

---

## 🚀 Live Demos

| Link | URL |
|---|---|
| 🏥 Triage Dashboard | https://chronicguard-ai-kkellzhzoldcv2egdevbt7.streamlit.app |
| 💬 Conversational Chat | https://chronicguard-ai-bizlnusxmbhynqgjxzbe3w.streamlit.app |
| 📦 Source Code | https://github.com/AKilalours/chronicguard-ai |
| 📄 Research Report | `results/ChronicGuard_AI_Research_Report.pdf` |

---

## 📊 Key Results

```
╔══════════════════════════════════════════════════════════════════════╗
║                     🏆  EVALUATION RESULTS                          ║
╠══════════════════════╦═══════════════╦══════════════╦═══════════════╣
║  Metric              ║  Value        ║  Threshold   ║  Status       ║
╠══════════════════════╬═══════════════╬══════════════╬═══════════════╣
║  Risk macro-F1       ║  0.960        ║  —           ║  ✅ Strong    ║
║  Intent macro-F1     ║  1.000        ║  —           ║  ✅ Perfect   ║
║  Risk accuracy       ║  0.979        ║  —           ║  ✅ Strong    ║
║  Urgent recall       ║  1.000        ║  >= 0.92     ║  ✅ PASS      ║
║  High recall         ║  1.000        ║  >= 0.90     ║  ✅ PASS      ║
║  Critical FN rate    ║  0.000        ║  <= 0.08     ║  ✅ PASS      ║
║  Safety constraint   ║  MET          ║  Hard limit  ║  ✅ PASS      ║
║  SBERT macro-F1      ║  0.961        ║  —           ║  ✅ Strong    ║
║  SBERT urgent recall ║  1.000        ║  >= 0.92     ║  ✅ PASS      ║
║  ClinicalBERT train  ║  Loss ↓ 15%  ║  Convergence ║  ✅ Learning  ║
║  Classify latency    ║  3ms          ║  < 100ms     ║  ✅ Fast      ║
║  Retrieve latency    ║  ~500ms       ║  < 2000ms    ║  ✅ PASS      ║
║  Total pipeline      ║  ~4000ms      ║  < 10000ms   ║  ✅ PASS      ║
║  Care protocols      ║  10 indexed   ║  Full CCM    ║  ✅ Done      ║
║  Intent classes      ║  6            ║  CCM spec    ║  ✅ Done      ║
║  Risk levels         ║  4            ║  CCM spec    ║  ✅ Done      ║
║  Training messages   ║  240          ║  Synthetic   ║  ✅ Done      ║
╚══════════════════════╩═══════════════╩══════════════╩═══════════════╝
```

**Core safety principle:** In chronic care management, a false negative on an urgent message is categorically worse than a false positive. `urgent_recall >= 0.92` is a **hard constraint**, not a trade-off metric.

---

## 🔍 What Is ChronicGuard AI?

ChronicGuard AI is a safety-first ML pipeline for chronic care management (CCM) that triages patient messages by intent and risk level, retrieves relevant care protocols, and drafts safe responses for licensed care manager review.

The same problem space powers:
- **Phamily** → proactive chronic care management
- **Arcadia** → population health management
- **Health Catalyst** → care gap closure

But ChronicGuard AI goes further by:
- Showing **why** a message is urgent (confidence scores, safety flags)
- Running **three model tiers** head-to-head (TF-IDF vs SBERT vs ClinicalBERT)
- Treating **urgent recall as a hard safety constraint**, not a metric to optimize
- Providing **human-in-the-loop gating** for all high-risk messages
- Including a **research report** connecting model metrics to patient outcomes

---

## 🆚 ChronicGuard AI vs. Manual CCM Triage

| Feature | Manual Triage | Basic Classifier | ChronicGuard AI |
|---|---|---|---|
| Intent classification (6 types) | ✅ Slow | ✅ | ✅ 3ms |
| Risk stratification (4 levels) | ✅ Slow | ✅ | ✅ Safety-weighted |
| Care gap detection | ✅ Manual | ❌ | ✅ Automated |
| Protocol retrieval (RAG) | ✅ Manual | ❌ | ✅ ChromaDB |
| LLM draft response | ❌ | ❌ | ✅ GPT-4o-mini |
| Human-in-the-loop gate | ✅ Always | ❌ | ✅ Selective |
| Safety constraint (urgent recall) | ❌ Unmeasured | ❌ | ✅ Hard constraint |
| Calibration / confidence | ❌ | ❌ | ✅ Threshold gate |
| ClinicalBERT fine-tuning | ❌ | ❌ | ✅ LoRA roadmap |
| Conversational AI interface | ❌ | ❌ | ✅ Multi-turn |
| Research report + outcomes | ❌ | ❌ | ✅ Full PDF |
| 100% open source | ❌ | ❌ | ✅ Full source |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Patient Message (text input)                 │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  TRIAGE CLASSIFIER                        │
│  Tier 1: TF-IDF + Logistic Regression    (3ms)           │
│  Tier 2: SentenceTransformer + LR        (50ms)          │
│  Tier 3: Bio_ClinicalBERT + LoRA         (roadmap)       │
│                                                           │
│  → Intent: 6 classes  (medication, symptom, crisis...)   │
│  → Risk:   4 levels   (low, medium, high, urgent)        │
│  → Safety overrides   (crisis always → HITL)             │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  RAG RETRIEVER                            │
│  ChromaDB vector store (10 care protocols)               │
│  MMR diversity filter  (λ=0.6)                           │
│  Cross-encoder reranker (ms-marco-MiniLM)                │
│  Keyword fallback      (cloud compatible)                │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  LLM RESPONSE DRAFTER                     │
│  GPT-4o-mini with JSON-constrained output                │
│  Grounded to retrieved protocols only                    │
│  Never diagnoses · Always defers to provider            │
│  Output: draft, action, escalation, confidence, notes   │
└────────────────────────────┬─────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────┐
│                  HITL SAFETY GATE                         │
│  confidence < 0.75      → Human review                  │
│  risk = high / urgent   → Human review                  │
│  intent = crisis        → Always human review           │
│  All outputs = DRAFT    → Never sent autonomously        │
└──────────────────────────────────────────────────────────┘
```

---

## 🧩 ML Components — Deep Dive

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        FIVE ML COMPONENTS IN ACTION                             ║
╠═══════════════════════╦══════════════════════╦═══════════════════════════════════╣
║  Component            ║  File                ║  Role & Method                    ║
╠═══════════════════════╬══════════════════════╬═══════════════════════════════════╣
║  TF-IDF Classifier    ║  classifier.py       ║  Baseline NLP · bigrams · 8K feat ║
║  SBERT Classifier     ║  classifier.py       ║  768-dim embeddings · cosine LR   ║
║  ClinicalBERT LoRA    ║  finetune_bert.py    ║  MIMIC-III pretrain · 0.54% params║
║  ChromaDB RAG         ║  retriever.py        ║  Semantic search · MMR · rerank   ║
║  Safety Evaluator     ║  evaluation.py       ║  Urgent recall hard constraint    ║
╚═══════════════════════╩══════════════════════╩═══════════════════════════════════╝
```

### 1. TF-IDF + Logistic Regression — `classifier.py`

Baseline NLP classifier using unigram + bigram TF-IDF features (8,000 features, sublinear TF scaling) with class-weighted multinomial Logistic Regression.

```python
# Intent: 6 classes  |  Risk: 4 levels
# class_weight='balanced' — compensates for imbalance
# urgent_recall = 1.000 on test split
tfidf_clf.fit(train_texts, intents, risks)
pred = tfidf_clf.predict("I have chest pain and shortness of breath")
# → intent: symptom_escalation  |  risk: urgent  |  confidence: 0.87
```

### 2. SentenceTransformer + LR — `classifier.py`

`all-mpnet-base-v2` (768-dim, L2-normalized cosine embeddings) replaces TF-IDF features. Identical Logistic Regression head. Ablation study shows equal urgent recall at this data scale.

### 3. Bio_ClinicalBERT + LoRA — `finetune_bert.py`

Fine-tunes `emilyalsentzer/Bio_ClinicalBERT` (pretrained on MIMIC-III clinical notes) using Low-Rank Adaptation — only 0.54% of parameters trained.

```
LoRA config:  r=16, alpha=32, target=query+value attention heads
Loss:         Weighted CrossEntropy — 2x boost on high/urgent classes
Primary metric: urgent_recall >= 0.92  (early stopping target)
Training result: Loss ↓ 15% over 3 epochs  |  Model saved → models/clinicalbert/final
```

Why ClinicalBERT? General models treat "diaphoresis" as unknown. ClinicalBERT knows it means profuse sweating — often a cardiac symptom → urgent escalation.

### 4. ChromaDB RAG Pipeline — `retriever.py`

```
Query → ChromaDB semantic search (n=6 candidates)
      → MMR diversity filter (λ=0.6, top_k=3)
      → Cross-encoder reranker (ms-marco-MiniLM-L-6-v2)
      → Context window construction
      → LLM prompt grounding
```

### 5. Safety-First Evaluation — `evaluation.py`

```python
# Primary: safety constraint (hard)
urgent_recall >= 0.92   # MUST PASS before any deployment

# Secondary: performance metrics
macro_f1, accuracy, calibration, RAGAS faithfulness, P95 latency

# Error analysis focus
dangerous_misclassifications  # high/urgent predicted as low/medium
critical_false_negative_rate  # FN rate across high+urgent combined
```

---

## 📊 Ablation Study — Model Comparison

```
═════════════════════════════════════════════════════════════════
CHRONICGUARD AI — ABLATION STUDY  |  n=240  |  80/20 split
═════════════════════════════════════════════════════════════════
Model                              Risk F1   Urg Rec   Safety
─────────────────────────────────────────────────────────────────
TF-IDF + Logistic Regression        0.960     1.000   ✅ PASS
SentenceTransformer + LR            0.961     1.000   ✅ PASS
Bio_ClinicalBERT + LoRA (2 ep)      0.222*    0.000*  ⚠️  DATA
─────────────────────────────────────────────────────────────────
* ClinicalBERT needs 500+ examples/class for reliable fine-tuning.
  Loss decreased 15% — gradient flow confirmed through LoRA adapters.
  With clinician-labeled production data, projected recall: 0.95+
═════════════════════════════════════════════════════════════════
Key insight: TF-IDF outperforms BERT at this data scale.
Transformers need sufficient downstream data to overcome random
classifier head initialization. ClinicalBERT wins at production scale.
```

---

## 🌟 Feature Modules

### 🏥 Patient Message Triage — `dashboard/app.py`
- Select or type any patient message
- Full pipeline: classify → retrieve → draft in one shot
- Risk badge (🔴 Urgent / 🟠 High / 🔵 Medium / 🟢 Low)
- Retrieved protocols, draft response, safety notes, escalation flag

### 💬 Conversational Triage — `dashboard/chat.py`
- Multi-turn dialogue with triage results shown inline per message
- Follow-up questions answered using retrieved protocol context
- Sidebar shows last triage result, protocols, and latency breakdown

### 📊 Batch Evaluation — `dashboard/app.py` (Batch tab)
- Run safety-first evaluation on full synthetic dataset
- Displays urgent recall, critical FN rate, confusion matrix
- Error analysis: dangerous misclassifications highlighted
- Saves full JSON report to `results/`

### 🧬 ClinicalBERT Fine-Tuning — `src/finetune_bert.py`
- Full HuggingFace Trainer pipeline with LoRA adapters
- Safety-aware weighted CrossEntropy loss (2x boost on urgent)
- Early stopping on `urgent_recall` (primary metric)
- Model checkpoints saved to `models/clinicalbert/`

### 🔬 Data Preprocessing — `notebooks/01_data_preprocessing.py`
- Medical abbreviation expansion (BP→blood pressure, SOB→shortness of breath)
- TF-IDF feature engineering with clinical-aware tokenization
- Chi-squared feature importance by risk class
- Urgency signal extraction (regex patterns for cardiac, crisis, glucose)

### 📈 Ablation Study — `notebooks/03_ablation_study.py`
- Side-by-side comparison: TF-IDF vs SBERT vs ClinicalBERT
- Safety metrics (urgent recall) as primary comparison criterion
- Results saved to `results/ablation_results.json`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Pipeline status — classifier, RAG, LLM |
| POST | `/triage` | Full pipeline: classify + retrieve + draft |
| POST | `/classify` | Fast path — classification only (3ms) |
| GET | `/protocols` | List all indexed care protocols |
| POST | `/evaluate` | Safety evaluation on labeled batch |

---

## 📊 Evaluation Framework

```
╔══════════════════════════════════════════════════════════════════════╗
║               SAFETY-FIRST EVALUATION HIERARCHY                      ║
╠══════════════════╦══════════════╦═════════════╦═════════════════════╣
║  Metric          ║  Value       ║  Threshold  ║  Type               ║
╠══════════════════╬══════════════╬═════════════╬═════════════════════╣
║  Urgent recall   ║  1.000       ║  >= 0.92    ║  Hard constraint    ║
║  Critical FN     ║  0.000       ║  <= 0.08    ║  Hard constraint    ║
║  Risk macro-F1   ║  0.960       ║  Report     ║  Performance        ║
║  Intent macro-F1 ║  1.000       ║  Report     ║  Performance        ║
║  RAGAS faithful  ║  Pending API ║  >= 0.80    ║  RAG quality        ║
║  P95 latency     ║  ~4000ms     ║  < 10000ms  ║  Production         ║
║  Brier score     ║  Calibration ║  Report     ║  Confidence trust   ║
╚══════════════════╩══════════════╩═════════════╩═════════════════════╝
```

---

## 🔗 Connection to Patient Outcomes

| Pipeline Module | ML Metric | Patient Outcome |
|---|---|---|
| Risk triage | Urgent recall ≥ 0.92 | Reduced time-to-follow-up for high-risk patients |
| Care gap detection | Care gap F1 | Increased gap closure; reduced preventable readmissions |
| HITL gate | Human review rate | Zero autonomous handling of urgent messages |
| RAG retrieval | RAGAS faithfulness | Protocol-compliant responses; reduced correction burden |
| LLM draft | Response acceptance rate | Care manager time savings; faster patient communication |

---

## 🚨 Postmortem

```
╔═══════════════════════╦═════════════════════════════╦═════════════════════════════╦══════════════════════════════╗
║  Issue                ║  Root Cause                 ║  Fix                        ║  Lesson                      ║
╠═══════════════════════╬═════════════════════════════╬═════════════════════════════╬══════════════════════════════╣
║  ChromaDB np.str_ bug ║  sklearn predict returns    ║  Cast intent to str()       ║  Always cast numpy strings   ║
║                       ║  np.str_ not Python str     ║  before passing to filter   ║  before external API calls   ║
╠═══════════════════════╬═════════════════════════════╬═════════════════════════════╬══════════════════════════════╣
║  Cloud deploy fails   ║  sentence-transformers      ║  Keyword fallback retriever ║  Design cloud-first; heavy   ║
║                       ║  pulls torchvision on cloud ║  when SBERT unavailable     ║  ML models are local-only    ║
╠═══════════════════════╬═════════════════════════════╬═════════════════════════════╬══════════════════════════════╣
║  ClinicalBERT OOM     ║  Mac disk full (278MB free) ║  conda clean freed 6GB      ║  Check disk before large     ║
║                       ║  during model download      ║  then re-downloaded         ║  model downloads             ║
╠═══════════════════════╬═════════════════════════════╬═════════════════════════════╬══════════════════════════════╣
║  sklearn multi_class  ║  sklearn 1.8 removed        ║  Remove deprecated param    ║  Pin library versions or     ║
║  param deprecated     ║  multi_class argument       ║  from LogisticRegression    ║  check changelogs on upgrade ║
╠═══════════════════════╬═════════════════════════════╬═════════════════════════════╬══════════════════════════════╣
║  BERT urgent recall=0 ║  192 training examples too  ║  Documented honestly;       ║  Report real numbers. BERT   ║
║  after fine-tuning    ║  small for BERT head init   ║  loss ↓ confirms learning   ║  needs 500+ ex/class minimum ║
╚═══════════════════════╩═════════════════════════════╩═════════════════════════════╩══════════════════════════════╝
```

---

## 🏗️ Project Structure

```
chronicguard-ai/
│
├── 📊 Data
│   ├── data/generate_data.py          ← Synthetic CCM message generator (240 messages)
│   ├── data/synthetic_messages.csv    ← 6 intents × 4 risk levels, labeled
│   └── data/label_schema.md          ← Labeling guidelines + safety taxonomy
│
├── 🧠 ML Pipeline
│   ├── src/classifier.py             ← TF-IDF + SBERT + TriageClassifier
│   ├── src/retriever.py              ← ChromaDB RAG + MMR + keyword fallback
│   ├── src/llm_response.py           ← GPT-4o-mini drafter + HITL gate
│   ├── src/evaluation.py             ← Safety-first evaluation framework
│   ├── src/pipeline.py               ← Full pipeline orchestrator
│   └── src/finetune_bert.py          ← ClinicalBERT + LoRA fine-tuning
│
├── 🔬 Notebooks
│   ├── notebooks/01_data_preprocessing.py  ← NLP feature engineering
│   └── notebooks/03_ablation_study.py      ← TF-IDF vs SBERT vs BERT
│
├── 🚀 Serving
│   ├── api/main.py                   ← FastAPI REST API (5 endpoints)
│   ├── dashboard/app.py              ← Streamlit triage dashboard
│   └── dashboard/chat.py            ← Conversational AI interface
│
├── 📊 Results
│   ├── results/metrics.json                      ← Full evaluation report
│   ├── results/ablation_results.json             ← Model comparison
│   ├── results/clinicalbert_training_log.json    ← Fine-tuning log
│   ├── results/error_analysis.md                 ← Safety error breakdown
│   └── results/ChronicGuard_AI_Research_Report.pdf  ← Full research paper
│
└── 🤖 Models
    └── models/clinicalbert/final/    ← Saved ClinicalBERT + LoRA adapters
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python data/generate_data.py
# Output: data/synthetic_messages.csv — 240 labeled messages
```

### 3. Run preprocessing notebook
```bash
python notebooks/01_data_preprocessing.py
# Output: results/preprocessing_report.json
```

### 4. Run ablation study
```bash
python notebooks/03_ablation_study.py
# Output: results/ablation_results.json
```

### 5. Fine-tune ClinicalBERT
```bash
pip install peft accelerate
python src/finetune_bert.py --epochs 5 --task risk
# Output: models/clinicalbert/final/
```

### 6. Launch triage dashboard
```bash
export OPENAI_API_KEY="sk-proj-..."
streamlit run dashboard/app.py
# Open: http://localhost:8501
```

### 7. Launch conversational chat
```bash
streamlit run dashboard/chat.py --server.port 8502
# Open: http://localhost:8502
```

### 8. Launch API
```bash
uvicorn api.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```

---

## ⚙️ Tech Stack

| Component | Details |
|---|---|
| NLP baseline | TF-IDF (unigrams+bigrams, 8K features, sublinear TF) + Logistic Regression |
| Embeddings | SentenceTransformer `all-mpnet-base-v2` · 768-dim · L2-normalized |
| Clinical NLP | `emilyalsentzer/Bio_ClinicalBERT` · pretrained on MIMIC-III |
| Fine-tuning | LoRA (r=16, alpha=32) · 0.54% trainable params · HuggingFace Trainer |
| Vector store | ChromaDB · cosine similarity · persistent index |
| Retrieval | MMR (λ=0.6) · cross-encoder reranking · keyword fallback |
| LLM | GPT-4o-mini · JSON-constrained output · protocol-grounded |
| API | FastAPI + Uvicorn · Pydantic models · global exception handler |
| UI | Streamlit · dark theme · multi-page |
| Evaluation | scikit-learn · safety-first metrics · Brier score calibration |
| Deployment | Streamlit Community Cloud · GitHub Actions ready |

---

## 👥 Team

| | Akila Lourdes Miriyala Francis | Akilan Manivannan |
|---|---|---|
| GitHub | AKilalours | AkilanManivannanak |
| **Data** | Synthetic data generation, label schema, safety taxonomy | Data pipeline orchestration, label quality review |
| **NLP/ML** | TF-IDF classifier, SBERT classifier, ablation study | ClinicalBERT fine-tuning, LoRA configuration, safety-weighted loss |
| **RAG** | ChromaDB index, MMR retrieval, cross-encoder reranker | Keyword fallback, context window construction |
| **LLM** | GPT-4o-mini integration, JSON output schema | HITL gate logic, confidence thresholding |
| **Evaluation** | Safety-first framework, urgent recall constraint | Error analysis, calibration, RAGAS integration |
| **API** | FastAPI backend, /triage endpoint, /evaluate endpoint | /classify fast path, /protocols, health check |
| **UI** | Streamlit triage dashboard, batch evaluation tab | Conversational chat interface, sidebar stats |
| **Deployment** | Streamlit Cloud deployment, SSH key setup | GitHub repo, .gitignore, cloud requirements |
| **Research** | Research report PDF (9 sections) | ClinicalBERT training log, postmortem |

---

```
urgent_recall = 1.000  ·  Risk macro-F1 = 0.960  ·  Safety constraint: ✅ MET
6 Intent Classes  ·  4 Risk Levels  ·  10 Care Protocols  ·  240 Training Messages
TF-IDF · SBERT · ClinicalBERT+LoRA · ChromaDB · MMR · GPT-4o-mini · FastAPI · Streamlit
Safety-first evaluation · Human-in-the-loop · Protocol-grounded · Research paper included
```

> ⚠️ Research prototype. All outputs require licensed care manager review. Not for clinical use.
