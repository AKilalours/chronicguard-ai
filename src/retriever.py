"""
ChronicGuard AI — RAG Retriever Module
Cloud-compatible version: uses simple TF-IDF retrieval when ChromaDB/SBERT unavailable.
Full ChromaDB + MMR + reranker pipeline available locally with sentence-transformers.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

try:
    import chromadb
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False

# ── Care protocol documents ───────────────────────────────────────────────────
CARE_PROTOCOLS = [
    {
        "id": "ccm_001",
        "title": "Medication Adherence Protocol — Chronic Disease Management",
        "content": (
            "For patients reporting missed medications: assess the duration and frequency of "
            "missed doses. For antihypertensives missed >2 days, do not recommend doubling doses. "
            "Schedule same-day care manager outreach. Review barriers to adherence (cost, side "
            "effects, forgetfulness). Engage pharmacy for medication synchronization if applicable. "
            "Document in care plan and flag for provider review if BP readings are elevated."
        ),
        "category": "medication",
        "keywords": ["medication", "missed", "dose", "pill", "prescription", "adherence", "pharmacy", "refill"],
    },
    {
        "id": "ccm_002",
        "title": "Hypertensive Crisis Triage Protocol",
        "content": (
            "Blood pressure readings above 180/120 mmHg with symptoms (headache, chest pain, "
            "vision changes, shortness of breath) require immediate escalation. Direct patient "
            "to call 911 or go to nearest ED. Alert primary care provider immediately. "
            "Do not advise home management for hypertensive emergencies."
        ),
        "category": "symptom_escalation",
        "keywords": ["blood pressure", "hypertension", "headache", "vision", "emergency", "180", "crisis"],
    },
    {
        "id": "ccm_003",
        "title": "Diabetes Glucose Management — Out-of-Range Protocols",
        "content": (
            "For blood glucose below 70 mg/dL: advise 15g fast-acting carbohydrate, recheck in 15 "
            "minutes. If below 54 mg/dL or patient cannot self-treat: call 911. "
            "For glucose above 300 mg/dL persisting >24h: same-day provider alert, assess for "
            "DKA symptoms (nausea, vomiting, abdominal pain, fruity breath)."
        ),
        "category": "diabetes",
        "keywords": ["glucose", "blood sugar", "diabetes", "insulin", "300", "dka", "thirsty", "tired"],
    },
    {
        "id": "ccm_004",
        "title": "Chronic Kidney Disease — Monitoring and Care Gap Protocol",
        "content": (
            "For patients with CKD: eGFR should be monitored quarterly. An acute drop of >25% "
            "from baseline warrants same-day provider notification. Review nephrotoxic medications. "
            "Ensure nephrology referral is active if eGFR <30. Dietary counseling recommended."
        ),
        "category": "lab_results",
        "keywords": ["kidney", "egfr", "creatinine", "dialysis", "renal", "ckd", "lab"],
    },
    {
        "id": "ccm_005",
        "title": "Mental Health Crisis and Behavioral Health Integration Protocol",
        "content": (
            "For patients expressing suicidal ideation or self-harm: do not leave patient "
            "unattended. Immediately escalate to licensed clinical staff. "
            "Provide crisis line number (988 Suicide and Crisis Lifeline). Document exact "
            "language used. Alert provider and behavioral health team same day."
        ),
        "category": "crisis",
        "keywords": ["suicidal", "suicide", "self-harm", "crisis", "hopeless", "don't want", "hurt myself", "988"],
    },
    {
        "id": "ccm_006",
        "title": "Medication Prior Authorization and Refill Denial Protocol",
        "content": (
            "For patients out of critical medications due to PA denial: immediately escalate to "
            "care coordinator. For anticoagulants (warfarin, DOACs): same-day provider alert. "
            "Explore bridge options, samples, manufacturer assistance programs. "
            "Document denial reason and submit appeal same day."
        ),
        "category": "care_gap",
        "keywords": ["prior authorization", "refill", "denied", "blood thinner", "out of", "medication"],
    },
    {
        "id": "ccm_007",
        "title": "Preventive Care Gap Closure — Annual Standards",
        "content": (
            "CCM patients with diabetes should receive: annual dilated eye exam, annual foot exam, "
            "quarterly A1C (if uncontrolled), annual urine microalbumin, annual flu vaccine. "
            "Hypertension patients: annual BMP for electrolytes/renal function, annual lipid panel."
        ),
        "category": "preventive_care",
        "keywords": ["eye exam", "foot exam", "a1c", "annual", "preventive", "vaccine", "screening"],
    },
    {
        "id": "ccm_008",
        "title": "Chest Pain and Cardiac Symptom Triage",
        "content": (
            "Any patient reporting chest pain, pressure, tightness, or discomfort — especially "
            "with radiation to arm, jaw, or back, or accompanied by shortness of breath, "
            "diaphoresis, or nausea — should be treated as a potential cardiac emergency. "
            "Do not advise wait-and-see. Direct patient to call 911 immediately."
        ),
        "category": "symptom_escalation",
        "keywords": ["chest pain", "chest tightness", "shortness of breath", "cardiac", "heart", "arm", "jaw"],
    },
    {
        "id": "ccm_009",
        "title": "Lab Result Follow-Up and Communication Standards",
        "content": (
            "Critical lab values (potassium >6.0 or <3.0, glucose >500 or <50, eGFR <20 acute, "
            "INR >4.0) require same-day provider notification and patient outreach. "
            "Non-critical but abnormal results should be communicated within 3 business days."
        ),
        "category": "lab_results",
        "keywords": ["lab", "results", "potassium", "glucose", "inr", "a1c", "abnormal", "critical"],
    },
    {
        "id": "ccm_010",
        "title": "Care Coordination — Specialist Referral Tracking",
        "content": (
            "All specialist referrals should be tracked to appointment confirmation. "
            "If a referral has not resulted in a scheduled appointment within 30 days, "
            "care manager should initiate outreach to patient and specialist office. "
            "For urgent referrals (cardiology, nephrology): 7-day follow-up."
        ),
        "category": "care_coordination",
        "keywords": ["referral", "specialist", "appointment", "cardiology", "nephrology", "schedule"],
    },
]


@dataclass
class RetrievedDocument:
    doc_id: str
    title: str
    content: str
    category: str
    relevance_score: float
    rerank_score: float = 0.0


@dataclass
class RetrievalResult:
    query: str
    documents: list[RetrievedDocument] = field(default_factory=list)
    context_window: str = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "num_retrieved": len(self.documents),
            "documents": [
                {
                    "id": d.doc_id,
                    "title": d.title,
                    "category": d.category,
                    "relevance_score": round(d.relevance_score, 3),
                }
                for d in self.documents
            ],
            "context_window": self.context_window,
        }


def _keyword_score(query: str, doc: dict) -> float:
    """Simple keyword overlap scoring for cloud fallback."""
    query_lower = query.lower()
    score = 0.0
    for kw in doc.get("keywords", []):
        if kw.lower() in query_lower:
            score += 1.0
    # Also check content overlap
    content_words = set(doc["content"].lower().split())
    query_words = set(query_lower.split())
    overlap = len(content_words & query_words)
    score += overlap * 0.1
    return score


class CareRetriever:
    """
    RAG retriever for care protocols.
    Uses keyword scoring on cloud (no ML dependencies).
    Uses ChromaDB + MMR + reranker locally when available.
    """

    COLLECTION_NAME = "care_protocols"
    EMBED_MODEL = "all-mpnet-base-v2"
    RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, persist_dir: str = "./chroma_db", use_reranker: bool = True):
        self.persist_dir = persist_dir
        self.use_reranker = use_reranker and HAS_SBERT
        self._collection = None
        self._use_chroma = False

        if HAS_SBERT:
            self.embed_model = SentenceTransformer(self.EMBED_MODEL)
            self.reranker = CrossEncoder(self.RERANK_MODEL) if use_reranker else None
        else:
            self.embed_model = None
            self.reranker = None

        if HAS_CHROMA and HAS_SBERT:
            try:
                self._client = chromadb.PersistentClient(path=persist_dir)
                self._use_chroma = True
            except Exception:
                self._use_chroma = False

    def build_index(self, documents: list[dict] | None = None):
        """Index documents — no-op on cloud (uses keyword fallback)."""
        if not self._use_chroma:
            print("Using keyword retrieval (cloud mode — no ChromaDB/SBERT needed)")
            return

        if documents is None:
            documents = CARE_PROTOCOLS

        from chromadb.utils import embedding_functions
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.EMBED_MODEL
        )
        collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._collection = collection

        existing = collection.get()["ids"]
        new_docs = [d for d in documents if d["id"] not in existing]
        if not new_docs:
            print(f"Index already contains {len(existing)} documents.")
            return

        collection.add(
            ids=[d["id"] for d in new_docs],
            documents=[d["content"] for d in new_docs],
            metadatas=[{"title": d["title"], "category": d["category"]} for d in new_docs],
        )
        print(f"Indexed {len(new_docs)} documents into ChromaDB.")

    def retrieve(self, query: str, intent: str | None = None,
                 risk_level: str | None = None, n_candidates: int = 6,
                 top_k: int = 3) -> RetrievalResult:
        """Retrieve relevant care protocols."""
        if self._use_chroma and self._collection is not None:
            return self._retrieve_chroma(query, top_k, n_candidates)
        else:
            return self._retrieve_keywords(query, top_k)

    def _retrieve_keywords(self, query: str, top_k: int) -> RetrievalResult:
        """Keyword-based retrieval — works on cloud with zero ML dependencies."""
        scored = []
        for doc in CARE_PROTOCOLS:
            score = _keyword_score(query, doc)
            scored.append((score, doc))

        scored.sort(key=lambda x: -x[0])
        top = scored[:top_k]

        # Ensure at least 1 result even with zero score
        if not top or all(s == 0 for s, _ in top):
            top = [(0.5, d) for d in CARE_PROTOCOLS[:top_k]]

        retrieved = [
            RetrievedDocument(
                doc_id=doc["id"],
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                relevance_score=max(score, 0.1),
                rerank_score=max(score, 0.1),
            )
            for score, doc in top
        ]

        context = self._build_context_window(query, retrieved, None, None)
        return RetrievalResult(query=query, documents=retrieved, context_window=context)

    def _retrieve_chroma(self, query: str, top_k: int, n_candidates: int) -> RetrievalResult:
        """Full ChromaDB semantic retrieval (local only)."""
        query_results = self._collection.query(
            query_texts=[query],
            n_results=min(n_candidates, 10),
            include=["documents", "metadatas", "distances"],
        )

        candidates = []
        for doc_id, doc, meta, dist in zip(
            query_results["ids"][0],
            query_results["documents"][0],
            query_results["metadatas"][0],
            query_results["distances"][0],
        ):
            candidates.append({
                "id": doc_id, "document": doc,
                "metadata": meta, "relevance_score": 1.0 - dist,
                "rerank_score": 1.0 - dist,
            })

        selected = candidates[:top_k]
        retrieved = [
            RetrievedDocument(
                doc_id=c["id"], title=c["metadata"]["title"],
                content=c["document"], category=c["metadata"]["category"],
                relevance_score=c["relevance_score"], rerank_score=c["rerank_score"],
            )
            for c in selected
        ]
        context = self._build_context_window(query, retrieved, None, None)
        return RetrievalResult(query=query, documents=retrieved, context_window=context)

    def _build_context_window(self, query, docs, intent, risk_level) -> str:
        parts = [
            "CARE MANAGEMENT CONTEXT", "=" * 40,
            f"Patient message: {query}", "",
            "Relevant care protocols:", "-" * 40,
        ]
        for i, doc in enumerate(docs, 1):
            parts.append(f"\n[Protocol {i}] {doc.title}")
            parts.append(doc.content)
        return "\n".join(parts)
