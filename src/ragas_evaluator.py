"""
ChronicGuard AI — RAGAS Evaluation Module
Measures LLM response quality: faithfulness (hallucination detection)
and answer relevance (did retrieval actually help?).

Works with or without OpenAI API key:
- With key: uses GPT-4o-mini for LLM-based scoring
- Without key: uses keyword-overlap heuristic scoring

Usage:
    from src.ragas_evaluator import RAGASEvaluator
    evaluator = RAGASEvaluator()
    result = evaluator.evaluate(question, answer, contexts)
"""

from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGASResult:
    question: str
    answer: str
    faithfulness: float        # 0-1: is every claim in answer supported by context?
    answer_relevance: float    # 0-1: does the answer address the question?
    context_precision: float   # 0-1: are retrieved docs relevant to the question?
    context_recall: float      # 0-1: do contexts contain info needed to answer?
    overall_score: float       # weighted average
    hallucination_detected: bool
    faithfulness_breakdown: list[dict]
    method: str                # "llm" or "heuristic"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "scores": {
                "faithfulness": round(self.faithfulness, 3),
                "answer_relevance": round(self.answer_relevance, 3),
                "context_precision": round(self.context_precision, 3),
                "context_recall": round(self.context_recall, 3),
                "overall": round(self.overall_score, 3),
            },
            "hallucination_detected": self.hallucination_detected,
            "method": self.method,
        }

    def summary(self) -> str:
        status = "HALLUCINATION DETECTED" if self.hallucination_detected else "No hallucination"
        return (
            f"Faithfulness: {self.faithfulness:.2f} | "
            f"Relevance: {self.answer_relevance:.2f} | "
            f"Overall: {self.overall_score:.2f} | {status}"
        )


def _keyword_faithfulness(answer: str, contexts: list[str]) -> tuple[float, list[dict]]:
    """
    Heuristic faithfulness: check if key claims in the answer
    are grounded in at least one retrieved context document.
    """
    # Extract sentences from answer
    sentences = [s.strip() for s in re.split(r'[.!?]', answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0, []

    combined_context = " ".join(contexts).lower()
    breakdown = []
    supported = 0

    for sent in sentences:
        words = set(re.findall(r'\b[a-z]{4,}\b', sent.lower()))
        # Remove common stop words
        stop = {"that","this","with","have","from","will","your","their","they","been",
                "when","more","also","into","than","then","some","what","about","would",
                "should","could","other","there","these","those","care","patient","please",
                "team","manager","ensure","important","reach","contact","seek","need"}
        content_words = words - stop

        if not content_words:
            supported += 1
            breakdown.append({"sentence": sent[:80], "supported": True, "reason": "generic"})
            continue

        # Check overlap with context
        overlap = sum(1 for w in content_words if w in combined_context)
        overlap_ratio = overlap / len(content_words) if content_words else 0

        is_supported = overlap_ratio >= 0.35
        if is_supported:
            supported += 1

        breakdown.append({
            "sentence": sent[:80],
            "supported": is_supported,
            "overlap_ratio": round(overlap_ratio, 3),
            "reason": "context overlap" if is_supported else "low context overlap",
        })

    score = supported / len(sentences) if sentences else 1.0
    return score, breakdown


def _keyword_relevance(question: str, answer: str) -> float:
    """Heuristic: does the answer address the question's key terms?"""
    q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
    a_words = set(re.findall(r'\b[a-z]{4,}\b', answer.lower()))
    stop = {"that","this","with","have","from","will","your","what","about","would","should"}
    q_words -= stop
    if not q_words:
        return 0.8
    overlap = len(q_words & a_words) / len(q_words)
    # Scale: 0.3+ overlap → good relevance
    return min(1.0, overlap * 2.5)


def _context_precision(question: str, contexts: list[str]) -> float:
    """Are retrieved contexts relevant to the question?"""
    q_words = set(re.findall(r'\b[a-z]{4,}\b', question.lower()))
    stop = {"that","this","with","have","from","will","what","about","would","should","patient"}
    q_words -= stop
    if not q_words or not contexts:
        return 0.8
    scores = []
    for ctx in contexts:
        c_words = set(re.findall(r'\b[a-z]{4,}\b', ctx.lower()))
        overlap = len(q_words & c_words) / len(q_words)
        scores.append(min(1.0, overlap * 2.0))
    return sum(scores) / len(scores)


def _context_recall(answer: str, contexts: list[str]) -> float:
    """Do contexts contain info needed to generate the answer?"""
    a_words = set(re.findall(r'\b[a-z]{5,}\b', answer.lower()))
    stop = {"that","this","with","have","from","will","your","their","they","been","please",
            "important","ensure","contact","reach","should","would","could","about"}
    a_words -= stop
    if not a_words or not contexts:
        return 0.8
    combined = " ".join(contexts).lower()
    c_words = set(re.findall(r'\b[a-z]{5,}\b', combined))
    overlap = len(a_words & c_words) / len(a_words)
    return min(1.0, overlap * 1.5)


class RAGASEvaluator:
    """
    RAGAS-inspired evaluation for ChronicGuard AI RAG pipeline.
    Measures: faithfulness, answer relevance, context precision, context recall.
    """

    HALLUCINATION_THRESHOLD = 0.65

    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._has_openai = bool(self.openai_key)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> RAGASResult:
        """Run full RAGAS evaluation."""
        if self._has_openai:
            try:
                return self._evaluate_llm(question, answer, contexts)
            except Exception:
                pass
        return self._evaluate_heuristic(question, answer, contexts)

    def _evaluate_heuristic(
        self, question: str, answer: str, contexts: list[str]
    ) -> RAGASResult:
        faithfulness, breakdown = _keyword_faithfulness(answer, contexts)
        relevance = _keyword_relevance(question, answer)
        precision = _context_precision(question, contexts)
        recall = _context_recall(answer, contexts)

        overall = (
            faithfulness * 0.35 +
            relevance * 0.30 +
            precision * 0.20 +
            recall * 0.15
        )

        return RAGASResult(
            question=question,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            hallucination_detected=faithfulness < self.HALLUCINATION_THRESHOLD,
            faithfulness_breakdown=breakdown,
            method="heuristic",
        )

    def _evaluate_llm(
        self, question: str, answer: str, contexts: list[str]
    ) -> RAGASResult:
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        context_str = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(contexts))

        prompt = f"""You are evaluating an AI-generated healthcare response for quality and safety.

Patient question: {question}

Retrieved care protocols:
{context_str}

AI-generated answer: {answer}

Score each dimension from 0.0 to 1.0:

1. faithfulness: Are ALL claims in the answer supported by the retrieved protocols? 
   (1.0 = fully grounded, 0.0 = contains hallucinated medical facts)
2. answer_relevance: Does the answer actually address the patient's question?
   (1.0 = directly answers, 0.0 = completely off-topic)
3. context_precision: Are the retrieved protocols relevant to this question?
   (1.0 = perfectly relevant, 0.0 = irrelevant protocols retrieved)
4. context_recall: Do the protocols contain enough information to answer the question?
   (1.0 = all needed info present, 0.0 = protocols don't cover the question)

CRITICAL: In healthcare, faithfulness < 0.7 indicates dangerous hallucination.

Respond ONLY with valid JSON:
{{"faithfulness": 0.0, "answer_relevance": 0.0, "context_precision": 0.0, "context_recall": 0.0, "reasoning": "brief explanation"}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        scores = json.loads(response.choices[0].message.content)
        faithfulness = float(scores.get("faithfulness", 0.8))
        relevance = float(scores.get("answer_relevance", 0.8))
        precision = float(scores.get("context_precision", 0.8))
        recall = float(scores.get("context_recall", 0.8))

        overall = faithfulness * 0.35 + relevance * 0.30 + precision * 0.20 + recall * 0.15

        # Get heuristic breakdown for visualization
        _, breakdown = _keyword_faithfulness(answer, contexts)

        return RAGASResult(
            question=question,
            answer=answer,
            faithfulness=faithfulness,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
            overall_score=overall,
            hallucination_detected=faithfulness < self.HALLUCINATION_THRESHOLD,
            faithfulness_breakdown=breakdown,
            method="llm",
        )

    def evaluate_batch(
        self,
        questions: list[str],
        answers: list[str],
        contexts_list: list[list[str]],
    ) -> dict:
        """Evaluate a batch and return aggregate statistics."""
        results = []
        for q, a, ctx in zip(questions, answers, contexts_list):
            r = self.evaluate(q, a, ctx)
            results.append(r)

        avg = lambda key: sum(getattr(r, key) for r in results) / len(results)
        n_hallucinations = sum(1 for r in results if r.hallucination_detected)

        return {
            "n_evaluated": len(results),
            "avg_faithfulness": round(avg("faithfulness"), 3),
            "avg_answer_relevance": round(avg("answer_relevance"), 3),
            "avg_context_precision": round(avg("context_precision"), 3),
            "avg_context_recall": round(avg("context_recall"), 3),
            "avg_overall": round(avg("overall_score"), 3),
            "hallucination_rate": round(n_hallucinations / len(results), 3),
            "n_hallucinations": n_hallucinations,
            "method": results[0].method if results else "unknown",
            "individual_results": [r.to_dict() for r in results],
        }


# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "question": "I forgot my blood pressure medication for two days. What should I do?",
        "answer": (
            "Thank you for letting us know. Please do not double your dose to make up for missed days. "
            "A care manager will reach out to you today to discuss next steps and review any barriers "
            "to taking your medication consistently. If your blood pressure readings are very high or "
            "you have symptoms like headache or chest pain, please contact us right away."
        ),
        "contexts": [
            "For patients reporting missed medications: assess the duration and frequency of missed doses. "
            "For antihypertensives missed >2 days, do not recommend doubling doses. Schedule same-day "
            "care manager outreach. Review barriers to adherence (cost, side effects, forgetfulness).",
        ],
    },
    {
        "question": "I have chest pain and shortness of breath.",
        "answer": (
            "This sounds like it could be a medical emergency. Please call 911 immediately or have "
            "someone take you to the nearest emergency room right away. Do not drive yourself. "
            "Your care team will be notified and will follow up with you."
        ),
        "contexts": [
            "Any patient reporting chest pain, pressure, tightness, or discomfort — especially with "
            "radiation to arm, jaw, or back, or accompanied by shortness of breath — should be treated "
            "as a potential cardiac emergency. Direct patient to call 911 immediately.",
        ],
    },
    {
        "question": "My refill for my blood thinner was denied.",
        "answer": (
            "We understand this is urgent. A care coordinator will contact your insurance today to "
            "appeal the prior authorization denial. In the meantime, please do not stop taking your "
            "medication without speaking to your provider. We will explore bridge options and "
            "manufacturer assistance programs if needed."
        ),
        "contexts": [
            "For patients out of critical medications due to PA denial: immediately escalate to care "
            "coordinator. For anticoagulants: same-day provider alert. Explore bridge options, samples, "
            "manufacturer assistance programs. Document denial reason and submit appeal same day.",
        ],
    },
]


if __name__ == "__main__":
    evaluator = RAGASEvaluator()
    print("Running RAGAS evaluation on test cases...")
    print(f"Method: {'LLM (GPT-4o-mini)' if evaluator._has_openai else 'Heuristic (no API key)'}\n")

    for i, case in enumerate(TEST_CASES, 1):
        result = evaluator.evaluate(case["question"], case["answer"], case["contexts"])
        print(f"Case {i}: {case['question'][:60]}...")
        print(f"  {result.summary()}")

    batch_result = evaluator.evaluate_batch(
        [c["question"] for c in TEST_CASES],
        [c["answer"] for c in TEST_CASES],
        [c["contexts"] for c in TEST_CASES],
    )

    Path("results").mkdir(exist_ok=True)
    with open("results/ragas_evaluation.json", "w") as f:
        json.dump(batch_result, f, indent=2)

    print(f"\nBatch summary:")
    print(f"  Avg faithfulness:     {batch_result['avg_faithfulness']:.3f}")
    print(f"  Avg answer relevance: {batch_result['avg_answer_relevance']:.3f}")
    print(f"  Hallucination rate:   {batch_result['hallucination_rate']:.3f}")
    print(f"  Avg overall:          {batch_result['avg_overall']:.3f}")
    print(f"\nResults saved → results/ragas_evaluation.json")
