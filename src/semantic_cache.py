"""
ChronicGuard AI — Semantic Cache
Caches triage results for similar patient messages.
Repeat or semantically similar queries return in <10ms instead of ~4000ms.

Uses cosine similarity on TF-IDF vectors for cloud compatibility.
Upgrades to SBERT embeddings when sentence-transformers is available.

Usage:
    from src.semantic_cache import SemanticCache
    cache = SemanticCache()
    result = cache.get("I forgot my blood pressure meds")
    if result is None:
        result = pipeline.run(message)
        cache.set(message, result)
"""

from __future__ import annotations
import time
import json
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from collections import OrderedDict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except Exception:
    HAS_SBERT = False


@dataclass
class CacheEntry:
    query: str
    result: dict
    embedding: np.ndarray
    timestamp: float
    hit_count: int = 0

    def age_seconds(self) -> float:
        return time.time() - self.timestamp


@dataclass
class CacheStats:
    total_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    hit_latencies: list[float] = field(default_factory=list)
    miss_latencies: list[float] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.total_queries if self.total_queries > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total_queries": self.total_queries,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": round(self.hit_rate, 3),
            "avg_hit_latency_ms": round(self.avg_hit_latency_ms, 2),
            "avg_miss_latency_ms": round(self.avg_miss_latency_ms, 2),
            "latency_speedup": round(
                self.avg_miss_latency_ms / self.avg_hit_latency_ms, 1
            ) if self.avg_hit_latency_ms > 0 else 0,
        }


class SemanticCache:
    """
    Semantic cache for patient message triage results.

    Two embedding modes:
    - SBERT (production): 768-dim cosine similarity, threshold=0.92
    - TF-IDF (cloud/fallback): fast, no ML dependencies, threshold=0.85

    CCM context: patients often send similar messages repeatedly.
    "I forgot my BP meds" and "I didn't take my blood pressure medication"
    should hit the same cache entry.
    """

    SBERT_THRESHOLD = 0.92    # higher precision with dense embeddings
    TFIDF_THRESHOLD = 0.85    # slightly lower for sparse vectors
    MAX_ENTRIES = 500
    TTL_SECONDS = 3600 * 8    # 8 hours — refresh each shift

    def __init__(self, use_sbert: bool = True):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats()
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
        self._tfidf_fitted = False
        self._tfidf_matrix = None
        self._tfidf_queries: list[str] = []

        if use_sbert and HAS_SBERT:
            self._embed_model = SentenceTransformer("all-mpnet-base-v2")
            self._mode = "sbert"
            self._threshold = self.SBERT_THRESHOLD
        else:
            self._embed_model = None
            self._mode = "tfidf"
            self._threshold = self.TFIDF_THRESHOLD

    def _embed(self, text: str) -> np.ndarray:
        if self._mode == "sbert" and self._embed_model:
            return self._embed_model.encode([text], normalize_embeddings=True)[0]
        else:
            # TF-IDF embedding
            if self._tfidf_fitted and self._tfidf_queries:
                vec = self._vectorizer.transform([text]).toarray()[0]
            else:
                vec = np.array([hash(w) % 1000 / 1000 for w in text.lower().split()[:50]])
                if len(vec) < 50:
                    vec = np.pad(vec, (0, 50 - len(vec)))
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        try:
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return float(np.dot(a, b) / (norm_a * norm_b))
        except Exception:
            return 0.0

    def _refit_tfidf(self):
        if len(self._tfidf_queries) >= 2:
            self._vectorizer.fit(self._tfidf_queries)
            self._tfidf_fitted = True

    def get(self, query: str) -> tuple[dict | None, float]:
        """
        Look up a query in the cache.
        Returns (result, latency_ms) — result is None on cache miss.
        """
        t0 = time.time()
        self._stats.total_queries += 1
        self._evict_expired()

        if not self._cache:
            latency = (time.time() - t0) * 1000
            self._stats.cache_misses += 1
            self._stats.miss_latencies.append(latency)
            return None, latency

        query_emb = self._embed(query)
        best_sim = 0.0
        best_entry = None

        for entry in self._cache.values():
            sim = self._similarity(query_emb, entry.embedding)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        latency = (time.time() - t0) * 1000

        if best_sim >= self._threshold and best_entry is not None:
            best_entry.hit_count += 1
            self._stats.cache_hits += 1
            self._stats.hit_latencies.append(latency)
            self._stats.avg_hit_latency_ms = np.mean(self._stats.hit_latencies)
            result = dict(best_entry.result)
            result["_cache_hit"] = True
            result["_similarity"] = round(best_sim, 3)
            result["_cache_latency_ms"] = round(latency, 2)
            return result, latency
        else:
            self._stats.cache_misses += 1
            self._stats.miss_latencies.append(latency)
            self._stats.avg_miss_latency_ms = np.mean(self._stats.miss_latencies)
            return None, latency

    def set(self, query: str, result: dict):
        """Store a triage result in the cache."""
        # LRU eviction if at capacity
        if len(self._cache) >= self.MAX_ENTRIES:
            self._cache.popitem(last=False)

        emb = self._embed(query)
        key = hashlib.md5(query.encode()).hexdigest()
        self._cache[key] = CacheEntry(
            query=query,
            result=result,
            embedding=emb,
            timestamp=time.time(),
        )

        # Update TF-IDF vocabulary
        self._tfidf_queries.append(query)
        if len(self._tfidf_queries) % 10 == 0:
            self._refit_tfidf()

    def _evict_expired(self):
        expired = [k for k, v in self._cache.items() if v.age_seconds() > self.TTL_SECONDS]
        for k in expired:
            del self._cache[k]

    @property
    def stats(self) -> CacheStats:
        self._stats.avg_hit_latency_ms = (
            float(np.mean(self._stats.hit_latencies)) if self._stats.hit_latencies else 0.0
        )
        self._stats.avg_miss_latency_ms = (
            float(np.mean(self._stats.miss_latencies)) if self._stats.miss_latencies else 0.0
        )
        return self._stats

    def size(self) -> int:
        return len(self._cache)

    def clear(self):
        self._cache.clear()
        self._stats = CacheStats()


if __name__ == "__main__":
    print("Testing SemanticCache...")
    cache = SemanticCache(use_sbert=False)

    test_result = {
        "intent": "medication_question",
        "risk_level": "high",
        "requires_human_review": True,
    }

    # Populate cache
    cache.set("I forgot to take my blood pressure medication today", test_result)
    cache.set("My blood sugar has been high for two days", {"intent": "symptom_escalation", "risk_level": "high"})
    cache.set("Can I reschedule my appointment", {"intent": "appointment_admin", "risk_level": "low"})

    # Test similar queries
    test_queries = [
        "I forgot my blood pressure meds",
        "I didn't take my BP medication",
        "my glucose has been elevated",
        "I need to change my appointment time",
        "I have chest pain",  # should miss
    ]

    print(f"\nCache size: {cache.size()} entries | Mode: {cache._mode}")
    print(f"Similarity threshold: {cache._threshold}\n")

    for q in test_queries:
        result, latency = cache.get(q)
        status = f"HIT  (sim={result.get('_similarity', 0):.3f}, {latency:.1f}ms)" if result else f"MISS ({latency:.1f}ms)"
        print(f"  {q[:50]:<50} {status}")

    stats = cache.stats
    print(f"\nCache stats:")
    print(f"  Hit rate:    {stats.hit_rate:.0%}")
    print(f"  Hits:        {stats.cache_hits}/{stats.total_queries}")
    print(f"  Avg hit ms:  {stats.avg_hit_latency_ms:.2f}ms")
    print(f"  Avg miss ms: {stats.avg_miss_latency_ms:.2f}ms")
    if stats.avg_hit_latency_ms > 0 and stats.avg_miss_latency_ms > 0:
        speedup = stats.avg_miss_latency_ms / stats.avg_hit_latency_ms
        print(f"  Speedup:     {speedup:.1f}x faster on cache hits")
