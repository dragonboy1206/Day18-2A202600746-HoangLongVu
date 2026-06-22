from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, re, math
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return " ".join(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk["text"]).lower().split()
            for chunk in chunks
        ]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        try:
            from rank_bm25 import BM25Okapi
            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception as exc:
            print(f"  ⚠️  rank-bm25 không khả dụng, dùng BM25 thuần Python: {exc}")
            self.bm25 = _SimpleBM25(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            SearchResult(
                text=self.documents[i]["text"],
                score=float(scores[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="bm25",
            )
            for i in top_indices if scores[i] > 0
        ]


class DenseSearch:
    def __init__(self):
        try:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
        except ImportError:
            self.client = None
        self._encoder = None
        self._local_documents = []
        self._local_vectors = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
            except Exception:
                self._encoder = False
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        self._local_documents = chunks
        encoder = self._get_encoder()
        if not chunks:
            return
        if not encoder:
            return
        if self.client is None:
            self._local_vectors = encoder.encode(
                [c["text"] for c in chunks], show_progress_bar=False
            )
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct
        texts = [c["text"] for c in chunks]
        vectors = encoder.encode(texts, show_progress_bar=False)
        self._local_vectors = vectors
        try:
            if hasattr(self.client, "collection_exists") and self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self.client.create_collection(
                collection,
                vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist(),
                    payload={**chunk.get("metadata", {}), "text": chunk["text"]},
                )
                for i, (chunk, vector) in enumerate(zip(chunks, vectors))
            ]
            self.client.upsert(collection, points)
        except Exception as exc:
            print(f"  ⚠️  Không kết nối được Qdrant, dùng tìm kiếm dense cục bộ: {exc}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        encoder = self._get_encoder()
        if not encoder or not self._local_documents:
            return []
        query_vector = encoder.encode(query)
        try:
            if self.client is None:
                raise RuntimeError("Qdrant client chưa được cài")
            response = self.client.query_points(
                collection, query=query_vector.tolist(), limit=top_k
            )
            return [
                SearchResult(
                    text=point.payload["text"],
                    score=float(point.score),
                    metadata={k: v for k, v in point.payload.items() if k != "text"},
                    method="dense",
                )
                for point in response.points
            ]
        except Exception:
            if self._local_vectors is None:
                return []
            import numpy as np
            vectors = np.asarray(self._local_vectors)
            query_array = np.asarray(query_vector)
            scores = vectors @ query_array / (
                np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_array) + 1e-9
            )
            indices = np.argsort(scores)[::-1][:top_k]
            return [
                SearchResult(
                    text=self._local_documents[i]["text"],
                    score=float(scores[i]),
                    metadata=self._local_documents[i].get("metadata", {}),
                    method="dense",
                )
                for i in indices
            ]


class _SimpleBM25:
    """Bản BM25 tối giản dùng khi môi trường chưa cài rank-bm25."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        self.avgdl = sum(map(len, corpus)) / max(len(corpus), 1)
        self.term_frequencies = [Counter(document) for document in corpus]
        document_frequency = Counter()
        for document in corpus:
            document_frequency.update(set(document))
        total = len(corpus)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []
        for document, frequencies in zip(self.corpus, self.term_frequencies):
            score = 0.0
            length_ratio = len(document) / max(self.avgdl, 1e-9)
            for term in query_tokens:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / denominator
                )
            scores.append(score)
        return scores


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = rrf_scores.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)
    ranked = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in ranked[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
