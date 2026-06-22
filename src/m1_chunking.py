from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import os, sys, glob, re
from dataclasses import dataclass, field
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DATA_DIR, HIERARCHICAL_PARENT_SIZE, HIERARCHICAL_CHILD_SIZE,
                    SEMANTIC_THRESHOLD)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        print("  ⚠️  Chưa cài pypdf, bỏ qua các tệp PDF.")
        return ""

    try:
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:
        print(f"  ⚠️  Không đọc được {os.path.basename(path)}: {exc}")
        return ""


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  ⚠️  Bỏ qua {os.path.basename(fp)}: PDF scan ảnh, không có text layer (cần OCR).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    if not sentences:
        return []

    try:
        from sentence_transformers import SentenceTransformer
        from numpy import dot
        from numpy.linalg import norm

        @lru_cache(maxsize=1)
        def _model():
            return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

        embeddings = _model().encode(sentences, show_progress_bar=False)

        def similarity(i: int) -> float:
            a, b = embeddings[i - 1], embeddings[i]
            return float(dot(a, b) / (norm(a) * norm(b) + 1e-9))
    except Exception:
        # Dự phòng không cần tải mô hình: hệ số Jaccard trên tập từ.
        token_sets = [
            set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
            for sentence in sentences
        ]

        def similarity(i: int) -> float:
            a, b = token_sets[i - 1], token_sets[i]
            return len(a & b) / max(len(a | b), 1)

        # Jaccard thường thấp hơn cosine nên quy đổi ngưỡng tương đối.
        threshold = min(threshold, 0.2)

    groups = [[sentences[0]]]
    for i in range(1, len(sentences)):
        if similarity(i) < threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])

    return [
        Chunk(
            text="\n\n".join(group),
            metadata={**metadata, "strategy": "semantic", "chunk_index": i},
        )
        for i, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [], []

    def split_oversized(value: str, limit: int) -> list[str]:
        pieces = []
        remaining = value.strip()
        while len(remaining) > limit:
            cut = remaining.rfind(" ", 0, limit + 1)
            cut = cut if cut > 0 else limit
            pieces.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        if remaining:
            pieces.append(remaining)
        return pieces

    parent_texts, current = [], ""
    for paragraph in paragraphs:
        for piece in split_oversized(paragraph, parent_size):
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > parent_size:
                parent_texts.append(current)
                current = piece
            else:
                current = candidate
    if current:
        parent_texts.append(current)

    parents, children = [], []
    source = metadata.get("source", "document")
    for parent_index, parent_text in enumerate(parent_texts):
        pid = f"{source}:parent_{parent_index}"
        parents.append(Chunk(
            text=parent_text,
            metadata={**metadata, "chunk_type": "parent", "parent_id": pid,
                      "chunk_index": parent_index},
        ))

        child_paragraphs = [p.strip() for p in parent_text.split("\n\n") if p.strip()]
        child_texts, child_current = [], ""
        for paragraph in child_paragraphs:
            for piece in split_oversized(paragraph, child_size):
                candidate = f"{child_current}\n\n{piece}".strip() if child_current else piece
                if child_current and len(candidate) > child_size:
                    child_texts.append(child_current)
                    child_current = piece
                else:
                    child_current = candidate
        if child_current:
            child_texts.append(child_current)

        for child_index, child_text in enumerate(child_texts):
            children.append(Chunk(
                text=child_text,
                metadata={**metadata, "chunk_type": "child",
                          "chunk_index": child_index},
                parent_id=pid,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    chunks, header, content = [], "", []

    def append_section():
        section_text = "\n".join([header, *content]).strip()
        if section_text:
            chunks.append(Chunk(
                text=section_text,
                metadata={**metadata, "section": header.lstrip("# ").strip() or "Mở đầu",
                          "strategy": "structure", "chunk_index": len(chunks)},
            ))

    for part in parts:
        if not part or not part.strip():
            continue
        if re.match(r"^#{1,3}\s+", part.strip()):
            append_section()
            header, content = part.strip(), []
        else:
            content.append(part.strip())
    append_section()
    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
