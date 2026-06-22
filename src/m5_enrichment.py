from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ENRICH_CACHE_PATH, ENRICH_MAX_WORKERS
from src.llm_client import create_chat_completion, has_llm_config


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _parse_json_object(content: str) -> dict:
    """Lấy object JSON đầu tiên, bỏ qua markdown và dữ liệu thừa phía sau."""
    if not content or not content.strip():
        raise ValueError("MiMo trả về nội dung rỗng.")

    cleaned = content.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Phản hồi MiMo không chứa object JSON.")

    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("JSON MiMo trả về không phải object.")
    return value


def _fallback_enrichment(text: str, source: str) -> dict:
    """Dự phòng cục bộ, không gọi thêm API sau khi combined call thất bại."""
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]
    lowered = text.lower()
    if any(word in lowered for word in ("mật khẩu", "vpn", "bảo mật")):
        category = "it"
    elif any(word in lowered for word in ("lương", "chi phí", "tạm ứng")):
        category = "finance"
    elif any(word in lowered for word in ("nhân viên", "nghỉ", "thử việc")):
        category = "hr"
    else:
        category = "policy"

    return {
        "summary": " ".join(sentences[:2]) if sentences else text,
        "questions": [
            f"Thông tin nào được nêu về {sentence.rstrip('.')}?"
            for sentence in sentences[:3]
            if len(sentence) > 10
        ],
        "context": (
            f"Đoạn trích thuộc tài liệu {source}."
            if source else "Đoạn trích thuộc tài liệu nguồn."
        ),
        "metadata": {
            "topic": re.sub(r"\s+", " ", text).strip()[:80],
            "entities": [],
            "category": category,
            "language": "vi",
        },
    }


def _cache_key(text: str, source: str) -> str:
    return hashlib.sha256(f"{source}\0{text}".encode("utf-8")).hexdigest()


def _load_cache() -> dict:
    try:
        with open(ENRICH_CACHE_PATH, encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    directory = os.path.dirname(ENRICH_CACHE_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary_path = f"{ENRICH_CACHE_PATH}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, ENRICH_CACHE_PATH)


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if has_llm_config():
        try:
            response = create_chat_completion(
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn sau trong 2 câu tiếng Việt ngắn gọn."},
                    {"role": "user", "content": text},
                ],
                max_completion_tokens=150,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  ⚠️  Không thể tóm tắt bằng OpenAI: {exc}")
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if has_llm_config():
        try:
            response = create_chat_completion(
                messages=[
                    {"role": "system", "content": f"Tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn trả lời được, mỗi câu một dòng."},
                    {"role": "user", "content": text},
                ],
                max_completion_tokens=200,
            )
            lines = response.choices[0].message.content.splitlines()
            return [
                line.strip().lstrip("0123456789.-) ")
                for line in lines if line.strip()
            ][:n_questions]
        except Exception as exc:
            print(f"  ⚠️  Không thể tạo HyQA bằng OpenAI: {exc}")
    sentences = [s.strip() for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"Thông tin nào được nêu về {sentence.rstrip('.')}?" for sentence in sentences[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if has_llm_config():
        try:
            response = create_chat_completion(
                messages=[
                    {"role": "system", "content": "Viết một câu mô tả vị trí và chủ đề đoạn văn."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\n{text}"},
                ],
                max_completion_tokens=80,
            )
            return f"{response.choices[0].message.content.strip()}\n\n{text}"
        except Exception as exc:
            print(f"  ⚠️  Không thể thêm ngữ cảnh bằng OpenAI: {exc}")
    prefix = f"Trích từ tài liệu {document_title}." if document_title else "Trích từ tài liệu nguồn."
    return f"{prefix}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if has_llm_config():
        try:
            response = create_chat_completion(
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": 'Trả về JSON gồm topic, entities, category, language.'},
                    {"role": "user", "content": text},
                ],
                max_completion_tokens=150,
            )
            return _parse_json_object(response.choices[0].message.content)
        except Exception as exc:
            print(f"  ⚠️  Không thể trích metadata bằng OpenAI: {exc}")
    lowered = text.lower()
    if any(word in lowered for word in ("mật khẩu", "vpn", "bảo mật")):
        category = "it"
    elif any(word in lowered for word in ("lương", "chi phí", "tạm ứng")):
        category = "finance"
    elif any(word in lowered for word in ("nhân viên", "nghỉ", "thử việc")):
        category = "hr"
    else:
        category = "policy"
    topic = re.sub(r"\s+", " ", text).strip()[:80]
    return {"topic": topic, "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if has_llm_config():
        try:
            response = create_chat_completion(
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": (
                        'Trả về JSON: {"summary":"", "questions":[], "context":"", '
                        '"metadata":{"topic":"","entities":[],"category":"","language":"vi"}}'
                    )},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                max_completion_tokens=400,
            )
            return _parse_json_object(response.choices[0].message.content)
        except Exception as exc:
            print(f"  ⚠️  Không thể làm giàu bằng MiMo, dùng dự phòng cục bộ: {exc}")
    return _fallback_enrichment(text, source)


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    def build_enriched(
        chunk: dict,
        combined_result: dict | None = None,
    ) -> EnrichedChunk:
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = combined_result or _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        return EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        )

    # Các chế độ riêng lẻ giữ cách chạy tuần tự để dễ quan sát/debug.
    if not use_combined or not has_llm_config() or ENRICH_MAX_WORKERS == 1:
        enriched = []
        for i, chunk in enumerate(chunks):
            enriched.append(build_enriched(chunk))
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)
        return enriched

    # Combined mode: dùng cache và chạy đồng thời có giới hạn.
    cache = _load_cache()
    results: list[EnrichedChunk | None] = [None] * len(chunks)
    pending = {}
    cache_hits = 0

    with ThreadPoolExecutor(max_workers=ENRICH_MAX_WORKERS) as executor:
        for index, chunk in enumerate(chunks):
            source = chunk.get("metadata", {}).get("source", "")
            key = _cache_key(chunk["text"], source)
            cached = cache.get(key)
            if isinstance(cached, dict):
                results[index] = build_enriched(chunk, cached)
                cache_hits += 1
            else:
                future = executor.submit(_enrich_single_call, chunk["text"], source)
                pending[future] = (index, chunk, key)

        completed = cache_hits
        if cache_hits:
            print(f"  Dùng cache cho {cache_hits}/{len(chunks)} chunks.", flush=True)

        for future in as_completed(pending):
            index, chunk, key = pending[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"  ⚠️  Enrichment lỗi ngoài dự kiến: {exc}", flush=True)
                result = _fallback_enrichment(
                    chunk["text"],
                    chunk.get("metadata", {}).get("source", ""),
                )
            results[index] = build_enriched(chunk, result)
            cache[key] = result
            completed += 1
            if completed % 10 == 0 or completed == len(chunks):
                print(f"  Enriched {completed}/{len(chunks)} chunks...", flush=True)

    _save_cache(cache)
    return [result for result in results if result is not None]


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
