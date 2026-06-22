from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DISABLE_LLM,
    MIMO_THINKING,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    RAGAS_MAX_WORKERS,
    RAGAS_EMBEDDING_MODEL,
    TEST_SET_PATH,
)


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    metric_names = [
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    ]
    if DISABLE_LLM:
        return {
            **{name: 0.0 for name in metric_names},
            "per_question": [],
            "evaluation_status": "disabled",
            "attempted_questions": len(questions),
        }
    try:
        from ragas import evaluate
        from ragas.run_config import RunConfig
        from ragas.metrics import (
            faithfulness, answer_relevancy, context_precision, context_recall
        )
        from datasets import Dataset
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI

        class MiMoChatOpenAI(ChatOpenAI):
            """ChatOpenAI tương thích MiMo Token Plan, loại tham số n."""

            def _get_request_payload(self, *args, **kwargs) -> dict:
                payload = super()._get_request_payload(*args, **kwargs)
                payload.pop("n", None)
                return payload

        if not OPENAI_API_KEY:
            raise RuntimeError("Chưa cấu hình MIMO_API_KEY trong tệp .env.")

        ragas_llm = MiMoChatOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            model=OPENAI_MODEL,
            temperature=0,
            timeout=OPENAI_TIMEOUT,
            max_retries=OPENAI_MAX_RETRIES,
            extra_body={"thinking": {"type": MIMO_THINKING}},
        )
        ragas_embeddings = HuggingFaceEmbeddings(
            model_name=RAGAS_EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=ragas_llm,
            embeddings=ragas_embeddings,
            run_config=RunConfig(
                timeout=max(OPENAI_TIMEOUT, 60),
                max_retries=OPENAI_MAX_RETRIES,
                max_workers=RAGAS_MAX_WORKERS,
            ),
        )
        df = result.to_pandas()
        per_question = [
            EvalResult(
                question=row["question"],
                answer=row["answer"],
                contexts=list(row["contexts"]),
                ground_truth=row["ground_truth"],
                **{
                    name: (
                        float(row.get(name, 0.0))
                        if math.isfinite(float(row.get(name, 0.0)))
                        else 0.0
                    )
                    for name in metric_names
                },
            )
            for _, row in df.iterrows()
        ]
        aggregates = {
            name: (
                sum(getattr(item, name) for item in per_question) / len(per_question)
                if per_question else 0.0
            )
            for name in metric_names
        }
        return {
            **aggregates,
            "per_question": per_question,
            "evaluation_status": "success",
            "attempted_questions": len(questions),
        }
    except Exception as exc:
        print(f"  ⚠️  Không chạy được RAGAS, trả về điểm 0: {exc}")
        return {
            **{name: 0.0 for name in metric_names},
            "per_question": [],
            "evaluation_status": "failed",
            "attempted_questions": len(questions),
            "error": str(exc),
        }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": (
            "Câu trả lời chứa thông tin không được ngữ cảnh hỗ trợ.",
            "Siết chặt prompt, yêu cầu trích dẫn và giảm temperature.",
        ),
        "context_recall": (
            "Thiếu đoạn tài liệu cần thiết trong kết quả truy xuất.",
            "Cải thiện cách chia đoạn, tăng top-k hoặc bổ sung BM25.",
        ),
        "context_precision": (
            "Kết quả truy xuất có quá nhiều đoạn không liên quan.",
            "Thêm reranking và bộ lọc metadata.",
        ),
        "answer_relevancy": (
            "Câu trả lời chưa đi thẳng vào nội dung câu hỏi.",
            "Cải thiện mẫu prompt và yêu cầu trả lời ngắn, trực tiếp.",
        ),
    }
    analyzed = []
    for item in eval_results:
        scores = {
            name: getattr(item, name)
            for name in diagnostic_tree
        }
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": item.question,
            "average_score": sum(scores.values()) / len(scores),
            "worst_metric": worst_metric,
            "score": scores[worst_metric],
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyzed, key=lambda item: item["average_score"])[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {
            key: results.get(key, 0.0)
            for key in (
                "faithfulness",
                "answer_relevancy",
                "context_precision",
                "context_recall",
            )
        },
        "num_questions": len(results.get("per_question", [])),
        "attempted_questions": results.get(
            "attempted_questions",
            len(results.get("per_question", [])),
        ),
        "evaluation_status": results.get("evaluation_status", "unknown"),
        "failures": failures,
    }
    if results.get("error"):
        report["error"] = results["error"]
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
