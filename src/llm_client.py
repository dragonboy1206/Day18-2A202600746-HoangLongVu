"""Cấu hình dùng chung cho API MiMo theo giao thức OpenAI."""

from functools import lru_cache

from config import (
    DISABLE_LLM,
    MIMO_THINKING,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_RETRIES,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
)


def has_llm_config() -> bool:
    """Kiểm tra người dùng đã cấu hình khóa API hay chưa."""
    return bool(OPENAI_API_KEY) and not DISABLE_LLM


@lru_cache(maxsize=1)
def get_llm_client():
    """Tạo OpenAI SDK client trỏ tới máy chủ MiMo."""
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Chưa cấu hình MIMO_API_KEY hoặc OPENAI_API_KEY trong tệp .env."
        )

    from openai import OpenAI

    return OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=OPENAI_TIMEOUT,
        max_retries=OPENAI_MAX_RETRIES,
    )


def create_chat_completion(messages: list[dict], **kwargs):
    """Gọi MiMo Chat Completions với tên mô hình dùng chung."""
    client = get_llm_client()
    extra_body = dict(kwargs.pop("extra_body", {}) or {})
    extra_body.setdefault("thinking", {"type": MIMO_THINKING})
    return client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        extra_body=extra_body,
        **kwargs,
    )
