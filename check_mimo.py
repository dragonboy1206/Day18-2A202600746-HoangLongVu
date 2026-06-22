"""Kiểm tra kết nối MiMo bằng giao thức OpenAI-compatible."""

import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from config import OPENAI_BASE_URL, OPENAI_MODEL
from src.llm_client import create_chat_completion, has_llm_config


def main() -> int:
    if not has_llm_config():
        print("Chưa có khóa API. Hãy điền MIMO_API_KEY trong tệp .env.")
        return 1

    print(f"Đang kết nối: {OPENAI_BASE_URL}")
    print(f"Mô hình: {OPENAI_MODEL}")
    try:
        response = create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": "Chỉ trả lời đúng hai từ: Kết nối thành công",
                }
            ],
            max_completion_tokens=30,
            temperature=0,
        )
        print(f"Phản hồi: {response.choices[0].message.content}")
        return 0
    except Exception as exc:
        print(f"Kết nối thất bại: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
