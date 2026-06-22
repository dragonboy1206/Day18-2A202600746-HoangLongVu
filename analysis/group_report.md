# Báo cáo kết quả — Lab 18: Production RAG

**Người thực hiện:** Hoàng Long Vũ
**Ngày:** 22/06/2026

## Phạm vi hoàn thành

| Mô-đun | Nội dung | Kết quả kiểm thử |
|---|---|---:|
| M1 | Semantic, hierarchical và structure-aware chunking | 13/13 |
| M2 | Tách từ tiếng Việt, BM25, dense search và RRF | 5/5 |
| M3 | Cross-encoder reranking và đo độ trễ | 5/5 |
| M4 | RAGAS và cây chẩn đoán lỗi | 4/4 |
| M5 | Làm giàu kết hợp một lần gọi và chế độ dự phòng | 10/10 |
| **Tổng** | Toàn bộ bài kiểm thử | **37/37** |

## Kết quả chạy

- Naive baseline (mốc cơ sở đơn giản): chạy thành công, tạo
  `reports/naive_baseline_report.json`.
- Production pipeline (đường ống xử lý hoàn chỉnh): chạy thành công với 25 tài
  liệu Markdown, 114 đoạn con và 20 câu hỏi.
- RAGAS chưa thể tính điểm thật vì thiếu thư viện `ragas` và khóa OpenAI.
- Ba tệp PDF được bỏ qua vì chưa cài `pypdf`; hai tệp scan còn cần OCR
  (nhận dạng ký tự quang học — chuyển chữ trong ảnh thành văn bản).

## Phát hiện chính

1. Hierarchical chunking giữ được liên kết giữa đoạn nhỏ chính xác và đoạn cha có ngữ cảnh rộng.
2. RRF (hợp nhất thứ hạng nghịch đảo — cộng điểm theo vị trí xếp hạng) cho phép kết hợp BM25 và dense search mà không cần chuẩn hóa hai loại điểm.
3. Các câu hỏi phiên bản cần metadata về trạng thái hiệu lực; chỉ dùng điểm tương đồng là chưa đủ.
4. Câu hỏi nhiều bước và số học cần tách truy vấn hoặc công cụ tính toán xác định.

## Giới hạn môi trường

Các cơ chế dự phòng giúp chương trình không dừng khi thiếu thư viện hoặc mô hình.
Tuy nhiên, để đánh giá chất lượng thật cần chạy lại sau khi cài `requirements.txt`,
khởi động Qdrant, tải mô hình và cấu hình `OPENAI_API_KEY`.
