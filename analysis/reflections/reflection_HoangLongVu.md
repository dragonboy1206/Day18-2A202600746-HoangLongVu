# Reflection cá nhân — Lab 18

**Tên:** Hoàng Long Vũ  
**Phạm vi:** M1 đến M5

## 1. Mapping bài giảng vào mã

| Khái niệm | Mô-đun | Hàm/lớp | Quan sát |
|---|---|---|---|
| Semantic chunking (chia đoạn theo ngữ nghĩa) | M1 | `chunk_semantic()` | Dùng cosine similarity khi có mô hình; dùng Jaccard dự phòng khi thiếu mô hình. |
| Hierarchical chunking (chia đoạn phân cấp cha–con) | M1 | `chunk_hierarchical()` | Đoạn con phục vụ truy xuất chính xác và giữ `parent_id` để quay về ngữ cảnh cha. |
| Hybrid search (tìm kiếm lai) | M2 | `BM25Search`, `DenseSearch`, `reciprocal_rank_fusion()` | BM25 xử lý từ khóa; dense search xử lý tương đồng ý nghĩa; RRF hợp nhất thứ hạng. |
| Cross-encoder reranking (xếp hạng lại bằng mô hình đọc cặp câu) | M3 | `CrossEncoderReranker.rerank()` | Có mô hình thật khi đã tải sẵn; nếu thiếu thì dùng độ trùng từ làm phương án dự phòng. |
| RAGAS 4 metrics (bốn chỉ số đánh giá RAG) | M4 | `evaluate_ragas()` | Hàm trả đủ bốn chỉ số và không làm pipeline dừng khi môi trường thiếu RAGAS. |
| Diagnostic Tree (cây chẩn đoán lỗi) | M4 | `failure_analysis()` | Chọn chỉ số thấp nhất để ánh xạ sang nguyên nhân và cách sửa. |
| Contextual enrichment (làm giàu ngữ cảnh) | M5 | `_enrich_single_call()` | Chế độ kết hợp giảm từ bốn lần gọi API xuống một lần cho mỗi đoạn. |

## 2. Khó khăn và cách giải quyết

### Lỗi thiếu thư viện

- Lỗi chính xác: `ModuleNotFoundError: No module named 'pypdf'`.
- Lỗi chính xác: `ModuleNotFoundError: No module named 'rank_bm25'`.
- Cách xử lý: bỏ qua PDF có cảnh báo khi thiếu `pypdf`; viết lớp `_SimpleBM25`
  tương thích với phương thức `get_scores()` để kiểm thử và pipeline vẫn chạy.

### Lỗi bảng mã

- Lỗi chính xác: `UnicodeEncodeError: 'charmap' codec can't encode characters`.
- Nguyên nhân: đầu ra PowerShell dùng `cp1252`, không hỗ trợ đầy đủ tiếng Việt.
- Cách xử lý: cấu hình `stdout` và `stderr` dùng UTF-8 trong các chương trình chạy chính.

### Thiếu dịch vụ và mô hình

- Qdrant, mô hình embedding và cross-encoder chưa sẵn có.
- Cách xử lý: thiết kế fallback (cơ chế dự phòng — cách thay thế khi thành phần chính
  không hoạt động) bằng BM25 và so khớp từ khóa cục bộ.
- Giới hạn: fallback giúp chương trình hoạt động nhưng không thay thế được kết quả
  chất lượng của mô hình thật.

## 3. Action Plan cho project

### Project: Trợ lý tra cứu chính sách nội bộ

#### Hiện tại

- Pipeline hiện tại: nạp Markdown/PDF, chia đoạn, tìm kiếm, xếp hạng lại và trả lời.
- Vấn đề đã biết: tài liệu nhiều phiên bản, PDF scan, câu hỏi phủ định, nhiều bước và số học.

#### Kế hoạch áp dụng

1. [ ] Dùng hierarchical chunking; lưu cả `parent_id` và tiêu đề mục.
2. [ ] Dùng hybrid search gồm BM25 + dense search, hợp nhất bằng RRF.
3. [ ] Dùng `BAAI/bge-reranker-v2-m3` để lấy 3 ngữ cảnh tốt nhất.
4. [ ] Đánh giá bằng RAGAS và bộ chỉ số riêng cho phiên bản, phủ định, số học.
5. [ ] Làm giàu metadata với `version`, `effective_date`, `status`, `department`.
6. [ ] Thêm OCR cho PDF scan và ghi nguồn/trang cho mỗi đoạn.

#### Timeline

- Tuần 1: chuẩn hóa dữ liệu, OCR và metadata phiên bản.
- Tuần 2: triển khai hybrid search, Qdrant và reranking.
- Tuần 3: xây tập kiểm thử, chạy RAGAS và phân tích bottom-5.
- Tuần 4: tối ưu độ trễ, chi phí và thêm kiểm thử hồi quy.

## 4. Tự đánh giá

| Tiêu chí | Điểm (1–5) |
|---|---:|
| Hiểu nội dung bài | 4 |
| Chất lượng mã | 4 |
| Giải quyết vấn đề | 4 |
| Kiểm thử và báo cáo | 4 |
