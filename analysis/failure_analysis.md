# Phân tích lỗi — Lab 18: Production RAG

**Cá nhân:** Hoàng Long Vũ
**Ngày thực hiện:** 22/06/2026

## Điểm RAGAS

| Chỉ số | Naive Baseline | Production | Chênh lệch |
|---|---:|---:|---:|
| Faithfulness (độ trung thực với ngữ cảnh) | 0,00* | 0,00* | Chưa đo được |
| Answer Relevancy (mức liên quan của câu trả lời) | 0,00* | 0,00* | Chưa đo được |
| Context Precision (độ chính xác của ngữ cảnh truy xuất) | 0,00* | 0,00* | Chưa đo được |
| Context Recall (khả năng lấy đủ ngữ cảnh cần thiết) | 0,00* | 0,00* | Chưa đo được |

> `*` Không phải điểm chất lượng thực tế. Môi trường chưa cài `ragas` và chưa có
> `OPENAI_API_KEY`, vì vậy hàm đánh giá dùng giá trị dự phòng 0,00. Không thể xếp
> bottom-5 theo điểm RAGAS một cách trung thực. Năm trường hợp dưới đây là năm
> rủi ro tiêu biểu được chọn từ tập kiểm thử để phân tích bằng Error Tree
> (cây chẩn đoán lỗi — chuỗi câu hỏi giúp tìm bước gây sai).

## Năm trường hợp rủi ro

### 1. Chính sách phép năm có nhiều phiên bản

- **Câu hỏi:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Đáp án đúng:** 15 ngày theo chính sách v2024; v2023 có 12 ngày nhưng đã bị thay thế.
- **Rủi ro đầu ra:** Trả lời 12 ngày nếu đoạn v2023 được xếp hạng cao.
- **Error Tree:** Đầu ra sai → ngữ cảnh chứa cả hai phiên bản → truy vấn chưa lọc phiên bản hiện hành.
- **Nguyên nhân gốc:** Metadata (siêu dữ liệu — thông tin mô tả tài liệu) chưa có trường `version` và `status`.
- **Cách sửa:** Trích xuất phiên bản, ngày hiệu lực, trạng thái; ưu tiên `status=current` khi tìm kiếm.

### 2. Chính sách mật khẩu có nhiều phiên bản

- **Câu hỏi:** Bao lâu phải đổi mật khẩu một lần?
- **Đáp án đúng:** 120 ngày theo v2.0; 90 ngày là chính sách cũ.
- **Rủi ro đầu ra:** BM25 tìm cả `mat_khau_v1.md` và `mat_khau_v2.md`.
- **Error Tree:** Đầu ra sai → ngữ cảnh không sai nhưng mâu thuẫn → reranker chưa hiểu trạng thái thay thế.
- **Nguyên nhân gốc:** Điểm liên quan từ khóa không thể tự quyết định tài liệu nào còn hiệu lực.
- **Cách sửa:** Lọc theo phiên bản trước reranking; thêm câu hướng dẫn ưu tiên tài liệu mới trong prompt.

### 3. Câu hỏi phủ định về nhân viên thử việc

- **Câu hỏi:** Nhân viên thử việc có được nghỉ phép năm không?
- **Đáp án đúng:** Không; nếu nghỉ phải xin nghỉ không lương.
- **Rủi ro đầu ra:** Hệ thống lấy đoạn chung về quyền nghỉ phép của nhân viên chính thức.
- **Error Tree:** Đầu ra sai → ngữ cảnh thiếu điều kiện “thử việc” → truy xuất bị nhiễu bởi cụm “nghỉ phép năm”.
- **Nguyên nhân gốc:** Từ khóa phổ biến lấn át từ khóa điều kiện và phủ định.
- **Cách sửa:** Tăng trọng số “thử việc”, bổ sung câu hỏi giả định HyQA và dùng cross-encoder thật.

### 4. Câu hỏi nhiều bước về phép và lương

- **Câu hỏi:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Đáp án đúng:** 18 ngày phép và lương 20–35 triệu đồng/tháng.
- **Rủi ro đầu ra:** Chỉ lấy được tài liệu phép hoặc chỉ lấy được bảng lương.
- **Error Tree:** Đầu ra thiếu → ngữ cảnh chỉ đủ một nửa → truy vấn nhiều ý chưa được tách.
- **Nguyên nhân gốc:** Một truy vấn yêu cầu kết hợp ít nhất hai nguồn và một phép tính.
- **Cách sửa:** Tách truy vấn thành hai truy vấn con, hợp nhất ngữ cảnh, sau đó tính `15 + 9/3 = 18`.

### 5. Câu hỏi số học về phí tạm ứng

- **Câu hỏi:** Tạm ứng 15 triệu, sau 20 ngày mới thanh toán, bị phạt bao nhiêu?
- **Đáp án đúng:** Khoảng 50.000 đồng cho 5 ngày quá hạn.
- **Rủi ro đầu ra:** Trả lời 300.000 đồng/tháng nhưng không tính theo tỷ lệ 5 ngày.
- **Error Tree:** Ngữ cảnh đúng → công thức đúng → bước tính toán hoặc diễn giải thời gian sai.
- **Nguyên nhân gốc:** RAG truy xuất được quy định nhưng không bảo đảm suy luận số học.
- **Cách sửa:** Tách dữ kiện có cấu trúc và dùng hàm tính toán xác định thay vì giao toàn bộ cho LLM.

## Trường hợp trình bày

Chọn câu hỏi phép năm vì nó thể hiện rõ lỗi phiên bản:

1. Kiểm tra đầu ra có dùng số 15 hay không.
2. Kiểm tra ngữ cảnh có tài liệu v2024 và v2023 hay không.
3. Kiểm tra metadata có `version`, `effective_date`, `status` hay không.
4. Sửa tại bước enrichment và lọc metadata trước hybrid search.

## Nếu có thêm một giờ

- Cài đủ thư viện và tải mô hình để đo RAGAS thật.
- Thêm metadata phiên bản/trạng thái cho tài liệu.
- Thêm bộ kiểm thử hồi quy riêng cho câu hỏi phiên bản, phủ định và nhiều bước.
