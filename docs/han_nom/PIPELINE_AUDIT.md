# Rà soát pipeline OCR Hán–Nôm, bước 1–8

Ngày rà soát: 2026-07-13. Pilot: `HVH_001_01`, ba scan đầu (sáu trang processed),
vision LLM dùng ba trang đầu.

## Kết luận chính

Pipeline chạy đúng về cấu trúc và tạo đủ hai output bắt buộc. Nút thắt không
nằm ở tải ảnh, tách trang hay định dạng output mà ở mô hình nhận dạng: Paddle
`PP-OCRv5/chinese_cht` được huấn luyện cho chữ Trung hiện đại, còn ảnh là chữ
viết tay Hán–Nôm cổ. Vision model `gemini-3.5-flash-low` cũng giản thể hóa,
đoán theo ngữ cảnh và tự đánh confidence cao cho chuỗi sai.

Không có ground truth đã chép tay nên hiện không thể tính accuracy/CER thật và
không thể khẳng định đạt 85%. Confidence Paddle và similarity LLM↔Paddle không
phải accuracy.

## Rà soát từng bước

| Bước | Trạng thái | Nhận xét / quyết định |
|---|---|---|
| 1. Tải ảnh | Đạt | Dùng bản `large`, kiểm tra đủ số trang theo catalog. |
| 2. Cấu trúc input | Đạt | Tách đúng work/chapter, tên trang liên tục và có checksum. |
| 3. Tiền xử lý | Đạt | Tách spread đúng, trang phải trước trang trái; giữ ảnh màu. CLAHE không giúp. |
| 4. OCR | Nút thắt | Paddle nhận dạng được bố cục cột nhưng sai nhiều chữ cổ/Nôm và sinh giản thể. Đã sửa `.env` để key cuối thắng khi bị lặp. |
| 5. So sánh | Chỉ là heuristic | Confidence, số chữ CJK và similarity không thay thế ground truth/CER. |
| 6. Ghép/tách câu | Đạt kỹ thuật | Ghép đúng manifest; dấu đỏ là heuristic, vẫn có nguy cơ tách thừa/thiếu. |
| 7. Validator | Đạt | Hai file đúng UTF-8, ID, tab và coverage; không đo nội dung đúng/sai. |
| 8. LLM correct | Chưa đạt để publish | Đã thêm chế độ `--ocr-guidance none` và guard giản thể. Cả chế độ có/không neo Paddle đều cần review. |

## Thí nghiệm preprocessing

| Biến thể | Mean Paddle confidence | Ký tự CJK | Similarity với ảnh màu |
|---|---:|---:|---:|
| Color hiện tại | 66.213% | 1005 | 100% |
| Gray | 66.772% | 1005 | 85.32% |
| CLAHE | 65.170% | 1014 | 81.86% |

Chênh lệch gray chỉ +0.559 điểm confidence và không có ground truth chứng minh
đúng hơn. Không đổi pipeline chính sang gray/CLAHE.

## Thí nghiệm vision LLM

`gemini-3.5-flash-low`, có OCR Paddle làm gợi ý: similarity lần lượt 76.51%,
76.14%, 78.66%; cả ba trang `review`.

Không cung cấp OCR Paddle: similarity 44.68%, 72.39%, 63.68%; cả ba trang
`review`. Model vẫn sinh `竜`, `録` và nhiều chữ giản thể, đồng thời tự gán
`high` không đáng tin. Vì vậy không chạy/publish toàn quyển bằng model này.

## Điều kiện để đo và đạt mục tiêu 85%

Cần một tập chuẩn nhỏ được người biết Hán–Nôm kiểm tra, tối thiểu khoảng 10–20
trang đại diện. Từ đó tính CER (Character Error Rate): `accuracy = 1 - CER`.
Không nên dùng confidence của model làm tiêu chí 85%.

Nếu hoàn toàn không thể hiệu chỉnh thủ công, chỉ có thể tạo bản OCR tự động có
cờ rủi ro/đồng thuận nhiều model; không thể chứng nhận độ chính xác 85% một cách
trung thực. Hướng cải thiện thực chất là dùng OCR/model chuyên Hán–Nôm hoặc
fine-tune trên dữ liệu cùng kiểu chữ, rồi đánh giá trên ground truth cố định.
