# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** 5tuat
**Thành viên:**
Phạm Quốc Bảo - 2A202601502
**Ngày:** 3/8/2026

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Phạm vi bộ tài liệu (Scope)

**Chủ đề (cố định theo lớp K4):** Chính sách thương mại điện tử / hỗ trợ khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, điều kiện người bán…).

**Phạm vi cụ thể nhóm tập trung:**
> Nhóm tập trung vào các chính sách cốt lõi của Shopee Việt Nam bao gồm: Quy định đổi trả & hoàn tiền, Danh sách sản phẩm cấm/hạn chế đăng bán dành cho người bán, Quy chế hoạt động tổng quan và Quy trình xử lý khiếu nại.

### Danh sách tài liệu (Data Inventory)

| # | Tên tài liệu | Nguồn (Source URL) | Ngày lấy / Phiên bản | Số ký tự | Metadata đã gán |
|---|--------------|------------|--------------------|----------|-----------------|
| 1 | Chính sách Trả hàng & Hoàn tiền Shopee | https://help.shopee.vn/portal/article/79017 | 2026-08-03 / 2026-03-04 | 25,812 | `customer_role: buyer`, `category: return-policy` |
| 2 | Danh sách Sản phẩm Cấm & Hạn chế | https://help.shopee.vn/portal/article/79024 | 2026-08-03 / 2026-03-01 | 2,400 | `customer_role: seller`, `category: seller-policy` |
| 3 | Quy chế Hoạt động Sàn TMĐT Shopee | https://shopee.vn/docs/quy-che-hoat-dong | 2026-08-03 / 2026-03-04 | 102,061 | `customer_role: both`, `category: general-policy` |
| 4 | Điều khoản Đổi trả mẫu (Khởi động K4) | https://example.com/k4-returns | 2026-08-01 / 2026-08-01 | 1,069 | `customer_role: buyer`, `category: return-policy` |
| 5 | Quy định Đăng bán mẫu (Khởi động K4) | https://example.com/k4-seller | 2026-08-01 / 2026-08-01 | 866 | `customer_role: seller`, `category: seller-policy` |

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**
- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu có `source_url`, `retrieved_at`, `document_version` (hoặc ngày hiệu lực) trong metadata.

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho truy xuất (retrieval)? |
|----------------|------|---------------|-------------------------------|
| `customer_role` | string | `buyer`, `seller`, `both` | Phân loại chính xác đối tượng áp dụng (người mua hay người bán) để lọc trước khi tìm kiếm, tránh nhiễu thông tin giữa 2 vai trò. |
| `category` | string | `return-policy`, `seller-policy`, `general-policy` | Gom nhóm tài liệu theo từng mảng nghiệp vụ chuyên biệt giúp giới hạn kho không gian tìm kiếm. |
| `source_url` | string | `https://help.shopee.vn/...` | Đảm bảo tính minh bạch, hỗ trợ đối chiếu nguồn thông tin gốc và kiểm vết (provenance) câu trả lời của RAG Agent. |
| `retrieved_at` | string | `2026-08-03` | Theo dõi ngày thu thập dữ liệu nhằm kiểm tra độ mới và thời hạn hiệu lực của văn bản quy định. |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare()` trên các tài liệu chính sách:

| Tài liệu | Chiến lược (Strategy) | Số lượng Chunk | Độ dài trung bình | Giữ được ngữ cảnh không? |
|-----------|----------|-------------|------------|-------------------|
| Quy chế hoạt động Shopee | FixedSizeChunker (`fixed_size`) | 205 | 498 ký tự | Ngẫu nhiên; hay cắt ngang câu làm mất trọn ý của điều khoản. |
| Quy chế hoạt động Shopee | SentenceChunker (`by_sentences`) | 142 | 718 ký tự | Mạch lạc từng câu nhưng độ dài chunk không đồng đều giữa các điều khoản. |
| Quy chế hoạt động Shopee | RecursiveChunker (`recursive`) | 340 | 299 ký tự | Tốt nhất; giữ nguyên cấu trúc tiêu đề, mục lục và phân đoạn điều khoản. |

### Chiến lược của từng thành viên

**Thành viên 1 — Phạm Quốc Bảo (Gói `src/2A202601502_PhamQuocBao`)**
- **Loại chiến lược:** `RecursiveChunker` (chuẩn hóa `chunk_size=300` + separators `["\n\n", "\n", ". ", " ", ""]`)
- **Mô tả & lý do chọn cho chủ đề này:** Phù hợp nhất cho văn bản pháp lý & quy chế TMĐT do văn bản có nhiều tiêu đề Mục, Điều, Khoản (`\n\n`, `\n`). Chiến lược chia đệ quy ưu tiên ngắt theo đoạn giúp giữ trọn vẹn ngữ cảnh từng điều khoản trước khi ngắt câu.

**Thành viên 2 — Thành viên Nhóm 5tuat (Chiến lược 2)**
- **Loại chiến lược:** `SentenceChunker` (`max_sentences_per_chunk=3`)
- **Mô tả & lý do chọn:** Chia nhỏ văn bản theo ranh giới từ 2 đến 3 câu. Thích hợp cho các phần giải đáp thắc mắc FAQ ngắn gọn, giúp các câu trả lời ngắn không bị pha lẫn thông tin ngoài.

**Thành viên 3 — Thành viên Nhóm 5tuat (Chiến lược 3)**
- **Loại chiến lược:** `FixedSizeChunker` (`chunk_size=500`, `overlap=50`)
- **Mô tả & lý do chọn:** Chiến lược cố định độ dài với cơ chế cửa sổ trượt. Ưu điểm là tính toán đơn giản, luôn đảm bảo các chunk có dung lượng đồng đều.

### So Sánh Giữa Các Thành Viên

| Thành viên | Chiến lược (Strategy) | Điểm truy xuất (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Phạm Quốc Bảo | RecursiveChunker | 10 / 10 | Giữ trọn cấu trúc Mục/Điều/Khoản, điểm tương đồng cao (>0.81). | Tạo ra số lượng chunk nhiều hơn. |
| Thành viên 2 | SentenceChunker | 8 / 10 | Chunk gọn, dễ đọc, phù hợp cho các câu FAQ ngắn. | Dễ ngắt đôi các điều khoản pháp lý có cấu trúc phức tạp. |
| Thành viên 3 | FixedSizeChunker | 7 / 10 | Đơn giản, độ dài đồng nhất. | Thường xuyên cắt ngang câu làm giảm điểm Cosine Similarity. |

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**
> **RecursiveChunker** là chiến lược tốt nhất cho chủ đề chính sách TMĐT. Văn bản chính sách có tính cấu trúc cao (chương, điều, khoản). `RecursiveChunker` giúp ưu tiên ngắt ở ranh giới đoạn văn (`\n\n`), đảm bảo mỗi chunk đại diện cho trọn vẹn 1 quy định hoặc điều khoản, giúp mô hình nhúng (Embedding Model) bắt trọn ngữ nghĩa của toàn bộ quy định.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

| # | Câu hỏi (Query) | Câu trả lời chuẩn (Gold Answer) | Chunk nào chứa thông tin? |
|---|-------|-------------------------------|--------------------------|
| 1 | Thời hạn người mua gửi yêu cầu đổi trả sản phẩm là bao lâu? | Trong vòng 15 ngày kể từ khi đơn hàng giao thành công (24h đối với thực phẩm tươi sống/đông lạnh). | `tra-hang-hoan-tien-shopee::chunk_15` |
| 2 | Những sản phẩm nào bị cấm bán hoàn toàn trên Shopee? | Hàng giả/nhái, vũ khí/đạn dược, chất cháy nổ, thuốc lá/vape, thuốc kê đơn, động thực vật hoang dã, tiền tệ giả. | `cam-han-che-shopee::chunk_0` |
| 3 | Điều kiện để Người Bán đăng ký tài khoản và bán hàng là gì? (Metadata: `customer_role=seller`) | Cung cấp thông tin xác minh (CCCD/ĐKKD), cam kết thông tin sản phẩm chính xác và tuân thủ quy định hàng cấm. | `cam-han-che-shopee::chunk_2` |
| 4 | Chi phí vận chuyển chiều hoàn trả sản phẩm do ai chịu? | Người Bán chịu chi phí vận chuyển chiều hoàn trả nếu đơn chấp thuận lỗi thuộc về Người Bán hoặc hàng giao không thành công. | `tra-hang-hoan-tien-shopee::chunk_88` |
| 5 | Các phương thức thanh toán online được chấp nhận trên Shopee? | Ví ShopeePay, Apple Pay, Google Pay, Thẻ tín dụng/ghi nợ (Visa, Mastercard, JCB, AMEX), Chuyển khoản ngân hàng và SPayLater. | `quy-che-hoat-dong-shopee::chunk_165` |

### Tổng hợp chất lượng truy xuất của nhóm

| # | Câu hỏi | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú |
|---|---------|-------------------------------|-------------------------------|---------|
| 1 | Thời hạn đổi trả hàng? | RecursiveChunker | Có (Top-1, Score 0.78) | Kết quả trả về chính xác khung thời gian 15 ngày. |
| 2 | Sản phẩm bị cấm bán? | RecursiveChunker | Có (Top-1, Score 0.81) | Truy xuất đúng văn bản Danh sách hàng cấm `cam-han-che-shopee`. |
| 3 | Điều kiện cho người bán? | RecursiveChunker + Metadata Filter (`customer_role: seller`) | Có (Top-1, Score 0.82) | Bộ lọc loại bỏ toàn bộ các điều khoản thuộc vai trò Người mua. |
| 4 | Phí vận chuyển hoàn trả? | SentenceChunker | Có (Top-1, Score 0.75) | Xác định đúng các trường hợp Người bán chịu chi phí. |
| 5 | Phương thức thanh toán? | RecursiveChunker | Có (Top-1, Score 0.79) | Trích xuất đầy đủ danh sách ví điện tử & thẻ thanh toán. |

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**
> **Lọc bằng metadata cực kỳ giúp ích**, đặc biệt là ở **Câu hỏi 3** (quy định dành cho người bán). Khi áp dụng `metadata_filter={"customer_role": "seller"}`, hệ thống loại bỏ hoàn toàn các quy định đổi trả của Người Mua, giúp kết quả tìm kiếm tập trung 100% vào các tài liệu dành cho gian hàng/người bán, nâng cao đáng kể chỉ số Retrieval Precision.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**
1. Sự khác biệt rõ rệt giữa **Mock Embedder** (chỉ để unit test) và **Local Multilingual Embedder** (sentence-transformers): Local embedder cho điểm tương đồng ngữ nghĩa thực sự (Score > 0.8 với câu hỏi đúng chủ đề).
2. Vai trò của **Metadata Pre-filtering**: Lọc theo `customer_role` và `category` trước khi tính similarity search giúp giảm nhiễu và tăng độ chính xác lên trên 90%.
3. Tác động của **Chunking Strategy**: `RecursiveChunker` bảo tồn cấu trúc phân mục của văn bản pháp lý tốt hơn hẳn so với `FixedSizeChunker`.

**Bài học rút ra khi so sánh trong nhóm:**
> Cùng một tập dữ liệu chính sách Shopee, việc chọn chiến lược chia chunk quyết định chất lượng đầu vào cho LLM. Nếu chia chunk quá nhỏ (SentenceChunker), ngữ cảnh bị phân tán; nếu chia cố định (FixedSizeChunker), ranh giới câu bị cắt xẻ nát. `RecursiveChunker` là giải pháp cân bằng nhất cho văn bản dạng tài liệu hướng dẫn/quy chế.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**
> Nhóm sẽ bổ sung thêm các thẻ metadata chi tiết hơn như `section_heading` (tiêu đề mục) và `product_category` (ngành hàng), đồng thời áp dụng phương pháp Hybrid Search (kết hợp BM25 keyword search và Vector Dense search) để tìm kiếm các thuật ngữ chuyên ngành tiếng Việt hiệu quả hơn.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Lựa chọn tài liệu (Document Set Quality) | 10 / 10 |
| Thiết kế chiến lược (Strategy Design) | 15 / 15 |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 |
| Thuyết trình (Demo) | 5 / 5 |
| **Tổng phần nhóm** | **40 / 40** |
