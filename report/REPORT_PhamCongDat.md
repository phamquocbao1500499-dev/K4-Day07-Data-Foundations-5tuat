# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Công Đạt  
**Mã sinh viên:** 2A202601406  
**Nhóm:** 5tuat  
**Ngày:** 03/08/2026

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất (10).

Package bài làm: `src.PhamCongDat_2A202601406`. Kịch bản benchmark có thể chạy lại tại `scripts/benchmark.py`.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine

**Độ tương tự cosine cao nghĩa là gì?**  
Hai vector embedding có hướng gần nhau, cho thấy hai văn bản biểu diễn nội dung hoặc ý nghĩa gần nhau, ngay cả khi cách dùng từ khác nhau.

**Ví dụ có độ tương tự cao:**

- Câu A: “Tôi muốn đổi trả sản phẩm bị lỗi.”
- Câu B: “Làm thế nào để hoàn hàng khi sản phẩm bị hỏng?”
- Lý do: hai câu cùng nói về việc trả lại một sản phẩm có lỗi.

**Ví dụ có độ tương tự thấp:**

- Câu A: “Chính sách đổi trả bảo vệ người mua.”
- Câu B: “Hôm nay thời tiết rất đẹp.”
- Lý do: hai câu thuộc hai chủ đề không liên quan.

**Tại sao ưu tiên cosine hơn khoảng cách Euclid?**
Cosine đo góc giữa hai vector nên tập trung vào hướng ngữ nghĩa và ít bị ảnh hưởng bởi độ lớn vector. Khoảng cách Euclid còn thay đổi theo độ lớn, vì vậy hai vector cùng hướng vẫn có thể bị xem là xa nhau.

### Bài toán Chunking

Với 10.000 ký tự, `chunk_size=500`, `overlap=50`:

```text
ceil((10000 - 50) / (500 - 50))
= ceil(9950 / 450)
= 23 chunks
```

Khi tăng `overlap` lên 100:

```text
ceil((10000 - 100) / (500 - 100))
= ceil(9900 / 400)
= 25 chunks
```

Số chunk tăng từ 23 lên 25 vì bước trượt giảm từ 450 xuống 400 ký tự. Overlap lớn hơn giúp giữ ngữ cảnh tại ranh giới chunk, đổi lại làm tăng số embedding, dung lượng lưu trữ và chi phí truy xuất.

---

## 2. Hướng tiếp cận của tôi — Cá nhân (10 điểm)

### Các hàm chia nhỏ

**`SentenceChunker.chunk`:**  
Dùng regex `(?<=[.!?])(?:\s+)` để tách tại khoảng trắng sau dấu kết thúc câu và giữ lại dấu câu. Văn bản rỗng trả về danh sách rỗng; các câu được loại khoảng trắng thừa rồi gom theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`:**  
Thử separator theo thứ tự `\n\n`, `\n`, `. `, khoảng trắng và chuỗi rỗng. Đoạn vượt `chunk_size` tiếp tục được chia bằng separator kế tiếp; khi hết separator thì cắt cứng theo kích thước. Dấu phân cách được giữ lại để không làm mất nội dung gốc.

**`HeadingAwareChunker` — chiến lược cá nhân:**
Tài liệu chính sách thường tổ chức theo điều, mục và tiêu đề, nên tôi tách tại heading Markdown, tiêu đề đánh số, tiêu đề chương La Mã (`V.`, `VI.`...) và các mục `Cách 1:`, `Cách 2:`. Mục dài hơn 900 ký tự được chia tiếp bằng `RecursiveChunker`, đồng thời lặp lại tiêu đề mục ở mỗi sub-chunk để giữ ngữ cảnh. Phần mở đầu ngắn chỉ được lặp khi đó không phải là một tiểu điều đánh số, tránh gán nhầm nội dung cho các chunk sau. Metadata `section`, `platform`, `customer_role` và `category` được dùng để thu hẹp đúng phạm vi chính sách.

### So sánh đường cơ sở trên ba tài liệu Amazon

| Tài liệu | FixedSize count / avg | Sentence count / avg | Recursive count / avg | Heading-aware count / avg |
|---|---:|---:|---:|---:|
| Chính sách người bán Amazon | 3 / 690,3 | 5 / 371,8 | 3 / 623,7 | 5 / 372,4 |
| Đăng ký người bán Amazon | 3 / 897,7 | 6 / 413,2 | 3 / 831,0 | 9 / 275,1 |
| Tình trạng sản phẩm Amazon | 4 / 708,8 | 8 / 315,1 | 4 / 633,8 | 6 / 432,0 |

Heading-aware tạo chunk dài vừa phải nhưng giữ trọn từng mục chính sách. So với fixed-size, nó không cắt ngang điều khoản; so với sentence chunking, nó giữ các câu trong cùng mục ở cạnh nhau.

### EmbeddingStore

**`add_documents` + `search`:**  
Mỗi record chứa ID duy nhất, nội dung, metadata và embedding. Store sử dụng ChromaDB khi có sẵn, nếu không sẽ lưu trong bộ nhớ. Query được embedding bằng cùng backend, so sánh bằng dot product trên vector đã chuẩn hóa, sau đó sắp xếp điểm giảm dần.

**`search_with_filter` + `delete_document`:**  
Metadata được lọc trước khi xếp hạng để loại ứng viên sai vai trò, sàn, danh mục hoặc mục chính sách. `delete_document` xóa mọi chunk có `metadata.doc_id` tương ứng và trả về `True` khi thực sự có dữ liệu bị xóa.

### KnowledgeBaseAgent

`answer` nhận câu hỏi, `top_k` và bộ lọc metadata tùy chọn. Agent đưa nội dung cùng `doc_id` của các chunk vào prompt, yêu cầu chỉ trả lời từ ngữ cảnh và gọi `llm_fn`. Benchmark dùng `llm_fn` trích xuất toàn bộ top-3 để chạy hoàn toàn offline. Cách này phù hợp với câu hỏi nhiều ý, chẳng hạn danh sách hàng cấm hoặc các phương thức thanh toán nằm ở nhiều mục liên tiếp, đồng thời mọi câu trả lời vẫn truy vết được về corpus.

---

## 3. Hoàn thiện code — Cá nhân (30 điểm)

Đã hoàn thiện:

- `SentenceChunker`, `RecursiveChunker`, `HeadingAwareChunker`.
- `compute_similarity`, `ChunkingStrategyComparator`.
- `EmbeddingStore`: thêm, tìm kiếm, đếm, lọc và xóa.
- `KnowledgeBaseAgent`: retrieval, metadata filter, tạo prompt và gọi `llm_fn`.

Lệnh kiểm thử:

```powershell
$env:LAB_SOLUTION_PACKAGE="src.PhamCongDat_2A202601406"
python -m pytest tests -v
```

Kết quả:

```text
platform win32 -- Python 3.11.9, pytest-9.1.1
collected 42 items
42 passed in 0.08s
```

**Số lượng test vượt qua: 42 / 42.**

---

## 4. Dự đoán độ tương tự — Cá nhân (5 điểm)

Backend thực tế: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, vector 384 chiều.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Tôi muốn đổi trả sản phẩm bị lỗi. | Làm thế nào để hoàn hàng khi sản phẩm bị hỏng? | Cao | 0,529098 | Có |
| 2 | Người bán phải mô tả sản phẩm chính xác. | Thông tin đăng bán cần đúng với tình trạng hàng. | Cao | 0,728903 | Có |
| 3 | Sản phẩm bị cấm không được đăng bán. | Người bán không được niêm yết hàng thuộc danh mục cấm. | Cao | 0,770703 | Có |
| 4 | Chính sách đổi trả bảo vệ người mua. | Hôm nay thời tiết rất đẹp. | Thấp | -0,096180 | Có |
| 5 | Python là một ngôn ngữ lập trình. | Người mua cần gửi bằng chứng khi hàng bị lỗi. | Thấp | 0,047932 | Có |

**Kết quả đáng chú ý:**
Cặp 1 có cùng ý định nhưng chỉ đạt 0,529098, thấp hơn hai cặp chính sách người bán. Nguyên nhân có thể là mô hình nhận ra quan hệ “đổi trả” và “hoàn hàng” nhưng hai câu sử dụng cấu trúc và từ vựng khác nhau. Điều này cho thấy nên đánh giá bằng thứ hạng retrieval và gold evidence, không nên dựa vào một ngưỡng cosine duy nhất.

---

## 5. Kết quả truy xuất của tôi — Cá nhân (10 điểm)

### Cấu hình có thể tái lập

- Corpus: 10 tài liệu công khai; loại hai file template `example.com`.
- Tổng số chunk: 209.
- Chunker: `HeadingAwareChunker(chunk_size=900)`.
- Embedder: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
- Agent: `KnowledgeBaseAgent` tổng hợp ngữ cảnh truy xuất top-3.
- Lệnh chạy: `python -m scripts.benchmark`.

| # | Câu hỏi | Top-1 chunk và bằng chứng | Score | Liên quan? | Câu trả lời của Agent |
|---|---|---|---:|---|---|
| 1 | Thời hạn người mua gửi yêu cầu đổi trả sản phẩm là bao lâu? | `tra-hang-hoan-tien-shopee`, chunk 6: 15 ngày; thực phẩm tươi sống/đông lạnh là 24 giờ | 0,652384 | Có | 15 ngày từ khi giao thành công; riêng thực phẩm tươi sống và đông lạnh là 24 giờ |
| 2 | Những sản phẩm nào bị cấm bán hoàn toàn trên Shopee? | `cam-han-che-shopee`, chunk 2: tiền tệ, giấy tờ có giá; top-2 chứa sáu nhóm cấm còn lại | 0,745597 | Có | Trả đủ hàng giả/nhái, vũ khí, chất cháy nổ, thuốc lá/vape, thuốc kê đơn, động thực vật hoang dã và tiền tệ giả |
| 3 | Điều kiện để Người Bán đăng ký tài khoản và bán hàng là gì? | `quy-che-hoat-dong-shopee`, chunk 62: CCCD/mã số thuế, đăng ký doanh nghiệp và tài khoản ngân hàng | 0,751488 | Có | Nêu thông tin xác minh, giấy phép cho hàng có điều kiện và tuân thủ danh mục hàng cấm |
| 4 | Chi phí vận chuyển chiều hoàn trả sản phẩm do ai chịu? | `tra-hang-hoan-tien-shopee`, chunk 20: các trường hợp Người Bán chịu phí | 0,588015 | Có | Người Bán chịu phí nếu Shopee chấp thuận mà không do lỗi Người Mua/đơn vị vận chuyển, hoặc giao không thành công |
| 5 | Các phương thức thanh toán online được chấp nhận trên Shopee? | `quy-che-hoat-dong-shopee`, chunk 30: ShopeePay, ApplePay, Google Pay và thẻ; top-2/3 chứa chuyển khoản và SPayLater | 0,644974 | Có | Trả đủ ShopeePay, Apple Pay, Google Pay, Visa/Mastercard/JCB/AMEX, chuyển khoản và SPayLater |

**Số câu có chunk chứa đúng bằng chứng trong top-3: 5 / 5.**

**Số câu có chunk chứa đúng bằng chứng ở top-1: 5 / 5.**

**Số câu Agent trả lời đúng gold answer: 5 / 5.**

### Hiệu quả metadata filter và failure analysis

Lần chạy đầu, câu đăng ký người bán kéo kết quả Amazon/TikTok lên trước và phần `Cách 4: SPayLater` bị kéo dài sang chương đăng ký tài khoản. Nguyên nhân là chunker chưa nhận diện tiêu đề chương La Mã. Sau khi bổ sung ranh giới `V.`, `VI.`... và metadata `platform`, các điều khoản thanh toán, đăng ký và quản lý thông tin được tách đúng chương.

Câu 2 và câu 5 là câu hỏi nhiều ý; bằng chứng nằm trong hai hoặc ba chunk cùng chủ đề. Vì vậy agent dùng top-3 thay vì chỉ top-1. Bộ lọc theo `section` được dùng cho thời hạn đổi trả và hàng cấm; các nhãn chủ đề dẫn xuất từ nội dung được dùng cho điều kiện người bán, phí hoàn trả và thanh toán online. Kết quả chạy thật đạt 5/5 top-1 liên quan, 5/5 top-3 chứa đủ bằng chứng và 5/5 câu trả lời đúng gold answer.

---

## Tự đánh giá phần cá nhân

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
