# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Công Đạt  
**Mã sinh viên:** 2A202601406  
**Nhóm:** 5tuat  
**Ngày:** 03/08/2026

> Package bài làm: `src.PhamCongDat_2A202601406`. Báo cáo nhóm trong repository hiện vẫn là mẫu trống; vì vậy phần benchmark cuối báo cáo là kết quả cá nhân tạm thời trên dữ liệu khởi động, cần được thay bằng bộ 5 câu hỏi thống nhất của nhóm trước khi nộp chính thức.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Bài tập 1.1)

**Độ tương tự cosine cao nghĩa là gì?**  
Hai vector embedding hướng gần giống nhau, cho thấy hai đoạn văn có nội dung hoặc ý nghĩa ngữ nghĩa gần nhau, dù từ ngữ cụ thể có thể khác.

**Ví dụ có độ tương tự CAO:**

- Câu A: “Tôi muốn đổi trả sản phẩm bị lỗi.”
- Câu B: “Làm thế nào để hoàn hàng khi sản phẩm bị hỏng?”
- Lý do: cả hai cùng hỏi về việc trả một sản phẩm có lỗi.

**Ví dụ có độ tương tự THẤP:**

- Câu A: “Chính sách đổi trả bảo vệ người mua.”
- Câu B: “Hôm nay thời tiết rất đẹp.”
- Lý do: hai câu nói về hai chủ đề không liên quan.

**Tại sao cosine thường được ưu tiên hơn khoảng cách Euclid?**  
Cosine so sánh hướng của vector và ít bị ảnh hưởng bởi độ lớn, nên tập trung tốt hơn vào quan hệ ngữ nghĩa. Khoảng cách Euclid còn thay đổi theo độ lớn vector, dù hai vector có thể cùng biểu diễn một hướng ý nghĩa.

### Bài toán tính toán Chunking (Bài tập 1.2)

**10.000 ký tự, `chunk_size=500`, `overlap=50`:**

`ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22,11) = 23` chunks.

**Khi tăng `overlap` lên 100:**

`ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = ceil(24,75) = 25` chunks. Số chunk tăng từ 23 lên 25 vì bước trượt giảm từ 450 xuống 400 ký tự. Overlap lớn hơn giúp giữ ngữ cảnh tại ranh giới chunk, nhưng làm tăng dung lượng lưu trữ và chi phí embedding/truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ

**`SentenceChunker.chunk`:**  
Dùng regex `(?<=[.!?])(?:\s+)` để nhận diện khoảng trắng hoặc xuống dòng sau dấu kết thúc câu, đồng thời giữ dấu câu trong kết quả. Văn bản rỗng trả về danh sách rỗng; các câu được `strip`, bỏ phần tử rỗng và nhóm theo `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`:**  
Thử lần lượt các separator theo độ ưu tiên `\n\n`, `\n`, `. `, khoảng trắng rồi chuỗi rỗng. Phần vượt `chunk_size` được xử lý tiếp bằng separator kế tiếp; base case là phần đã đủ ngắn, hết separator, hoặc separator rỗng, khi đó cắt cứng theo kích thước. Dấu phân cách được giữ lại để nội dung không bị mất.

### Lớp `EmbeddingStore`

**`add_documents` + `search`:**  
Mỗi tài liệu được chuẩn hóa thành record có ID duy nhất, nội dung, metadata và embedding. Nếu ChromaDB khả dụng thì dùng collection tạm; nếu không thì lưu trong danh sách bộ nhớ. Khi tìm kiếm, vector truy vấn được so sánh với các record bằng dot product và sắp xếp điểm giảm dần.

**`search_with_filter` + `delete_document`:**  
Metadata được lọc trước khi xếp hạng để các ứng viên sai vai trò/chủ đề không chiếm top-k. `delete_document` xóa tất cả record có `metadata.doc_id` trùng giá trị cần xóa và trả về `True` chỉ khi thực sự có record bị xóa.

### Tác tử `KnowledgeBaseAgent`

**`answer`:**  
Agent lấy top-k chunk rồi ghép thành phần “Ngữ cảnh” có đánh số. Prompt yêu cầu LLM chỉ trả lời dựa trên ngữ cảnh, thừa nhận không biết khi dữ liệu không đủ, sau đó chèn nguyên câu hỏi và gọi `llm_fn`.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Đã hoàn thiện toàn bộ TODO trong package cá nhân, bao gồm chunking, cosine similarity, comparator, vector store, lọc metadata, xóa tài liệu và RAG agent.

### Kết quả kiểm thử

Lệnh chạy:

```powershell
$env:LAB_SOLUTION_PACKAGE='src.PhamCongDat_2A202601406'
python -m pytest tests -v
```

Kết quả:

```text
collected 42 items
42 passed in 0.10s
```

**Số lượng bài test vượt qua:** 42 / 42.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi dự đoán trước theo nghĩa của câu, sau đó chạy `_mock_embed` và `compute_similarity`. Mock embedder chỉ tạo vector xác định theo toàn chuỗi, không phải embedding ngữ nghĩa; vì vậy điểm dưới đây dùng để xác minh code, không dùng kết luận chất lượng tiếng Việt.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|---|---|---|---|---:|---|
| 1 | Tôi muốn đổi trả sản phẩm bị lỗi. | Làm thế nào để hoàn hàng khi sản phẩm bị hỏng? | Cao | 0,074766 | Không |
| 2 | Người bán phải mô tả sản phẩm chính xác. | Thông tin đăng bán cần đúng với tình trạng hàng. | Cao | -0,071941 | Không |
| 3 | Sản phẩm bị cấm không được đăng bán. | Người bán không được niêm yết hàng thuộc danh mục cấm. | Cao | -0,227776 | Không |
| 4 | Chính sách đổi trả bảo vệ người mua. | Hôm nay thời tiết rất đẹp. | Thấp | -0,037627 | Có |
| 5 | Python là một ngôn ngữ lập trình. | Người mua cần gửi bằng chứng khi hàng bị lỗi. | Thấp | 0,011802 | Có |

**Kết quả bất ngờ nhất:**  
Cặp 3 tương đương gần như hoàn toàn về nghĩa nhưng lại có điểm âm thấp nhất. Điều này minh họa giới hạn đã nêu trong README: mock embedder gần như ngẫu nhiên theo cả chuỗi và không biểu diễn ngữ nghĩa; benchmark chính thức phải chạy local multilingual embedder.

---

## 5. Kết quả truy xuất cá nhân tạm thời (Competition Results) — Cá nhân (10 điểm)

### Cấu hình thử nghiệm

- Dữ liệu: 2 tài liệu khởi động trong `data/k4_ecommerce` (5 chunk).
- Chiến lược: `SentenceChunker(max_sentences_per_chunk=2)`.
- Embedder: `_mock_embed` để chạy offline; điểm không phản ánh chất lượng ngữ nghĩa.
- Câu 5 lọc trước bằng `metadata_filter={"customer_role": "seller"}`.

| # | Câu hỏi | Top-1 chunk truy xuất (tóm tắt) | Score | Top-1 liên quan? | Câu trả lời có thể tổng hợp từ top-3 |
|---|---|---|---:|---|---|
| 1 | Người mua cần làm gì khi sản phẩm bị lỗi hoặc không đúng mô tả? | Quy định đăng bán: người bán phải cung cấp thông tin chính xác | -0,002824 | Không | Gửi yêu cầu đổi trả đúng thời hạn và kèm bằng chứng phù hợp |
| 2 | Người bán phải cung cấp những thông tin nào khi đăng sản phẩm? | Thông tin sản phẩm phải chính xác: giá, mô tả, tình trạng | 0,250545 | Có | Cung cấp chính xác giá, mô tả và tình trạng hàng |
| 3 | Sản phẩm bị hạn chế hoặc bị cấm có được đăng bán không? | Thông tin sản phẩm phải chính xác | 0,224394 | Không | Không được đăng bán sản phẩm bị hạn chế hoặc bị cấm |
| 4 | Ai có trách nhiệm phản hồi yêu cầu đổi trả? | Người bán có trách nhiệm phản hồi theo quy trình của sàn | 0,066145 | Có | Người bán chịu trách nhiệm phản hồi |
| 5 | Với vai trò người bán, trách nhiệm về thông tin sản phẩm là gì? | Không được đăng sản phẩm bị hạn chế hoặc bị cấm | 0,061476 | Không | Phải cung cấp thông tin chính xác về giá, mô tả và tình trạng hàng |

**Số câu có chunk liên quan trong top-3:** 5 / 5.  
**Số câu có chunk liên quan ở top-1:** 2 / 5.

**Nhận xét:**  
Metadata filter ở câu 5 loại toàn bộ chunk dành cho người mua, nhưng mock embedding vẫn xếp chunk “hàng cấm” trên chunk nói về độ chính xác thông tin. Đây là một failure case rõ ràng: metadata giúp thu hẹp đúng vai trò nhưng không thể sửa một mô hình embedding không hiểu ngữ nghĩa. Khi nhóm hoàn thiện corpus và 5 gold answers chung, cần chạy lại bằng `EMBEDDING_PROVIDER=local` rồi thay bảng tạm này.

---

## Tự đánh giá (phần cá nhân)

| Tiêu chí | Điểm tự đánh giá |
|---|---:|
| Khởi động | 5 / 5 |
| Hướng tiếp cận | 10 / 10 |
| Hoàn thiện code | 30 / 30 |
| Dự đoán độ tương tự | 5 / 5 |
| Kết quả truy xuất | Chờ benchmark chính thức của nhóm |
| **Tổng phần đã xác minh** | **50 / 50, chưa tính 10 điểm benchmark nhóm** |
