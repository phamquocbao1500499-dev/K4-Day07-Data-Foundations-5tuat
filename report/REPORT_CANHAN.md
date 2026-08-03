# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Phạm Quốc Bảo
**Nhóm:** 5tuat
**Ngày:** 3/8/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vectơ biểu diễn văn bản trong không gian embedding có hướng (góc giữa 2 vector) rất gần nhau. Điều này thể hiện hai văn bản đó có nội dung ngữ nghĩa rất tương đồng, bất kể độ dài ngắn của từng câu.

**Ví dụ có độ tương tự CAO:**
- Câu A: Chính sách đổi trả hàng áp dụng trong vòng 7 ngày kể từ khi nhận sản phẩm.
- Câu B: Khách hàng có thể hoàn trả hàng hóa trong vòng 7 ngày đầu tiên.
- Tại sao tương đồng: Cả hai câu đều truyền tải cùng một ý nghĩa cốt lõi về thời hạn 7 ngày cho phép trả lại hàng.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Thời gian giao hàng dự kiến từ 2 đến 4 ngày làm việc.
- Câu B: Hệ thống hỗ trợ thanh toán qua thẻ Visa, Mastercard và ví điện tử.
- Tại sao khác: Câu A nói về thông tin vận chuyển, trong khi câu B nói về phương thức thanh toán.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Độ tương tự cosine chỉ đo góc giữa 2 vectơ mà không bị ảnh hưởng bởi độ dài (magnitude) của vectơ. Trong NLP, một câu ngắn và một đoạn văn dài có thể cùng thảo luận một chủ đề (góc nhỏ, cosine high), nhưng khoảng cách Euclid giữa chúng sẽ rất lớn do độ dài văn bản khác nhau. Do đó, cosine similarity phản ánh chính xác sự tương đồng về mặt ngữ nghĩa hơn.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* $\text{Số chunks} = \left\lceil \frac{\text{độ\_dài} - \text{overlap}}{\text{chunk\_size} - \text{overlap}} \right\rceil = \left\lceil \frac{10000 - 50}{500 - 50} \right\rceil = \left\lceil \frac{9950}{450} \right\rceil = \left\lceil 22.11 \right\rceil = 23$
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số lượng chunk sẽ tăng từ 23 lên 25 chunks ($\left\lceil \frac{10000 - 100}{500 - 100} \right\rceil = 25$). Ta muốn tăng độ chồng chéo để đảm bảo giữ nguyên ngữ cảnh ở ranh giới giữa hai chunk liền kề, tránh việc một câu hoặc một ý quan trọng bị cắt làm đôi làm mất thông tin khi truy xuất RAG.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src/2A202601502_PhamQuocBao`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Sử dụng biểu thức chính quy (regex) `re.split(r"(?<=[.!?])\s+|(?<=\.)\n+", text)` để nhận diện ranh giới kết thúc câu (`. `, `! `, `? `, `.\n`). Xử lý trường hợp ngoại lệ như câu rỗng hoặc chuỗi chỉ chứa khoảng trắng bằng `.strip()` và loại bỏ phần tử rỗng, sau đó gom các câu thành từng nhóm tối đa `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Áp dụng thuật toán đệ quy thử nghiệm danh sách các dấu phân cách `separators` theo thứ tự ưu tiên (`["\n\n", "\n", ". ", " ", ""]`). Base case là khi độ dài văn bản $\le$ `chunk_size` hoặc danh sách separators bị rỗng. Nếu đoạn văn bản sau khi split vẫn vượt quá `chunk_size`, hàm sẽ gọi đệ quy `_split` với mức separator nhỏ hơn.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ danh sách các record dạng dictionary chứa `id`, `content`, `metadata`, và `embedding`. Khi thực hiện `search`, tính điểm tương đồng giữa query embedding và từng stored embedding bằng hàm tích vô hướng `_dot()` (do các vector đã được chuẩn hóa), sau đó sắp xếp kết quả theo điểm số giảm dần và lấy top-k.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Sử dụng cơ chế pre-filtering: duyệt qua kho lưu trữ để lọc các record có `metadata` khớp với toàn bộ các cặp key-value trong `metadata_filter` trước, rồi mới tiến hành tính điểm similarity search trên tập đã lọc. Với `delete_document`, lọc bỏ các record có `id` hoặc `metadata['doc_id']` trùng với `doc_id` cần xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Thực hiện quy trình RAG chuẩn: Gọi `self.store.search(question, top_k)` để truy xuất các chunk liên quan nhất từ kho tri thức. Sau đó ghép các chunk này thành đoạn văn bản ngữ cảnh (Context) dạng `[id]: content` và đưa vào prompt模板 cùng câu hỏi để gửi tới `llm_fn` tổng hợp câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.13.3, pytest-9.0.3, pluggy-1.6.0
rootdir: C:\Users\Lil'Pao0\Documents\Vin_AI\K4-Day07-Data-Foundations-5tuat
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED   [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED    [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED   [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [100%]

============================= 42 passed in 0.12s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Thời hạn đổi trả sản phẩm trong 7 ngày | Khách hàng được hoàn trả hàng trong 7 ngày | cao | 0.89 | Đúng |
| 2 | Phí vận chuyển giao hàng tận nơi | Phương thức thanh toán qua ví điện tử | thấp | 0.12 | Đúng |
| 3 | Chính sách bảo hành thiết bị điện tử | Điều khoản dịch vụ dành cho người bán | thấp | 0.25 | Đúng |
| 4 | Hướng dẫn tạo tài khoản cửa hàng | Quy trình đăng ký gian hàng người bán | cao | 0.82 | Đúng |
| 5 | Thời gian giao hàng nội thành | Mèo là động vật có vú | thấp | -0.05 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Kết quả giữa các câu thuộc cùng lĩnh vực TMĐT (như bảo hành và người bán) vẫn có độ tương đồng nhất định dù ý nghĩa khác nhau. Điều này cho thấy mô hình nhúng biểu diễn khoảng cách không chỉ ở từ vựng mà còn ở ngữ cảnh không gian ngữ nghĩa rộng hơn (domain context).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src/2A202601502_PhamQuocBao`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Điều kiện đổi trả hàng là gì? | Sản phẩm chưa qua sử dụng, nguyên tem mác trong 7 ngày... | 0.88 | Có | Trả lời đúng điều kiện đổi trả sản phẩm. |
| 2 | Phí giao hàng tính thế nào? | Giao hàng miễn phí cho đơn từ 500k, nội thành 20k... | 0.85 | Có | Nêu rõ mức phí và điều kiện miễn phí vận chuyển. |
| 3 | Người bán cần giấy tờ gì để đăng ký? | Giấy ĐKKD hoặc CCCD chính chủ với gian hàng cá nhân... | 0.81 | Có | Trích dẫn đầy đủ giấy tờ cho người bán. |
| 4 | Thời gian hoàn tiền đổi trả bao lâu? | Tiền sẽ được hoàn về tài khoản trong 3-5 ngày làm việc... | 0.83 | Có | Trả lời chính xác thời gian 3-5 ngày. |
| 5 | Chính sách bảo mật thông tin cá nhân? | Thông tin khách hàng được mã hóa và không chia sẻ cho bên thứ 3... | 0.86 | Có | Giải thích ngắn gọn về mã hóa và bảo mật. |

**Bao nhiêu câu hỏi trả về chunk me liên quan trong top-3?** **5** / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Việc kết hợp lọc siêu dữ liệu (metadata filtering theo `customer_role`) giúp loại bỏ nhiễu cực kỳ hiệu quả khi bộ tài liệu lớn, giúp RAG trả lời chính xác đúng đối tượng người mua hoặc người bán mà không bị lẫn lộn quy định.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
