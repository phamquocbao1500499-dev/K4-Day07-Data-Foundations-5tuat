# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Hoàng Long
**Nhóm:** 5tuat
**Ngày:** 3/8/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Nghĩa là 2 vector có hướng gần giống nhau trong không gian nhiều chiều. Tức là góc giữa chúng gần 0° (cosine similarity gần 1), bất kể độ dài (magnitude) của vector khác nhau ra sao. Áp dụng trong thực tế, so sánh 2 tài liệu mà cosine cao nghĩa là nội dung, ngữ nghĩa rất tương đồng nhau.*

**Ví dụ có độ tương tự CAO:**
- Câu A: "Tôi có thể đổi trả sản phẩm trong bao nhiêu ngày kể từ khi nhận hàng?"
- Câu B: "Thời hạn hoàn trả đơn hàng đã mua là bao lâu?"
- Tại sao tương đồng: Hai câu dùng từ ngữ khác nhau ("đổi trả" vs "hoàn trả", "bao nhiêu ngày" vs "bao lâu") nhưng cùng hỏi về **chính sách và thời hạn đổi/trả hàng**, nên embedding của chúng nằm gần nhau về mặt ngữ nghĩa.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Tôi có thể đổi trả sản phẩm trong bao nhiêu ngày kể từ khi nhận hàng?"
- Câu B: "Người bán cần cung cấp giấy tờ gì để mở gian hàng trên nền tảng?"
- Tại sao khác: Hai câu tuy cùng nằm trong chủ đề thương mại điện tử nhưng đề cập đến **hai chủ đề con hoàn toàn khác nhau** (chính sách đổi trả của người mua vs điều kiện đăng ký của người bán), nên vector embedding chỉ theo các hướng khác biệt.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Cosine chỉ quan tâm đến hướng của vector chứ không quan tâm độ dài, nên không bị ảnh hưởng bởi độ dài văn bản (câu dài hay ngắn đều được so sánh công bằng về mặt ngữ nghĩa). Trong khi đó, Euclidean distance bị chi phối bởi độ lớn (magnitude) của vector, nên hai văn bản cùng nghĩa nhưng có độ dài/tần suất từ khác nhau có thể bị đánh giá là "xa nhau" một cách sai lệch.*

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính: Mỗi chunk sau chunk đầu tiên chỉ "tiến thêm" (step) = chunk_size − overlap = 500 − 50 = 450 ký tự so với chunk trước (vì 50 ký tự cuối được lặp lại ở chunk kế tiếp). Số chunk = ceil((tổng_ký_tự − overlap) / (chunk_size − overlap)) = ceil((10000 − 50) / 450) = ceil(9950 / 450) = ceil(22.11) = 23.*

> *Đáp án:* **23 chunks** (chunk 1–22 có độ dài 500 ký tự với bước nhảy 450; chunk cuối cùng ngắn hơn vì phần dư của tài liệu ít hơn 500 ký tự).

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Overlap=100 → step = 500 − 100 = 400, số chunk = ceil((10000 − 100) / 400) = ceil(24.75) = **25 chunks** (tăng từ 23 lên 25). Muốn overlap nhiều hơn vì nó giảm nguy cơ một câu/ý quan trọng bị cắt đứt ngay tại ranh giới giữa hai chunk, giúp mỗi chunk giữ được nhiều ngữ cảnh hơn và cải thiện chất lượng truy xuất (retrieval), đổi lại là tốn thêm dung lượng lưu trữ và thời gian embedding do dữ liệu bị trùng lặp nhiều hơn.*

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Dùng `re.split` với pattern `r"(?<=[.!?])\s+"` (hoặc tương đương tách theo ". ", "! ", "? ", ".\n") để cắt văn bản thành danh sách câu, sau đó gom từng nhóm tối đa `max_sentences_per_chunk` câu lại thành một chunk bằng `" ".join(...)`. Edge case cần xử lý: chuỗi rỗng trả về `[]`, khoảng trắng thừa giữa các câu phải được `strip()`, và câu cuối cùng không có dấu kết thúc câu vẫn phải được giữ lại thay vì bị bỏ sót.*

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Thuật toán đệ quy: thử tách văn bản bằng separator đầu tiên trong danh sách ưu tiên (`\n\n`, `\n`, ". ", " ", ""); nếu một phần vẫn dài hơn `chunk_size`, gọi đệ quy `_split` trên phần đó với danh sách separator còn lại (bỏ separator vừa dùng). Base case là khi `remaining_separators` rỗng (dùng separator `""` để cắt cứng theo ký tự) hoặc khi đoạn văn bản hiện tại đã ngắn hơn hoặc bằng `chunk_size`, lúc đó trả về `[current_text]` luôn thay vì tách tiếp.*

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *`add_documents` duyệt qua từng `Document`, gọi `embedding_fn` để sinh vector rồi lưu thành record `{id, text, embedding, metadata}` — nếu có ChromaDB thì dùng `collection.add(ids=..., documents=..., embeddings=...)`, còn không thì append vào danh sách `self._store` trong bộ nhớ. `search` embed câu query bằng cùng `embedding_fn`, tính cosine similarity (qua `compute_similarity`/`_dot`) giữa vector query với từng vector đã lưu, sắp xếp giảm dần theo điểm số và trả về `top_k` kết quả cao nhất.*

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *`search_with_filter` lọc `metadata` trước (ví dụ theo `doc_id`, `source`) để thu hẹp tập ứng viên, rồi mới chạy similarity search trên tập đã lọc đó — tiết kiệm chi phí tính toán so với việc tính tương tự trên toàn bộ store rồi mới lọc. `delete_document` duyệt `self._store` và loại bỏ mọi record có `metadata['doc_id'] == doc_id`, trả về `True` nếu có ít nhất một chunk bị xóa, `False` nếu không tìm thấy chunk nào khớp.*

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *`answer` gọi `store.search(question, top_k)` để lấy các chunk liên quan nhất, ghép nội dung các chunk đó lại thành một khối "Context", rồi dựng prompt theo mẫu dạng "Dựa trên ngữ cảnh sau, hãy trả lời câu hỏi: {context}\n\nCâu hỏi: {question}" trước khi truyền cho `llm_fn`. Cách đưa ngữ cảnh vào là nối trực tiếp các đoạn chunk (kèm nguồn nếu cần) vào phần đầu prompt, đảm bảo LLM chỉ trả lời dựa trên thông tin được truy xuất thay vì bịa đặt (giảm hallucination).*

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
========================================================================================= test session starts ==========================================================================================
platform win32 -- Python 3.11.15, pytest-9.1.1, pluggy-1.6.0 -- E:\Vin\Vin lab\DAY7_2A202601646_TranHoangLong\K4-Day07-Data-Foundations-5tuat - bai lam rieng\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: E:\Vin\Vin lab\DAY7_2A202601646_TranHoangLong\K4-Day07-Data-Foundations-5tuat - bai lam rieng
collected 42 items                                                                                                                                                                                      

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                                                             [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                                                      [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                                                               [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                                                                [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                                                                     [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                                                                     [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                                                           [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                                                            [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                                                          [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                                                            [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                                                            [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                                                       [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                                                                   [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                                                             [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                                                                    [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                                                        [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                                                                  [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                                                        [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                                                            [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                                                              [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                                                                [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                                                      [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                                                           [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                                                             [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                                                                 [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                                                              [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                                                       [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                                                      [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                                                                 [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                                                             [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                                                        [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                                                            [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                                                                  [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                                                            [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                                                         [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                                                       [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                                                      [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                                                          [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                                                                     [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                                                              [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                                                                    [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED                                                                                        [100%]

========================================================================================== 42 passed in 0.16s ==========================================================================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Tôi có thể đổi trả sản phẩm trong bao nhiêu ngày kể từ khi nhận hàng? | Thời hạn hoàn trả đơn hàng đã mua là bao lâu? | cao | -0.0082 | Sai |
| 2 | Người bán cần cung cấp giấy tờ gì để mở gian hàng trên nền tảng? | Điều kiện đăng ký tài khoản bán hàng là gì? | cao | -0.0749 | Sai |
| 3 | Tôi có thể đổi trả sản phẩm trong bao nhiêu ngày kể từ khi nhận hàng? | Người bán cần cung cấp giấy tờ gì để mở gian hàng trên nền tảng? | thấp | 0.1156 | Đúng |
| 4 | Phí vận chuyển được tính như thế nào cho đơn hàng nội thành? | Sàn thương mại điện tử xử lý dữ liệu cá nhân của khách hàng ra sao? | thấp | -0.0514 | Đúng |
| 5 | Đơn hàng của tôi bị giao trễ so với ngày dự kiến, tôi phải làm sao? | Làm thế nào để khiếu nại khi shipper giao hàng chậm hơn cam kết? | cao | -0.0070 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Bất ngờ nhất là Cặp 1 và Cặp 5. Hai câu gần như đồng nghĩa, đều cùng hỏi về thời hạn đổi/trả hàng, hoặc cùng hỏi về cách xử lý khi giao hàng trễ, nhưng lại có điểm cosine gần 0 hoặc âm. Cặp 3, hai câu thuộc hai chủ đề hoàn toàn khác nhau (đổi trả của người mua vs mở gian hàng của người bán), lại có điểm cao nhất trong 5 cặp (0.1156). Nguyên nhân là dự án đang chạy với `MockEmbedder` (src/embeddings.py), một bộ sinh vector giả từ hash MD5 của chuỗi ký tự, không hề học ngữ nghĩa tiếng Việt nên các vector gần như ngẫu nhiên và gần trực giao với nhau bất kể nội dung câu nói gì. Điều này cho thấy embedding chỉ biểu diễn được ý nghĩa thật khi vector được sinh ra bởi một mô hình đã học ngữ nghĩa (ví dụ sentence-transformers hoặc OpenAI embedding); nếu dùng embedding "giả"/ngẫu nhiên thì independent của nội dung, mọi phép so sánh similarity đều vô nghĩa dù thuật toán tính cosine hoàn toàn đúng.*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 |  |  |  |  |  |
| 2 |  |  |  |  |  |
| 3 |  |  |  |  |  |
| 4 |  |  |  |  |  |
| 5 |  |  |  |  |  |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** __ / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | / 10 |
| **Tổng phần cá nhân** | **/ 60** |
