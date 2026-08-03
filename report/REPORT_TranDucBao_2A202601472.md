# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Trần Đức Bảo
**Nhóm:** 5tuat
**Ngày:** 3/8/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

> **Lưu ý về vị trí mã nguồn:** Toàn bộ phần cài đặt của tôi nằm trong `src/TranDucBao_2A202601472/` (bản sao package cá nhân, cùng cấu trúc và import tương đối như `src/`), không sửa `src/` gốc. Chạy test với bản của tôi bằng: `LAB_SOLUTION_PACKAGE=src.TranDucBao_2A202601472 pytest tests/ -v`.

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao nghĩa là hai vector có hướng gần nhau (góc giữa chúng nhỏ, cosine của góc gần 1), tức hai đoạn văn bản mà chúng biểu diễn mang ý nghĩa/ngữ cảnh tương đồng nhau, bất kể độ dài hay cách diễn đạt khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Con chó là loài vật trung thành và thường được nuôi làm thú cưng."
- Câu B: "Chó là động vật gắn bó với con người, hay được nuôi trong nhà."
- Tại sao tương đồng: Hai câu cùng nói về chủ đề "chó" và diễn đạt cùng một ý (loài vật trung thành, được nuôi trong nhà) dù dùng từ ngữ khác nhau, nên vector embedding của chúng có hướng gần nhau → cosine similarity cao.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Con chó là loài vật trung thành và thường được nuôi làm thú cưng."
- Câu B: "Thị trường chứng khoán hôm nay giảm điểm do lo ngại lạm phát."
- Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau (động vật vs. tài chính), không chia sẻ khái niệm hay ngữ cảnh, nên vector embedding lệch hướng nhiều → cosine similarity thấp (gần 0).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Vì cosine chỉ quan tâm đến hướng (góc) của vector chứ không bị ảnh hưởng bởi độ lớn (magnitude), trong khi văn bản dài/ngắn khác nhau có thể tạo ra embedding với magnitude khác nhau dù ngữ nghĩa tương tự — Euclidean distance sẽ bị lệch bởi yếu tố độ dài này còn cosine thì không.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:
> Với `FixedSizeChunker`, bước trượt (step) = `chunk_size - overlap` = 500 - 50 = 450. Cửa sổ bắt đầu tại các vị trí `start = 0, 450, 900, ...` (bội số của 450), vòng lặp dừng khi `start + chunk_size >= 10000`, tức `start >= 9500`. Bội số nhỏ nhất của 450 thỏa điều kiện này là `9900` (vì `9450 + 500 = 9950 < 10000` chưa đạt, còn `9900 + 500 = 10400 ≥ 10000` thì dừng). Số điểm bắt đầu từ 0 đến 9900 (bước 450) là `9900 / 450 + 1 = 23`.
> Đáp án: **23 chunks** (chunk cuối chỉ còn 100 ký tự: `text[9900:10000]`). Đã kiểm chứng lại bằng cách chạy trực tiếp `FixedSizeChunker(chunk_size=500, overlap=50).chunk("a"*10000)` → `len(...) == 23`.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Overlap tăng lên 100 → step giảm còn 400 → số chunk tăng từ 23 lên **25** (đã kiểm chứng bằng code). Overlap lớn hơn giúp giữ ngữ cảnh liên tục qua ranh giới chunk (một câu/ý bị cắt ở cuối chunk này vẫn xuất hiện trọn vẹn ở đầu chunk kế tiếp), giảm rủi ro mất thông tin khi truy xuất, đổi lại phải trả giá bằng nhiều chunk hơn (tốn lưu trữ + chi phí embedding hơn).

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Tôi dùng regex `(?<=[.!?])\s+` để tách câu: lookbehind khớp khi ký tự ngay trước là `.`, `!` hoặc `?`, rồi tách tại khoảng trắng/xuống dòng theo sau — bao quát cả ba dạng phân cách nêu trong docstring (". ", "! ", "? ", ".\n") mà không cần liệt kê từng trường hợp. Sau khi tách, tôi lọc bỏ chuỗi rỗng và `strip()` từng câu để tránh chunk chứa khoảng trắng thừa. Edge case xử lý: văn bản rỗng trả về `[]` ngay từ đầu, và văn bản không có dấu câu kết thúc thì `re.split` trả nguyên văn bản như 1 câu duy nhất.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán đệ quy thử lần lượt các separator theo thứ tự ưu tiên (`\n\n → \n → ". " → " " → ""`): nếu văn bản hiện tại đã ≤ `chunk_size` thì dừng (base case), nếu không thì tách theo separator đầu tiên có xuất hiện trong văn bản, rồi gộp các phần lại thành từng "buffer" sao cho không vượt `chunk_size`; phần nào một mình đã lớn hơn `chunk_size` thì được đệ quy tiếp với separator kế tiếp trong danh sách. Base case thứ hai: hết separator (hoặc gặp separator rỗng `""`) thì cắt cứng theo từng khối `chunk_size` ký tự — đảm bảo luôn trả về danh sách non-empty kể cả khi không tìm được điểm cắt tự nhiên nào.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được chuẩn hoá qua `_make_record`: embed `content` bằng `self._embedding_fn`, và gắn `metadata["doc_id"]` mặc định bằng `doc.id` nếu tài liệu chưa có sẵn (để tương thích cả khi thêm document trực tiếp lẫn khi nhận chunk đã có `doc_id` từ `ingest.py`). Với backend in-memory (mặc định vì không cài `chromadb`), record được append vào `self._store`; nếu `chromadb` có sẵn thì dùng `collection.add(...)`. `search` embed câu truy vấn rồi tính dot-product (`_dot`) với embedding của từng record đã lưu — vì các embedder (`MockEmbedder`, `LocalEmbedder` với `normalize_embeddings=True`) đều trả vector đã chuẩn hoá về độ dài 1, nên dot-product ở đây tương đương cosine similarity — sort giảm dần theo score rồi cắt `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Tôi lọc theo metadata **trước**, sau đó mới chạy similarity search trên tập đã lọc — cách này rẻ hơn (không phải embed lại) và cho kết quả top-k chính xác *trong phạm vi đã lọc*, thay vì lọc sau top-k (dễ làm mất kết quả liên quan nếu chúng bị các kết quả không thuộc bộ lọc chiếm hết vị trí). `delete_document` duyệt `self._store` và giữ lại các record có `metadata["doc_id"] != doc_id`, trả `True` nếu kích thước store giảm sau khi lọc (tức có ít nhất 1 chunk bị xoá), `False` nếu không tìm thấy `doc_id`.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> `__init__` chỉ lưu tham chiếu tới `store` và `llm_fn`. `answer` gọi `store.search(question, top_k=top_k)` để lấy các chunk liên quan, nối `content` của chúng lại bằng `"\n\n"` làm phần ngữ cảnh, rồi dựng một prompt tiếng Việt có cấu trúc rõ ràng gồm 3 phần: hướng dẫn cho LLM (kèm yêu cầu nói rõ nếu ngữ cảnh không đủ, để tránh model bịa (hallucination) khi retrieval không tìm được thông tin phù hợp) → khối "Ngữ cảnh" (context) → câu hỏi gốc. Cuối cùng gọi `llm_fn(prompt)` và trả thẳng kết quả.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

Lệnh chạy: `LAB_SOLUTION_PACKAGE=src.TranDucBao_2A202601472 pytest tests/ -v` (biến `LAB_SOLUTION_PACKAGE` trỏ bộ test vào package cá nhân thay vì `src` gốc — mặc định của biến này là `src`).

```
============================= test session starts =============================
platform win32 -- Python 3.12.1, pytest-9.1.1, pluggy-1.6.0
rootdir: E:\VinAI\K4-Day07-Data-Foundations-5tuat
plugins: anyio-4.14.2
collecting ... collected 42 items

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
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.07s ==============================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Dùng `LocalEmbedder` (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, đặt `EMBEDDING_PROVIDER=local`) thay vì mock, vì mock chỉ xác định theo hash của chuỗi và không phản ánh ngữ nghĩa thật (theo đúng lưu ý trong README).

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Con chó là loài vật trung thành và thường được nuôi làm thú cưng." | "Chó là động vật gắn bó với con người, hay được nuôi trong nhà." | cao | 0.8015 | ✅ |
| 2 | "Đơn hàng của tôi khi nào được giao?" | "Thời gian giao hàng dự kiến là bao lâu?" | cao | 0.6823 | ✅ |
| 3 | "Tôi muốn đổi trả sản phẩm bị lỗi." | "Chính sách hoàn tiền khi hàng không đúng mô tả như thế nào?" | cao | 0.3561 | ❌ (thực tế chỉ ở mức trung bình) |
| 4 | "Người bán cần cung cấp thông tin sản phẩm chính xác." | "Hôm nay thời tiết Hà Nội khá mát mẻ." | thấp | 0.0534 | ✅ |
| 5 | "Sàn thương mại điện tử bảo vệ thông tin cá nhân người dùng ra sao?" | "Chính sách quyền riêng tư quy định gì về dữ liệu khách hàng?" | cao | 0.7117 | ✅ |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Bất ngờ nhất là cặp 3: tôi dự đoán "cao" vì cả hai câu đều xoay quanh chủ đề đổi trả/hoàn tiền, nhưng điểm thực tế chỉ 0.356 — thấp hơn nhiều so với cặp 2 dù cặp 2 cũng chỉ là hai cách hỏi cùng một ý. Điều này cho thấy embedding không chỉ bắt "từ khóa chủ đề" chung mà còn nhạy với *vai trò/ý định* của câu: câu A là hành động của người mua ("tôi muốn đổi trả"), câu B là câu hỏi về chính sách của người bán ("chính sách hoàn tiền quy định gì") — hai *loại phát ngôn* khác nhau (tuyên bố ý định vs. câu hỏi tra cứu chính sách) dù cùng miền chủ đề, nên vector lệch hướng nhiều hơn dự đoán trực giác dựa trên từ khóa.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

> **Lưu ý quan trọng:** Tại thời điểm viết báo cáo này, `report/REPORT_NHOM.md` của nhóm **chưa được điền** (bộ tài liệu thật và 5 câu hỏi đánh giá chính thức của nhóm chưa chốt). Vì vậy tôi tạm chạy demo trên **bộ dữ liệu khởi động** `data/k4_ecommerce/` (2 tài liệu mẫu: `returns-policy.md`, `seller-listing.md`) với 5 câu hỏi minh hoạ tự đặt theo đúng chủ đề K4 (đổi trả + điều kiện người bán), để chứng minh pipeline cá nhân hoạt động đúng end-to-end (`load_documents → chunk → EmbeddingStore → search/search_with_filter → KnowledgeBaseAgent`). **Bảng này cần chạy lại bằng 5 câu hỏi chính thức + bộ tài liệu đầy đủ (5–10 tài liệu) khi nhóm hoàn thiện `REPORT_NHOM.md`.**
>
> Cấu hình demo: `EMBEDDING_PROVIDER=local`, `RecursiveChunker(chunk_size=300)`, tổng 5 chunk được nạp từ 2 tài liệu.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người mua cần làm gì để yêu cầu đổi trả hàng? | "Người mua cần gửi yêu cầu đổi trả trong thời hạn được nêu trên trang sản phẩm..." (`k4-returns-policy`) | 0.681 | ✅ Có | Trích đúng đoạn hướng dẫn gửi yêu cầu đổi trả + bằng chứng kèm theo |
| 2 | Ai chịu trách nhiệm phản hồi yêu cầu đổi trả? | "Người bán có trách nhiệm phản hồi theo quy trình của sàn..." (`k4-returns-policy`) | 0.591 | ✅ Có | Trả lời đúng: người bán chịu trách nhiệm phản hồi |
| 3 | Người bán cần cung cấp những thông tin gì khi đăng bán sản phẩm? | "Người bán chịu trách nhiệm cung cấp thông tin sản phẩm chính xác, bao gồm giá, mô tả, tình trạng hàng..." (`k4-seller-listing`) | 0.763 | ✅ Có | Trả lời đúng: giá, mô tả, tình trạng hàng |
| 4 | Sản phẩm như thế nào thì không được phép đăng bán? (lọc `category=listing`) | "...Sản phẩm bị hạn chế hoặc bị cấm không được đăng bán." (`k4-seller-listing`) | 0.700 | ✅ Có | Trả lời đúng, và bộ lọc metadata loại bỏ hoàn toàn 2 chunk thuộc `returns-policy` khỏi kết quả |
| 5 | Khi hàng không đúng mô tả thì người mua cần làm gì? (lọc `category=returns`) | "Người mua cần gửi yêu cầu đổi trả... kèm bằng chứng phù hợp khi hàng bị lỗi hoặc không đúng mô tả." (`k4-returns-policy`) | 0.627 | ✅ Có | Trả lời đúng, bộ lọc loại chunk của `seller-listing` |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5 (trên bộ dữ liệu khởi động 2 tài liệu — cần đánh giá lại nghiêm túc hơn khi có bộ 5–10 tài liệu thật của nhóm, vì với chỉ 5 chunk trong store thì bài toán truy xuất gần như không có "nhiễu" để phân biệt).

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Cần cập nhật sau buổi thuyết trình/so sánh trong nhóm — hiện REPORT_NHOM.md và phần demo chéo giữa các thành viên chưa diễn ra tại thời điểm viết báo cáo này.*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 7 / 10 (demo tạm trên dữ liệu khởi động, chưa dùng bộ câu hỏi + tài liệu chính thức của nhóm) |
| **Tổng phần cá nhân** | **57 / 60** |
