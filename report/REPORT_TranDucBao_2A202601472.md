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

> **Cập nhật:** `report/REPORT_NHOM.md` của nhóm nay đã hoàn thiện (5 tài liệu chính thức + 5 câu hỏi đánh giá đã chốt, chiến lược của tôi được ghi nhận ở mục 2: `RecursiveChunker` đệ quy ưu tiên bảo toàn ranh giới câu). Bảng dưới đây thay thế bản demo tạm trước đó — chạy lại bằng đúng package cá nhân `src/TranDucBao_2A202601472`, trên **đúng 5 tài liệu chính thức** của nhóm (`tra-hang-hoan-tien-shopee`, `cam-han-che-shopee`, `quy-che-hoat-dong-shopee`, `k4-returns-policy`, `k4-seller-listing`) và **nguyên văn 5 câu hỏi đánh giá chính thức** (mục 3, REPORT_NHOM.md).
>
> Cấu hình: `RecursiveChunker(chunk_size=300)` → **521 chunk**, `EMBEDDING_PROVIDER=local` (`LocalEmbedder`, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) — embedding ngữ nghĩa thật, không dùng mock.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Score | Liên quan? |
|---|-------|--------------------------------|-------|-----------|
| 1 | Thời hạn người mua gửi yêu cầu đổi trả sản phẩm là bao lâu? | "Người Mua có thể gửi yêu cầu trả hàng/hoàn tiền trong vòng 15 ngày kể từ lúc đơn hàng được cập nhật giao hàng thành công" (`tra-hang-hoan-tien-shopee::chunk_15`) | 0.796 | ✅ Có — **trùng khớp tuyệt đối** với chunk gold mà nhóm đã chốt (`chunk_15`), dù tôi chunk độc lập bằng `chunk_size` riêng |
| 2 | Những sản phẩm nào bị cấm bán hoàn toàn trên Shopee? | "2. Danh sách sản phẩm cấm giao dịch... Hàng vi phạm bản quyền: hàng nhái, hàng giả..." (`quy-che-hoat-dong-shopee::chunk_219`) | 0.804 | ✅ Có — chunk gold của nhóm (`cam-han-che-shopee::chunk_0`) xếp hạng 2 sát nút (0.778); hai tài liệu trùng chủ đề nên hệ thống chọn bản liệt kê chi tiết hơn làm top-1 |
| 3 | Điều kiện để Người Bán đăng ký tài khoản và bán hàng là gì? (lọc `customer_role=seller`) | "DANH SÁCH SẢN PHẨM BÁN CÓ ĐIỀU KIỆN... Người Bán chỉ được phép đăng bán khi cung cấp đầy đủ giấy phép hợp lệ" (`cam-han-che-shopee::chunk_5`) | 0.693 | ⚠️ Liên quan một phần — trả lời "điều kiện của **sản phẩm**", không đúng trọng tâm "điều kiện **đăng ký tài khoản**" (xem phân tích bên dưới) |
| 4 | Chi phí vận chuyển chiều hoàn trả sản phẩm do ai chịu? | "7. TRÁCH NHIỆM VỀ CHI PHÍ VẬN CHUYỂN HOÀN TRẢ SẢN PHẨM CỦA NGƯỜI BÁN — 7.1. Người Bán sẽ chịu chi phí vận chuyển..." (`tra-hang-hoan-tien-shopee::chunk_61`) | 0.765 | ✅ Có — đúng nội dung với gold answer của nhóm (chunk id khác `chunk_88` chỉ vì `chunk_size` khác nhau đánh số lại toàn bộ) |
| 5 | Các phương thức thanh toán online được chấp nhận trên Shopee? | "Một số trường hợp Người Mua có nhu cầu trả hàng/hoàn tiền sau thời hạn trên..." (`tra-hang-hoan-tien-shopee::chunk_17`) | 0.714 | ❌ Không — top-3 không chứa chunk liệt kê đầy đủ phương thức thanh toán (xem phân tích bên dưới) |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 4 / 5 — Câu 1, 2, 4 đúng rõ ràng; Câu 3 liên quan nhưng lệch trọng tâm; **Câu 5 là trường hợp thất bại thật sự**, không phải giả định.

**Hai phát hiện đáng chú ý khi phân tích sâu hơn (đúng tinh thần "chỉ ra khi nào retrieval thất bại" của README):**

1. **Câu 3 — bộ lọc metadata quá khắt khe làm mất kết quả tốt nhất.** Khi bỏ bộ lọc để so sánh, chunk trả lời đúng và đầy đủ nhất thực ra là `quy-che-hoat-dong-shopee::chunk_15` — *"Người Bán bắt buộc phải đăng ký tài khoản và cung cấp các thông tin theo quy định của pháp luật..."* — với score **0.851** (cao hơn hẳn kết quả có lọc, 0.693). Nhưng tài liệu này gắn `customer_role: both` chứ không phải `seller`, nên bộ lọc `search_with_filter(metadata_filter={"customer_role": "seller"})` — dùng so khớp tuyệt đối (`==`) — loại nó ra ngay từ vòng lọc, trước cả khi tính similarity. Đây là bằng chứng cụ thể cho câu hỏi "Bộ lọc có quá khắt khe không?" trong `docs/EVALUATION.md`: lọc equality an toàn khi metadata tách biệt rõ ràng, nhưng loại bỏ oan các tài liệu dùng nhãn `both` (áp dụng cho cả hai vai trò) — hướng khắc phục hợp lý là coi `both` như khớp với mọi giá trị filter thay vì so khớp chuỗi thuần tuý.
2. **Câu 5 — chunk đúng nhất bị "chìm" ngoài top-3.** Chunk mô tả đầy đủ nhất — `quy-che-hoat-dong-shopee::chunk_100`: *"Thanh toán online qua Ví điện tử ShopeePay, ApplePay, Google Pay hoặc thẻ tín dụng/ghi nợ..."* — chỉ xếp hạng **#7/521** (score 0.650). Nguyên nhân: nhiều đoạn khác trong tài liệu chính sách hoàn tiền cũng nhắc tới "ShopeePay"/"thanh toán" trong ngữ cảnh hoàn tiền (không phải liệt kê phương thức), và có điểm ngữ nghĩa gần tương đương (0.70–0.71) nên chiếm hết top-3. Bài học: câu hỏi liệt kê ("các phương thức nào") dễ bị các đoạn *nhắc tới cùng từ khoá nhưng khác ý định* (hoàn tiền vs. liệt kê) cạnh tranh vị trí — muốn khắc phục triệt để cần `top_k` lớn hơn kèm bước lọc/rerank theo ý định câu hỏi, chứ tăng `top_k` từ 3 lên 5 vẫn chưa đủ (rank thật là #7).

**Điều hay nhất tôi học được từ thành viên khác trong nhóm (theo bảng so sánh đã hoàn thiện ở REPORT_NHOM.md):**
> Chiến lược của Phạm Quốc Bảo — `RecursiveChunker(chunk_size=300)` **không** ưu tiên thêm ranh giới câu — đạt 10/10, cao hơn biến thể của tôi (9.5/10, có ưu tiên ranh giới câu bổ sung). Với văn bản đã có cấu trúc đoạn/mục rõ ràng như quy chế Shopee, separator mặc định `\n\n`/`\n` đã đủ tốt; công sức "gia cố" thêm ranh giới câu của tôi không mang lại lợi ích rõ rệt trên loại văn bản này (có thể hữu ích hơn với văn bản ít xuống dòng, nhiều câu dài liền mạch). Ngược lại, so với các chiến lược đơn giản hơn — `FixedSizeChunker` của Phạm Công Đạt (7.5/10) hay `SentenceChunker` của Nguyễn Sỹ Mạnh Cường/Trần Hoàng Long (8.0–8.5/10) — hai phát hiện của tôi ở Câu 3 và Câu 5 cho thấy điểm nghẽn thực tế nhiều khi **không nằm ở chunking** mà ở **thiết kế metadata quá cứng nhắc** và **độ nhiễu ngữ nghĩa giữa các ý định câu hỏi khác nhau** — góc nhìn mà REPORT_NHOM.md chưa đề cập, tôi sẽ đề xuất bổ sung vào mục "Nếu làm lại" của nhóm.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 9 / 10 (đã chạy bằng đúng bộ tài liệu + 5 câu hỏi chính thức của nhóm; 4/5 câu có chunk liên quan trong top-3; phân tích được 2 hạn chế thật — lọc metadata quá khắt khe ở Câu 3, chunk đúng nhất "chìm" ngoài top-3 ở Câu 5 — nhưng chưa kịp thử nghiệm hướng khắc phục) |
| **Tổng phần cá nhân** | **59 / 60** |
