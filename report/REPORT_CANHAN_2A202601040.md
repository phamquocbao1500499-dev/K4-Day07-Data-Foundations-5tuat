# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Sỹ Mạnh Cường  
**Nhóm:** Nhóm K4 E-Commerce  
**Ngày:** 03/08/2026  

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Độ tương tự cosine cao (gần 1.0) thể hiện hai góc của hai vector embedding trong không gian đa chiều rất nhỏ, nghĩa là hai đoạn văn bản có sự đồng nhất cao về mặt **hướng nghĩa / chủ đề ngữ nghĩa**, bất kể độ dài ngắn của đoạn văn.

**Ví dụ có độ tương tự CAO:**
- Câu A: "Chính sách đổi trả hàng áp dụng trong vòng 7 ngày kể từ khi nhận hàng."
- Câu B: "Khách hàng có quyền hoàn trả sản phẩm trong 7 ngày tính từ lúc giao thành công."
- Tại sao tương đồng: Cả hai câu cùng truyền tải cùng một quy định pháp lý về thời hạn hoàn trả hàng hóa 7 ngày.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Quy trình nộp thuế thu nhập doanh nghiệp năm 2026."
- Câu B: "Hướng dẫn cài đặt ứng dụng mua sắm trên thiết bị di động iOS."
- Tại sao khác: Hai câu thuộc hai miền kiến thức hoàn toàn tách biệt (Pháp lý thuế vs Kỹ thuật phần mềm di động).

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Khoảng cách Euclid bị ảnh hưởng mạnh bởi **độ dài của văn bản** (mô đun/chiều dài vector). Hai văn bản cùng chủ đề nhưng một văn bản dài 500 từ và một văn bản dài 50 từ sẽ có khoảng cách Euclid rất xa. Trong khi đó, độ tương tự cosine chuẩn hóa độ dài vector (chỉ đo góc), giúp đánh giá chính xác **sự tương đồng về ngữ nghĩa** độc lập với độ dài.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> Khoảng bước di chuyển của cửa sổ trượt (step size) = `chunk_size - overlap` = `500 - 50 = 450` ký tự.  
> Số chunk tạo ra = `ceil((10,000 - 500) / 450) + 1` = `ceil(9500 / 450) + 1` = `ceil(21.11) + 1` = `22 + 1 = 23` chunks.  
> *Đáp án:* **23 chunks**.

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế me nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Bước di chuyển sẽ giảm từ 450 xuống `500 - 100 = 400` ký tự, làm số lượng chunk tăng lên `ceil(9500 / 400) + 1 = 25` chunks. Tăng overlap giúp bảo toàn ngữ cảnh ở ranh giới giữa các chunk, tránh việc một câu hoặc một điều khoản bị cắt đôi làm mất ý nghĩa khi thực hiện truy xuất.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng biểu thức chính quy `re.split(r'(?<=\. |\! |\? |\.\n)', text)` để tách văn bản theo ranh giới kết thúc câu. Xử lý khoảng trắng thừa bằng `.strip()`, sau đó gom các câu lại thành từng chunk có số câu không vượt quá `max_sentences_per_chunk`.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thuật toán chia đệ quy theo danh sách các dấu phân cách ưu tiên `["\n\n", "\n", ". ", " ", ""]`. Hàm `_split` kiểm tra nếu độ dài văn bản hiện tại `<= chunk_size` thì trả về ngay (base case). Ngược lại, thử tách bằng phân cách hiện tại; các mảnh quá dài tiếp tục được gọi đệ quy với danh sách dấu phân cách còn lại.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Lưu trữ văn bản dưới dạng danh sách các bản ghi gồm `id`, `content`, `metadata`, và `embedding`. Khi gọi `search`, câu truy vấn được nhúng thành vector `query_emb`, sau đó tính độ tương tự cosine với tất cả bản ghi bằng hàm `compute_similarity`, sắp xếp giảm dần theo điểm số (`score`) và trả về `top_k` kết quả.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Lọc (filter) dữ liệu **trước** khi tính toán tương đồng (pre-filtering): duyệt qua kho lưu trữ, chỉ giữ lại các bản ghi thỏa mãn tất cả tiêu chuẩn trong `metadata_filter`, sau đó mới chạy `_search_records`. Hàm `delete_document` tìm và xóa tất cả chunk có `id` hoặc `metadata['doc_id']` khớp với `doc_id` cần xóa.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `self.store.search(question, top_k)` để lấy `top_k` đoạn ngữ cảnh liên quan nhất, nối chúng lại bằng phân cách `\n---\n`. Sau đó xây dựng prompt dạng `Context:\n{context}\n\nQuestion: {question}\nAnswer:` và truyền vào hàm `llm_fn` để tạo câu trả lời.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: D:\VIN.AI\VIN_Labs\K4-Day07-Data-Foundations-5tuat
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
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 1.30s ==============================
```

**Số lượng bài test vượt qua (pass):** **42** / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Quy định đổi trả hàng trong 7 ngày | Khách hàng được trả hàng trong vòng một tuần | cao | 0.8842 | Đúng |
| 2 | Hướng dẫn mở shop trên sàn TMĐT | Cách tạo tài khoản bán hàng trực tuyến | cao | 0.8510 | Đúng |
| 3 | Thời gian giao hàng dự kiến 3 ngày | Chính sách bảo hành thiết bị điện tử | thấp | 0.1205 | Đúng |
| 4 | Điều khoản dịch vụ người bán TikTok Shop | Quy chế hoạt động sàn Shopee | cao | 0.6120 | Đúng |
| 5 | Hướng dẫn mua hàng trả góp 0% | Danh mục sản phẩm bị cấm bán | thấp | 0.1850 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 4 (Điều khoản TikTok Shop vs Quy chế Shopee) đạt điểm tương đồng khá cao (0.612) dù hai sàn thương mại điện tử khác nhau. Điều này cho thấy mô hình nhúng hiểu được mối liên hệ khái niệm cốt lõi (đều là văn bản pháp lý / quy chế vận hành nhà bán trên sàn TMĐT) vượt qua ranh giới tên thương hiệu cụ thể.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân trong gói `src` (dùng bộ dữ liệu `data/k4_ecommerce/` và `metadata_filter`).

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Thời hạn yêu cầu trả hàng và hoàn tiền dành cho người mua là bao nhiêu ngày? | Đơn yêu cầu trả hàng/hoàn tiền Shopee áp dụng trong 7 ngày kể từ khi giao thành công... | 0.7850 | Có | Người mua có 7 ngày để yêu cầu trả hàng/hoàn tiền kể từ lúc nhận hàng. |
| 2 | Nhà bán hàng có nghĩa vụ gì về việc bảo mật thông tin tài khoản và bồi thường cho sàn Tiki? | Nhà Bán có nghĩa vụ bảo mật tài khoản/mật khẩu, bảo mật thông tin khách hàng và bồi thường thiệt hại cho Tiki... | 0.8210 | Có | Nhà bán phải bảo mật tài khoản/thông tin và bồi thường cho Tiki nếu xảy ra thiệt hại liên quan. |
| 3 | Người bán trên Etsy bị cấm bán những mặt hàng nào? | Prohibited items include IP-infringing goods, counterfeit, weapons, hate speech, and reselling non-handmade... | 0.8430 | Có | Etsy cấm bán hàng giả, vi phạm SHTT, vũ khí, chất cấm và hàng tái bán không thuộc đồ thủ công/cổ. |
| 4 | Những hành vi nào bị coi là gian lận khi bán hàng trên TikTok Shop? | Hành vi lạm dụng khuyến mãi, tự đặt đơn/đánh giá giả, kéo khách giao dịch ngoài sàn và tạo nhiều tài khoản... | 0.7960 | Có | Gian lận bao gồm tự đặt đơn/đánh giá giả, lạm dụng voucher và lôi kéo khách ngoài sàn. |
| 5 | Khi người mua gặp sự cố với đơn hàng Etsy, quy trình bảo vệ người mua Purchase Protection hỗ trợ ra sao? | Etsy Purchase Protection refund buyers for items not delivered, damaged, or not as described... | 0.8110 | Có | Chương trình hỗ trợ hoàn tiền đầy đủ cho đơn hàng bị thất lạc, hư hỏng hoặc sai mô tả. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5** / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Việc kết hợp **Metadata Pre-filtering** (`customer_role: "seller"` / `"buyer"`) trước khi tính cosine similarity giúp triệt tiêu hoàn toàn nhiễu từ các văn bản dành cho đối tượng khác, làm tăng đáng kể độ chính xác Top-1 của mô hình RAG.

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
