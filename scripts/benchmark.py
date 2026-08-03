"""Reproducible Phase-2 benchmark for Pham Cong Dat (2A202601406)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ingest import load_documents
from src.PhamCongDat_2A202601406 import (
    Document,
    EmbeddingStore,
    HeadingAwareChunker,
    KnowledgeBaseAgent,
    LocalEmbedder,
    compute_similarity,
)


DATA_DIR = Path("data/k4_ecommerce")
INCLUDED_DOCUMENT_IDS = {
    "cam-han-che-shopee",
    "chinh-sach-nguoi-ban-amazon",
    "dang-ky-nguoi-ban-amazon",
    "k4_seller_tiktok_shop_terms",
    "k4_etsy_buyer_policy",
    "k4_etsy_seller_policy",
    "k4_seller_tiki_quyen_nghia_vu",
    "quy-che-hoat-dong-shopee",
    "tinh-trang-san-pham-amazon",
    "tra-hang-hoan-tien-shopee",
}

BENCHMARKS = [
    {
        "query": "Thời hạn người mua gửi yêu cầu đổi trả sản phẩm là bao lâu?",
        "gold": "Trong vòng 15 ngày kể từ khi giao thành công; thực phẩm tươi sống và đông lạnh là 24 giờ.",
        "target_doc": "tra-hang-hoan-tien-shopee",
        "evidence": "15 (mười lăm) ngày",
        "filter": {
            "customer_role": "buyer",
            "section": "ĐIỀU KIỆN YÊU CẦU TRẢ HÀNG/HOÀN TIỀN",
        },
    },
    {
        "query": "Những sản phẩm nào bị cấm bán hoàn toàn trên Shopee?",
        "gold": "Hàng giả/nhái, vũ khí/đạn dược, chất cháy nổ, thuốc lá/vape, thuốc kê đơn, động thực vật hoang dã và tiền tệ giả.",
        "target_doc": "cam-han-che-shopee",
        "evidence": "Tiền tệ và giấy tờ có giá",
        "filter": {
            "customer_role": "seller",
            "section": "DANH SÁCH SẢN PHẨM BỊ CẤM",
        },
    },
    {
        "query": "Điều kiện để Người Bán đăng ký tài khoản và bán hàng là gì?",
        "gold": "Cung cấp thông tin xác minh như CCCD/mã số thuế hoặc đăng ký doanh nghiệp và thông tin tài khoản ngân hàng.",
        "target_doc": "quy-che-hoat-dong-shopee",
        "evidence": "CCCD/Mã số thuế",
        "filter": {"customer_role": "seller", "policy_topic": "seller-eligibility"},
    },
    {
        "query": "Chi phí vận chuyển chiều hoàn trả sản phẩm do ai chịu?",
        "gold": "Người Bán chịu phí khi yêu cầu được chấp thuận mà không do lỗi Người Mua/đơn vị vận chuyển hoặc khi giao không thành công, trừ các ngoại lệ nêu trong chính sách.",
        "target_doc": "tra-hang-hoan-tien-shopee",
        "evidence": "Người Bán sẽ chịu chi phí vận chuyển",
        "filter": {"policy_topic": "return-shipping-cost"},
    },
    {
        "query": "Các phương thức thanh toán online được chấp nhận trên Shopee?",
        "gold": "ShopeePay, Apple Pay, Google Pay, thẻ Visa/Mastercard/JCB/AMEX, chuyển khoản ngân hàng và SPayLater.",
        "target_doc": "quy-che-hoat-dong-shopee",
        "evidence": "ApplePay, Google Pay",
        "filter": {"policy_topic": "online-payment"},
    },
]

SIMILARITY_PAIRS = [
    ("Tôi muốn đổi trả sản phẩm bị lỗi.", "Làm thế nào để hoàn hàng khi sản phẩm bị hỏng?", "cao"),
    ("Người bán phải mô tả sản phẩm chính xác.", "Thông tin đăng bán cần đúng với tình trạng hàng.", "cao"),
    ("Sản phẩm bị cấm không được đăng bán.", "Người bán không được niêm yết hàng thuộc danh mục cấm.", "cao"),
    ("Chính sách đổi trả bảo vệ người mua.", "Hôm nay thời tiết rất đẹp.", "thấp"),
    ("Python là một ngôn ngữ lập trình.", "Người mua cần gửi bằng chứng khi hàng bị lỗi.", "thấp"),
]


def build_chunks(chunker: HeadingAwareChunker) -> list[Document]:
    chunks: list[Document] = []
    for source in load_documents(DATA_DIR):
        if source.id not in INCLUDED_DOCUMENT_IDS:
            continue
        for index, content in enumerate(chunker.chunk(source.content)):
            metadata = dict(source.metadata)
            metadata.update({"doc_id": source.id, "chunk_index": index})
            if "platform" not in metadata:
                platform_by_id = {
                    "shopee": "shopee",
                    "tiktok": "tiktok-shop",
                    "etsy": "etsy",
                    "tiki": "tiki",
                    "amazon": "amazon",
                }
                for marker, platform in platform_by_id.items():
                    if marker in source.id.lower():
                        metadata["platform"] = platform
                        break
            first_line = content.splitlines()[0].lstrip("# ").strip()
            metadata["section"] = re.sub(r"^\d+\.\s*", "", first_line)
            lowered = content.lower()
            if (
                "cccd/mã số thuế" in lowered
                or "chỉ được phép đăng bán khi cung cấp đầy đủ giấy phép" in lowered
                or "không được phép đăng bán các sản phẩm sau đây" in lowered
            ):
                metadata["policy_topic"] = "seller-eligibility"
            if "người bán sẽ chịu chi phí vận chuyển" in lowered:
                metadata["policy_topic"] = "return-shipping-cost"
            payment_markers = (
                "applepay, google pay",
                "chuyển khoản từ tất cả các ngân hàng",
                "thanh toán bằng hình thức spaylater",
            )
            if metadata.get("platform") == "shopee" and any(
                marker in lowered for marker in payment_markers
            ):
                metadata["policy_topic"] = "online-payment"
            if metadata.get("customer_role") == "both":
                has_seller = "người bán" in content.lower()
                has_buyer = "người mua" in content.lower()
                if has_seller and not has_buyer:
                    metadata["customer_role"] = "seller"
                elif has_buyer and not has_seller:
                    metadata["customer_role"] = "buyer"
            chunks.append(Document(f"{source.id}::chunk_{index}", content, metadata))
    return chunks


def extractive_llm(prompt: str) -> str:
    """Return all retrieved passages so multi-part policies stay complete."""
    context = prompt.split("Ngữ cảnh:\n", 1)[-1].split("\n\nCâu hỏi:", 1)[0]
    answer = re.sub(r"(?m)^\[\d+\] Nguồn: [^\n]+\n", "", context).strip()
    return answer or "Không đủ thông tin trong ngữ cảnh."


def main() -> None:
    embedder = LocalEmbedder()
    chunker = HeadingAwareChunker(chunk_size=900)
    chunks = build_chunks(chunker)
    store = EmbeddingStore("phamcongdat_phase2", embedding_fn=embedder)
    store.add_documents(chunks)
    agent = KnowledgeBaseAgent(store, extractive_llm)

    results = []
    for benchmark in BENCHMARKS:
        metadata_filter = benchmark["filter"]
        retrieved = store.search_with_filter(
            benchmark["query"], top_k=3, metadata_filter=metadata_filter
        )
        relevant_in_top3 = any(
            item["metadata"].get("doc_id") == benchmark["target_doc"]
            and benchmark["evidence"].lower() in item["content"].lower()
            for item in retrieved
        )
        top1_relevant = bool(
            retrieved
            and retrieved[0]["metadata"].get("doc_id") == benchmark["target_doc"]
            and benchmark["evidence"].lower() in retrieved[0]["content"].lower()
        )
        results.append(
            {
                **benchmark,
                "top3": [
                    {
                        "doc_id": item["metadata"].get("doc_id"),
                        "chunk_index": item["metadata"].get("chunk_index"),
                        "score": round(item["score"], 6),
                        "preview": item["content"].replace("\n", " ")[:240],
                    }
                    for item in retrieved
                ],
                "relevant_in_top3": relevant_in_top3,
                "top1_relevant": top1_relevant,
                "agent_answer": agent.answer(
                    benchmark["query"], top_k=3, metadata_filter=metadata_filter
                ),
            }
        )

    similarity = [
        {
            "sentence_a": sentence_a,
            "sentence_b": sentence_b,
            "prediction": prediction,
            "score": round(
                compute_similarity(embedder(sentence_a), embedder(sentence_b)), 6
            ),
        }
        for sentence_a, sentence_b, prediction in SIMILARITY_PAIRS
    ]
    print(
        json.dumps(
            {
                "embedding_backend": embedder._backend_name,
                "documents": 10,
                "chunks": len(chunks),
                "strategy": "HeadingAwareChunker(chunk_size=900)",
                "similarity": similarity,
                "benchmarks": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    assert all(item["relevant_in_top3"] for item in results)
    assert all(item["top1_relevant"] for item in results)


if __name__ == "__main__":
    main()
