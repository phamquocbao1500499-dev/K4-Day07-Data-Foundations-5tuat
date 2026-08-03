from __future__ import annotations

import os
import sys
import importlib
from pathlib import Path
from dotenv import load_dotenv

from ingest import parse_front_matter

# Dynamic import để hỗ trợ tên package bắt đầu bằng số
pkg = importlib.import_module("src.2A202601502_PhamQuocBao")

Document = pkg.Document
EmbeddingStore = pkg.EmbeddingStore
KnowledgeBaseAgent = pkg.KnowledgeBaseAgent
RecursiveChunker = pkg.RecursiveChunker
SentenceChunker = pkg.SentenceChunker
FixedSizeChunker = pkg.FixedSizeChunker
LocalEmbedder = pkg.LocalEmbedder
OpenAIEmbedder = pkg.OpenAIEmbedder
_mock_embed = pkg._mock_embed
EMBEDDING_PROVIDER_ENV = pkg.EMBEDDING_PROVIDER_ENV
LOCAL_EMBEDDING_MODEL = pkg.LOCAL_EMBEDDING_MODEL
OPENAI_EMBEDDING_MODEL = pkg.OPENAI_EMBEDDING_MODEL

DEFAULT_DATA_DIR = "data/k4_ecommerce"


def _select_embedder():
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            print("Local embedder không sẵn sàng; tạm dùng mock.")
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            print("OpenAI embedder không sẵn sàng; tạm dùng mock.")
            return _mock_embed
    return _mock_embed


def load_documents_custom(data_dir: str | Path) -> list[Document]:
    data_path = Path(data_dir)
    documents: list[Document] = []
    text_extensions = {".md", ".txt"}
    for path in sorted(data_path.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = str(metadata.get("doc_id") or path.stem)
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("source", str(path))
        documents.append(Document(id=doc_id, content=body, metadata=metadata))
    return documents


def build_knowledge_base_custom(data_dir: str | Path, embedding_fn, chunker=None) -> EmbeddingStore:
    chunker = chunker or RecursiveChunker(chunk_size=300)
    chunk_docs: list[Document] = []
    for doc in load_documents_custom(data_dir):
        for index, piece in enumerate(chunker.chunk(doc.content)):
            chunk_meta = dict(doc.metadata)
            chunk_meta["doc_id"] = doc.id
            chunk_meta["chunk_index"] = index
            chunk_docs.append(
                Document(id=f"{doc.id}::chunk_{index}", content=piece, metadata=chunk_meta)
            )

    store = EmbeddingStore(collection_name="my_kb", embedding_fn=embedding_fn)
    store.add_documents(chunk_docs)
    return store


def demo_llm(prompt: str) -> str:
    preview = prompt[:300].replace("\n", " ")
    return f"[DEMO LLM] Trả lời từ Context:\n{preview}..."


def main():
    query = " ".join(sys.argv[1:]).strip() or "Chính sách đổi trả hàng như thế nào?"
    data_dir = os.getenv("LAB_DATA_DIR", DEFAULT_DATA_DIR)

    print("==================================================")
    print("🚀 Demo Pipeline với Code trong src/2A202601502_PhamQuocBao")
    print(f"📁 Thư mục dữ liệu: {data_dir}")
    print("==================================================")

    embedder = _select_embedder()
    print(f"🔹 Embedding Provider: {getattr(embedder, '_backend_name', 'mock')}")

    store = build_knowledge_base_custom(data_dir, embedding_fn=embedder)
    print(f"✅ Đã nạp thành công {store.get_collection_size()} chunk vào Vector Store")

    print(f"\n🔍 [1. Vector Search] Câu hỏi: '{query}'")
    results = store.search(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"   Top-{i} [Score: {r['score']:.4f}] [Doc: {r['metadata'].get('doc_id')}]")
        print(f"   Content: {r['content'][:120].replace(chr(10), ' ')}...\n")

    print(f"🤖 [2. RAG Agent] Trả lời câu hỏi:")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    answer = agent.answer(query, top_k=3)
    print(f"   {answer}")


if __name__ == "__main__":
    main()
