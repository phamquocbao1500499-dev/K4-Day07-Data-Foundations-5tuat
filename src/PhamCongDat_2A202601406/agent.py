from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(
        self,
        question: str,
        top_k: int = 3,
        metadata_filter: dict | None = None,
    ) -> str:
        results = (
            self.store.search_with_filter(question, top_k=top_k, metadata_filter=metadata_filter)
            if metadata_filter
            else self.store.search(question, top_k=top_k)
        )
        context = "\n\n".join(
            f"[{index}] Nguồn: {result['metadata'].get('doc_id', result['id'])}\n"
            f"{result['content']}"
            for index, result in enumerate(results, start=1)
        )
        if not context:
            context = "Không tìm thấy ngữ cảnh liên quan trong kho tri thức."
        prompt = (
            "Bạn là trợ lý hỏi đáp dựa trên kho tri thức. Chỉ sử dụng ngữ cảnh "
            "được cung cấp; nếu ngữ cảnh không đủ, hãy nói rõ rằng bạn không biết.\n\n"
            f"Ngữ cảnh:\n{context}\n\n"
            f"Câu hỏi: {question}\n"
            "Câu trả lời:"
        )
        return self.llm_fn(prompt)
