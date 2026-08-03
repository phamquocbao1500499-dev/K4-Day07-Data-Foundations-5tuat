from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.

    Rules:
        - Each chunk is at most chunk_size characters long.
        - Consecutive chunks share overlap characters.
        - The last chunk contains whatever remains.
        - If text is shorter than chunk_size, return [text].
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.

    Sentence detection: split on ". ", "! ", "? " or ".\n".
    Strip extra whitespace from each chunk.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        # Keep terminal punctuation with its sentence and accept spaces or newlines
        # after it as a boundary. Text without terminal punctuation is one sentence.
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])(?:\s+)", text.strip())
            if sentence.strip()
        ]
        return [
            " ".join(sentences[start : start + self.max_sentences_per_chunk])
            for start in range(0, len(sentences), self.max_sentences_per_chunk)
        ]


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.

    Default separator priority:
        ["\n\n", "\n", ". ", " ", ""]
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        return [part for part in self._split(text, self.separators) if part]

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if len(current_text) <= self.chunk_size:
            return [current_text]
        if not remaining_separators:
            return [
                current_text[start : start + self.chunk_size]
                for start in range(0, len(current_text), self.chunk_size)
            ]

        separator, *next_separators = remaining_separators
        if separator == "":
            return [
                current_text[start : start + self.chunk_size]
                for start in range(0, len(current_text), self.chunk_size)
            ]
        if separator not in current_text:
            return self._split(current_text, next_separators)

        # Retain delimiters so concatenating the chunks still reconstructs the input.
        tokens = re.split(f"({re.escape(separator)})", current_text)
        pieces: list[str] = []
        index = 0
        while index < len(tokens):
            piece = tokens[index]
            if index + 1 < len(tokens):
                piece += tokens[index + 1]
            if piece:
                pieces.append(piece)
            index += 2

        chunks: list[str] = []
        buffer = ""
        for piece in pieces:
            subparts = (
                [piece]
                if len(piece) <= self.chunk_size
                else self._split(piece, next_separators)
            )
            for part in subparts:
                if buffer and len(buffer) + len(part) > self.chunk_size:
                    chunks.append(buffer)
                    buffer = ""
                if len(part) > self.chunk_size:
                    chunks.extend(self._split(part, next_separators))
                elif len(buffer) + len(part) <= self.chunk_size:
                    buffer += part
                else:
                    chunks.append(part)
        if buffer:
            chunks.append(buffer)
        return chunks


class HeadingAwareChunker:
    """Split Markdown policies by heading, then split oversized sections.

    Repeating the section heading on every oversized sub-chunk preserves the
    policy topic when a chunk is retrieved without its surrounding document.
    """

    HEADING_PATTERN = re.compile(
        r"^(?:"
        r"#{1,6}\s+\S|"
        r"[IVXLCDM]+\.\s+\S|"
        r"\d+(?:\.\d+)*\.\s+[A-ZÀ-Ỹ0-9][A-ZÀ-Ỹ0-9\s/&()_-]{4,}$|"
        r"Cách\s+\d+\s*:"
        r")"
    )

    def __init__(self, chunk_size: int = 900) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        sections: list[str] = []
        current_lines: list[str] = []
        for line in text.splitlines():
            if self.HEADING_PATTERN.match(line.strip()) and current_lines:
                section = "\n".join(current_lines).strip()
                if section:
                    sections.append(section)
                current_lines = [line]
            else:
                current_lines.append(line)
        final_section = "\n".join(current_lines).strip()
        if final_section:
            sections.append(final_section)

        chunks: list[str] = []
        for section in sections:
            if len(section) <= self.chunk_size:
                chunks.append(section)
                continue

            first_line, separator, body = section.partition("\n")
            has_heading = bool(separator and self.HEADING_PATTERN.match(first_line.strip()))
            heading = first_line.strip() if has_heading else ""
            content = body.strip() if has_heading else section
            available_size = max(1, self.chunk_size - len(heading) - 2)
            repeated_lead = ""
            lead, paragraph_break, remainder = content.partition("\n\n")
            if (
                paragraph_break
                and remainder
                and len(lead) < available_size // 4
                and available_size > len(lead) + 2
                and not re.match(r"^\d+(?:\.\d+)+\.", lead.strip())
            ):
                # A short section introduction is useful context, not a useful
                # standalone retrieval result. Repeat it on each sub-chunk.
                repeated_lead = lead.strip()
                content = remainder.strip()
                available_size -= len(repeated_lead) + 2
            subchunks = RecursiveChunker(chunk_size=available_size).chunk(content)
            for subchunk in subchunks:
                body_parts = [part for part in (repeated_lead, subchunk.strip()) if part]
                chunk_body = "\n\n".join(body_parts)
                combined = f"{heading}\n\n{chunk_body}" if heading else chunk_body
                if combined:
                    chunks.append(combined)
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    cosine_similarity = dot(a, b) / (||a|| * ||b||)

    Returns 0.0 if either vector has zero magnitude.
    """
    magnitude_a = math.sqrt(_dot(vec_a, vec_a))
    magnitude_b = math.sqrt(_dot(vec_b, vec_b))
    if magnitude_a == 0.0 or magnitude_b == 0.0:
        return 0.0
    return _dot(vec_a, vec_b) / (magnitude_a * magnitude_b)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")

        strategies = {
            "fixed_size": FixedSizeChunker(chunk_size=chunk_size, overlap=0),
            "by_sentences": SentenceChunker(max_sentences_per_chunk=3),
            "recursive": RecursiveChunker(chunk_size=chunk_size),
        }
        comparison = {}
        for name, chunker in strategies.items():
            chunks = chunker.chunk(text)
            comparison[name] = {
                "count": len(chunks),
                "avg_length": (
                    sum(len(chunk) for chunk in chunks) / len(chunks) if chunks else 0.0
                ),
                "chunks": chunks,
            }
        return comparison
