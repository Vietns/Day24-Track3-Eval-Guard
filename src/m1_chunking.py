from __future__ import annotations

"""Module 1: advanced chunking strategies."""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, HIERARCHICAL_CHILD_SIZE, HIERARCHICAL_PARENT_SIZE, SEMANTIC_THRESHOLD


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    reader = PdfReader(path)
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    docs: list[dict] = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            print(f"  Skip {os.path.basename(fp)}: PDF has no text layer or pypdf is unavailable.")
    return docs


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


def _sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta | tb), 1)


def chunk_semantic(
    text: str,
    threshold: float = SEMANTIC_THRESHOLD,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Group adjacent sentences that appear to discuss the same topic.

    The production path can be swapped to embeddings, but this lexical fallback
    is deterministic, fast, and works without model downloads.
    """
    metadata = metadata or {}
    sentences = _sentence_split(text)
    if not sentences:
        return []

    groups: list[list[str]] = [[sentences[0]]]
    for sentence in sentences[1:]:
        sim = _jaccard(groups[-1][-1], sentence)
        if sim < threshold and len(" ".join(groups[-1])) >= 120:
            groups.append([sentence])
        else:
            groups[-1].append(sentence)

    return [
        Chunk(text="\n".join(group), metadata={**metadata, "strategy": "semantic", "chunk_index": i})
        for i, group in enumerate(groups)
        if group
    ]


def chunk_hierarchical(
    text: str,
    parent_size: int = HIERARCHICAL_PARENT_SIZE,
    child_size: int = HIERARCHICAL_CHILD_SIZE,
    metadata: dict | None = None,
) -> tuple[list[Chunk], list[Chunk]]:
    """Create parent chunks for context and child chunks for precise retrieval."""
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()] or ([text.strip()] if text.strip() else [])

    parents: list[Chunk] = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > parent_size:
            pid = f"parent_{len(parents)}"
            parents.append(Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}))
            current = ""
        if len(para) > parent_size:
            for start in range(0, len(para), parent_size):
                piece = para[start:start + parent_size].strip()
                if piece:
                    pid = f"parent_{len(parents)}"
                    parents.append(Chunk(piece, {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}))
        else:
            current = f"{current}\n\n{para}".strip() if current else para
    if current:
        pid = f"parent_{len(parents)}"
        parents.append(Chunk(current.strip(), {**metadata, "chunk_type": "parent", "parent_id": pid, "chunk_index": len(parents)}))

    children: list[Chunk] = []
    for parent in parents:
        pid = parent.metadata["parent_id"]
        current_child = ""
        for sentence in _sentence_split(parent.text) or [parent.text]:
            if current_child and len(current_child) + len(sentence) + 1 > child_size:
                children.append(Chunk(current_child.strip(), {**metadata, "chunk_type": "child", "parent_id": pid, "child_index": len(children)}, parent_id=pid))
                current_child = ""
            if len(sentence) > child_size:
                for start in range(0, len(sentence), child_size):
                    piece = sentence[start:start + child_size].strip()
                    if piece:
                        children.append(Chunk(piece, {**metadata, "chunk_type": "child", "parent_id": pid, "child_index": len(children)}, parent_id=pid))
            else:
                current_child = f"{current_child} {sentence}".strip() if current_child else sentence
        if current_child:
            children.append(Chunk(current_child.strip(), {**metadata, "chunk_type": "child", "parent_id": pid, "child_index": len(children)}, parent_id=pid))

    return parents, children


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """Chunk Markdown by logical sections while preserving section headers."""
    metadata = metadata or {}
    chunks: list[Chunk] = []
    current_header = "Document"
    current_lines: list[str] = []

    def flush() -> None:
        if not current_lines:
            return
        body = "\n".join(current_lines).strip()
        if body:
            chunks.append(Chunk(
                text=body,
                metadata={**metadata, "section": current_header.lstrip("# ").strip(), "strategy": "structure", "chunk_index": len(chunks)},
            ))

    for line in text.splitlines():
        if re.match(r"^#{1,3}\s+.+", line):
            flush()
            current_header = line.strip()
            current_lines = [line.strip()]
        else:
            current_lines.append(line)
    flush()

    if not chunks and text.strip():
        chunks.append(Chunk(text.strip(), {**metadata, "section": "Document", "strategy": "structure", "chunk_index": 0}))
    return chunks


def compare_strategies(documents: list[dict]) -> dict:
    def stats(chunk_list: list[Chunk]) -> dict:
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {"count": len(lengths), "avg_len": round(sum(lengths) / len(lengths)), "min_len": min(lengths), "max_len": max(lengths)}

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}
    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)
    results = {
        "basic": stats(basic),
        "semantic": stats(semantic),
        "hierarchical": {**stats(children), "parents": len(parents)},
        "structure": stats(structure),
    }
    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, item in results.items():
        print(f"{name:<15} {item['count']:>7} {item['avg_len']:>5} {item['min_len']:>5} {item['max_len']:>5}")
    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    compare_strategies(docs)
