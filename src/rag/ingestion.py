"""Ingest product specification PDFs into a Qdrant vector store via LlamaIndex.

When LlamaIndex / Qdrant aren't installed, ingestion falls back to a plain
JSON corpus so retrieval still operates with simple keyword scoring.
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _read_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("PDF read failed for %s (%s)", path, exc)
            return ""
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    return ""


class SpecIngestion:
    """Ingests a folder of product spec files into Qdrant (or JSON fallback)."""

    def __init__(
        self,
        specs_dir: str = "data/sample_specs",
        collection: str = "product_specs",
        qdrant_url: Optional[str] = None,
        fallback_path: str = "data/spec_corpus.json",
    ):
        self.specs_dir = Path(specs_dir)
        self.collection = collection
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.fallback_path = Path(fallback_path)

    def _list_docs(self) -> List[Path]:
        if not self.specs_dir.exists():
            return []
        return [
            p for p in self.specs_dir.iterdir()
            if p.suffix.lower() in {".pdf", ".txt", ".md"}
        ]

    def ingest(self) -> int:
        docs = self._list_docs()
        if not docs:
            logger.warning("No spec docs found in %s", self.specs_dir)
            return 0

        try:
            return self._ingest_qdrant(docs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant ingestion failed (%s) — using JSON fallback", exc)
            return self._ingest_fallback(docs)

    def _ingest_qdrant(self, docs: List[Path]) -> int:
        from llama_index.core import VectorStoreIndex, Document, Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        from llama_index.vector_stores.qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient

        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        Settings.llm = None  # retrieval-only

        client = (
            QdrantClient(url=self.qdrant_url)
            if self.qdrant_url
            else QdrantClient(":memory:")
        )
        vstore = QdrantVectorStore(client=client, collection_name=self.collection)

        documents = []
        for p in docs:
            text = _read_text(p)
            if text.strip():
                documents.append(Document(text=text, metadata={"source": p.name}))
        from llama_index.core import StorageContext
        storage_context = StorageContext.from_defaults(vector_store=vstore)
        VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        logger.info("Ingested %d docs into Qdrant collection %s", len(documents), self.collection)
        return len(documents)

    def _ingest_fallback(self, docs: List[Path]) -> int:
        corpus = []
        for p in docs:
            text = _read_text(p)
            if not text.strip():
                continue
            for chunk in self._chunk(text):
                corpus.append({"source": p.name, "text": chunk})
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)
        self.fallback_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
        logger.info("Saved fallback corpus: %d chunks → %s", len(corpus), self.fallback_path)
        return len(corpus)

    @staticmethod
    def _chunk(text: str, size: int = 400) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) < size:
                current += " " + s
            else:
                if current:
                    chunks.append(current.strip())
                current = s
        if current:
            chunks.append(current.strip())
        return chunks
