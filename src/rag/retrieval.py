"""Retrieve product specification context for a given defect.

Tries Qdrant + LlamaIndex first; if the vector store is empty or unavailable
falls back to BM25-ish keyword scoring over the JSON corpus written by
``SpecIngestion``.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SpecContext:
    query: str
    snippets: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n---\n".join(self.snippets)

    def to_dict(self) -> dict:
        return {"query": self.query, "snippets": self.snippets, "sources": self.sources}


class SpecRetriever:
    def __init__(
        self,
        collection: str = "product_specs",
        qdrant_url: Optional[str] = None,
        fallback_path: str = "data/spec_corpus.json",
        top_k: int = 3,
    ):
        self.collection = collection
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL")
        self.fallback_path = Path(fallback_path)
        self.top_k = top_k
        self._engine = None
        self._corpus: Optional[List[dict]] = None
        self._init_backends()

    def _init_backends(self) -> None:
        try:
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            from llama_index.vector_stores.qdrant import QdrantVectorStore
            from qdrant_client import QdrantClient

            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            Settings.llm = None
            client = (
                QdrantClient(url=self.qdrant_url)
                if self.qdrant_url
                else QdrantClient(":memory:")
            )
            vstore = QdrantVectorStore(client=client, collection_name=self.collection)
            index = VectorStoreIndex.from_vector_store(vstore)
            self._engine = index.as_retriever(similarity_top_k=self.top_k)
            logger.info("Spec retriever initialised on Qdrant")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Qdrant retriever unavailable (%s) — JSON fallback", exc)
            self._load_corpus()

    def _load_corpus(self) -> None:
        if self.fallback_path.exists():
            self._corpus = json.loads(self.fallback_path.read_text(encoding="utf-8"))
        else:
            self._corpus = []

    def retrieve(self, defect_type: str, product: Optional[str] = None) -> SpecContext:
        query = f"{defect_type} tolerance specification" + (f" for {product}" if product else "")
        if self._engine is not None:
            try:
                nodes = self._engine.retrieve(query)
                snippets = [n.get_content() for n in nodes]
                sources = [n.metadata.get("source", "spec") for n in nodes]
                if snippets:
                    return SpecContext(query=query, snippets=snippets, sources=sources)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Vector retrieval failed (%s) — fallback", exc)
        return self._keyword_retrieve(query, defect_type)

    def _keyword_retrieve(self, query: str, defect_type: str) -> SpecContext:
        if self._corpus is None:
            self._load_corpus()
        if not self._corpus:
            return SpecContext(query=query, snippets=[], sources=[])

        terms = re.findall(r"\w+", query.lower())
        scored = []
        for entry in self._corpus:
            text_l = entry["text"].lower()
            score = sum(text_l.count(t) for t in terms)
            if defect_type.lower() in text_l:
                score += 5
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: self.top_k]
        return SpecContext(
            query=query,
            snippets=[e["text"] for _, e in top],
            sources=[e["source"] for _, e in top],
        )
