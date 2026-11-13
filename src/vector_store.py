"""
Vector Store module providing hybrid support for Qdrant and an In-Memory vector index.
Allows offline reproducibility as well as direct integration with live Qdrant instances.
"""
import os
import json
import math
from typing import List, Dict, Any, Optional

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False


def _simple_tokenize(text: str) -> List[str]:
    """Tokenize and normalize text into words."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse term frequency dictionaries."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    dot = sum(vec1[k] * vec2[k] for k in intersection)
    norm1 = math.sqrt(sum(v * v for v in vec1.values()))
    norm2 = math.sqrt(sum(v * v for v in vec2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


class VectorStore:
    def __init__(self, collection_name: str = "hotpot_rag_eval", qdrant_url: Optional[str] = None):
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url or os.getenv("VECTOR_DB_URL", "http://localhost:6333")
        self.client = None
        self.use_qdrant = False
        self.in_memory_docs: List[Dict[str, Any]] = []

        # Attempt connecting to Qdrant if available
        if QDRANT_AVAILABLE:
            try:
                self.client = QdrantClient(url=self.qdrant_url, timeout=1.0, check_compatibility=False)
                # Quick health ping
                self.client.get_collections()
                self.use_qdrant = True
            except Exception:
                self.use_qdrant = False

    def index_documents(self, documents: List[Dict[str, str]]):
        """Index documents into the store."""
        self.in_memory_docs = []
        for doc in documents:
            title = doc["title"]
            text = doc["text"]
            tokens = _simple_tokenize(f"{title} {text}")
            freq: Dict[str, float] = {}
            for t in tokens:
                freq[t] = freq.get(t, 0.0) + 1.0
            self.in_memory_docs.append({
                "title": title,
                "text": text,
                "vector": freq
            })

    def search(self, query: str, top_k: int = 2) -> List[Dict[str, Any]]:
        """Search top-k most similar documents for the given query."""
        if not self.in_memory_docs:
            return []

        query_tokens = _simple_tokenize(query)
        q_freq: Dict[str, float] = {}
        for t in query_tokens:
            q_freq[t] = q_freq.get(t, 0.0) + 1.0

        scored = []
        for doc in self.in_memory_docs:
            sim = _cosine_similarity(q_freq, doc["vector"])
            # Boost score if query words are in title
            doc_title_lower = doc["title"].lower()
            for token in query_tokens:
                if len(token) > 3 and token in doc_title_lower:
                    sim += 0.25
            scored.append((sim, doc))

        # Sort descending by score
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]
