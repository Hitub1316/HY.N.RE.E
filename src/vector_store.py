"""
vector_store.py — VectorStore interface and Qdrant implementation for HY.N.RE.E.

Abstracts vector store operations so swapping to pgvector or another engine
in the future requires zero changes to indexer.py or retriever.py.
"""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
import sys

# Ensure the src directory is in the import path for relative imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import QDRANT_DIR

# Global singleton for Qdrant client to avoid multiple local instances accessing the same storage folder
_global_qdrant_client = None

class VectorStore(ABC):
    """Abstract interface for vector database operations."""

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int = 384, distance: str = "Cosine") -> None:
        pass

    @abstractmethod
    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        """
        Upsert a list of point dicts into the collection.
        Point format: {'id': int|str, 'vector': list[float], 'payload': dict}
        """
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        """
        Search for top-N nearest vectors.
        Returns list of dicts: [{'id': ..., 'score': float, 'payload': dict}]
        """
        pass


class QdrantVectorStore(VectorStore):
    """Qdrant local storage implementation of VectorStore interface."""

    def __init__(self, path: Path | str = QDRANT_DIR):
        from qdrant_client import QdrantClient
        global _global_qdrant_client
        if _global_qdrant_client is None:
            _global_qdrant_client = QdrantClient(path=str(path))
        self.client = _global_qdrant_client

    def collection_exists(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name)

    def delete_collection(self, collection_name: str) -> None:
        self.client.delete_collection(collection_name)

    def create_collection(self, collection_name: str, vector_size: int = 384, distance: str = "Cosine") -> None:
        from qdrant_client.models import Distance, VectorParams
        dist_enum = Distance.COSINE if distance.upper() == "COSINE" else Distance.EUCLID
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=dist_enum),
        )

    def upsert(self, collection_name: str, points: list[dict[str, Any]]) -> None:
        from qdrant_client.models import PointStruct
        q_points = [
            PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"]
            )
            for p in points
        ]
        self.client.upsert(collection_name=collection_name, points=q_points)

    def search(self, collection_name: str, query_vector: list[float], limit: int = 5) -> list[dict[str, Any]]:
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
            return [
                {
                    "id": hit.id,
                    "score": float(hit.score),
                    "payload": hit.payload
                }
                for hit in hits
            ]

        res = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_payload=True,
        )
        return [
            {
                "id": point.id,
                "score": float(point.score),
                "payload": point.payload
            }
            for point in res.points
        ]
