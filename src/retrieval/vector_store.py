"""
Vector Store for Document Retrieval.

Provides semantic search capabilities for supporting documents
alongside the causal knowledge graph.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document embeddings and retrieval.

    Supports multiple backends: FAISS (default), ChromaDB, Pinecone.

    Example:
        >>> store = VectorStore(backend="faiss")
        >>> store.add_documents([
        ...     {"id": "PMID:12345", "text": "Metformin activates AMPK..."},
        ... ])
        >>> results = store.search("AMPK activation mechanism", top_k=10)
    """

    def __init__(
        self,
        backend: str = "faiss",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = None,
        dimension: int = 384,
    ):
        """
        Initialize the vector store.

        Args:
            backend: Storage backend (faiss, chromadb, pinecone)
            embedding_model: HuggingFace embedding model
            index_path: Path to save/load index
            dimension: Embedding dimension
        """
        self.backend = backend
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None

        # Initialize embedding model
        self._init_embedding_model(embedding_model)

        # Initialize index
        self._init_index()

        # Document storage
        self.documents = {}

    def _init_embedding_model(self, model_name: str):
        """Initialize the embedding model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to load embedding model: {e}")
            self.embedding_model = None

    def _init_index(self):
        """Initialize the vector index."""
        if self.backend == "faiss":
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product
                logger.info("Initialized FAISS index")
            except ImportError:
                logger.warning("FAISS not available, using simple numpy index")
                self.index = None
                self._embeddings = []

        elif self.backend == "chromadb":
            try:
                import chromadb
                self.client = chromadb.Client()
                self.collection = self.client.create_collection("tc_drugrag")
                logger.info("Initialized ChromaDB")
            except ImportError:
                logger.warning("ChromaDB not available")
                self.index = None

        else:
            logger.warning(f"Unknown backend: {self.backend}")
            self.index = None

    def add_documents(
        self,
        documents: list[dict],
        batch_size: int = 100,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of dicts with 'id' and 'text' fields
            batch_size: Batch size for embedding
        """
        if not self.embedding_model:
            logger.warning("No embedding model available")
            return

        logger.info(f"Adding {len(documents)} documents")

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            # Extract texts and IDs
            texts = [doc.get("text", "") for doc in batch]
            ids = [doc.get("id", str(i + j)) for j, doc in enumerate(batch)]

            # Compute embeddings
            embeddings = self.embedding_model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Add to index
            if self.backend == "faiss" and self.index is not None:
                import faiss
                self.index.add(embeddings.astype(np.float32))
            elif hasattr(self, "_embeddings"):
                self._embeddings.extend(embeddings)

            # Store documents
            for doc_id, doc in zip(ids, batch):
                self.documents[doc_id] = doc

        logger.info(f"Index now contains {len(self.documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict = None,
    ) -> list[dict]:
        """
        Search for similar documents.

        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of matching documents with scores
        """
        if not self.embedding_model:
            return []

        # Embed query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True,
        )[0]

        # Search index
        if self.backend == "faiss" and self.index is not None:
            import faiss
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype(np.float32),
                top_k,
            )
            results = []
            doc_ids = list(self.documents.keys())
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    doc = self.documents[doc_id].copy()
                    doc["score"] = float(score)
                    results.append(doc)
            return results

        elif hasattr(self, "_embeddings") and self._embeddings:
            # Simple numpy search
            embeddings = np.array(self._embeddings)
            scores = embeddings @ query_embedding
            top_indices = np.argsort(scores)[::-1][:top_k]

            results = []
            doc_ids = list(self.documents.keys())
            for idx in top_indices:
                if idx < len(doc_ids):
                    doc_id = doc_ids[idx]
                    doc = self.documents[doc_id].copy()
                    doc["score"] = float(scores[idx])
                    results.append(doc)
            return results

        return []

    def get_document(self, doc_id: str) -> Optional[dict]:
        """Get a document by ID."""
        return self.documents.get(doc_id)

    def save(self, path: str = None) -> None:
        """Save the index to disk."""
        path = Path(path) if path else self.index_path
        if not path:
            logger.warning("No path specified for saving")
            return

        path.mkdir(parents=True, exist_ok=True)

        if self.backend == "faiss" and self.index is not None:
            import faiss
            faiss.write_index(self.index, str(path / "index.faiss"))

        # Save documents
        import json
        with open(path / "documents.json", "w") as f:
            json.dump(self.documents, f)

        logger.info(f"Saved index to {path}")

    def load(self, path: str = None) -> None:
        """Load the index from disk."""
        path = Path(path) if path else self.index_path
        if not path or not path.exists():
            logger.warning("Index path does not exist")
            return

        if self.backend == "faiss":
            import faiss
            index_file = path / "index.faiss"
            if index_file.exists():
                self.index = faiss.read_index(str(index_file))

        # Load documents
        import json
        docs_file = path / "documents.json"
        if docs_file.exists():
            with open(docs_file) as f:
                self.documents = json.load(f)

        logger.info(f"Loaded index with {len(self.documents)} documents")

    def clear(self) -> None:
        """Clear the index."""
        self._init_index()
        self.documents = {}
