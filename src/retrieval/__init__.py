"""
Module 2: Causal Retrieval Engine

This module implements retrieval based on causal relevance rather than
just semantic similarity. It traverses causal paths in the knowledge
graph and scores them by causal strength.
"""

from .causal_retriever import CausalRetrievalEngine
from .vector_store import VectorStore
from .path_scorer import CausalPathScorer
from .query_processor import QueryProcessor

__all__ = [
    "CausalRetrievalEngine",
    "VectorStore",
    "CausalPathScorer",
    "QueryProcessor",
]
