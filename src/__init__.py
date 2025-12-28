"""
TC-DrugRAG: Temporal Causal Retrieval-Augmented Generation for Drug Discovery

A framework that combines temporal causal reasoning with RAG for
hypothesis-driven drug discovery.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .knowledge_graph import TemporalCausalKG
from .retrieval import CausalRetrievalEngine
from .hypothesis import HypothesisGenerator
from .validation import ValidationPipeline

__all__ = [
    "TemporalCausalKG",
    "CausalRetrievalEngine",
    "HypothesisGenerator",
    "ValidationPipeline",
]
