"""
Causal Retrieval Engine

The core innovation: retrieve information based on causal relevance,
not just semantic similarity. Combines knowledge graph traversal with
temporal filtering and counterfactual reasoning.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

from ..knowledge_graph import TemporalCausalKG
from ..knowledge_graph.schemas import (
    CausalPath,
    CausalRelationType,
    DrugDiscoveryQuery,
    Node,
)
from .path_scorer import CausalPathScorer
from .query_processor import QueryProcessor
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of a causal retrieval query."""
    query: DrugDiscoveryQuery
    causal_paths: list[CausalPath]
    supporting_documents: list[dict]
    aggregated_evidence: dict
    confidence_scores: dict
    uncertainty_bounds: tuple[float, float]


class CausalRetrievalEngine:
    """
    Causal Retrieval Engine for Drug Discovery.

    Unlike standard RAG that retrieves based on embedding similarity,
    this engine:
    1. Traverses causal paths in the knowledge graph
    2. Scores paths by causal strength and temporal validity
    3. Retrieves supporting documents for each causal claim
    4. Aggregates evidence with uncertainty quantification

    Example:
        >>> engine = CausalRetrievalEngine(knowledge_graph, vector_store)
        >>> query = DrugDiscoveryQuery(
        ...     query_type="repurposing",
        ...     drug="Metformin",
        ...     disease="Glioblastoma",
        ... )
        >>> results = engine.retrieve(query)
        >>> for path in results.causal_paths[:5]:
        ...     print(path.get_mechanism_description())
    """

    def __init__(
        self,
        knowledge_graph: TemporalCausalKG,
        vector_store: VectorStore = None,
        config: dict = None,
    ):
        """
        Initialize the Causal Retrieval Engine.

        Args:
            knowledge_graph: The temporal causal knowledge graph
            vector_store: Vector store for document retrieval
            config: Configuration dictionary
        """
        self.kg = knowledge_graph
        self.vector_store = vector_store or VectorStore()
        self.config = config or {}

        self.query_processor = QueryProcessor()
        self.path_scorer = CausalPathScorer(
            evidence_weight=self.config.get("evidence_weight", 0.35),
            causal_depth_weight=self.config.get("causal_depth_weight", 0.25),
            temporal_weight=self.config.get("temporal_consistency_weight", 0.20),
            diversity_weight=self.config.get("source_diversity_weight", 0.20),
        )

        # Retrieval parameters
        self.max_path_depth = self.config.get("max_path_depth", 4)
        self.min_edge_confidence = self.config.get("min_edge_confidence", 0.3)
        self.top_k_paths = self.config.get("top_k_paths", 50)
        self.top_k_documents = self.config.get("top_k_documents", 20)

    def retrieve(
        self,
        query: DrugDiscoveryQuery,
        return_documents: bool = True,
    ) -> RetrievalResult:
        """
        Retrieve causal evidence for a drug discovery query.

        Args:
            query: Structured drug discovery query
            return_documents: Whether to retrieve supporting documents

        Returns:
            RetrievalResult with causal paths and evidence
        """
        logger.info(f"Processing query: {query.query_type} - {query.drug} / {query.disease}")

        # Step 1: Process query to identify entities
        query_entities = self.query_processor.process(query, self.kg)

        # Step 2: Find causal paths
        causal_paths = self._find_causal_paths(query_entities, query.time_point)

        # Step 3: Score and rank paths
        scored_paths = self._score_paths(causal_paths, query.time_point)

        # Step 4: Retrieve supporting documents
        documents = []
        if return_documents:
            documents = self._retrieve_supporting_documents(scored_paths)

        # Step 5: Aggregate evidence
        aggregated = self._aggregate_evidence(scored_paths, documents)

        # Step 6: Calculate uncertainty bounds
        uncertainty = self._calculate_uncertainty(scored_paths)

        return RetrievalResult(
            query=query,
            causal_paths=scored_paths[:self.top_k_paths],
            supporting_documents=documents[:self.top_k_documents],
            aggregated_evidence=aggregated,
            confidence_scores=self._get_confidence_breakdown(scored_paths),
            uncertainty_bounds=uncertainty,
        )

    def _find_causal_paths(
        self,
        query_entities: dict,
        time_point: datetime,
    ) -> list[CausalPath]:
        """Find all causal paths connecting query entities."""
        all_paths = []

        source_entities = query_entities.get("sources", [])
        target_entities = query_entities.get("targets", [])

        # If we have both source and target, find paths between them
        if source_entities and target_entities:
            for source in source_entities:
                for target in target_entities:
                    paths = list(self.kg.find_causal_paths(
                        source_id=source.id,
                        target_id=target.id,
                        max_depth=self.max_path_depth,
                        time_point=time_point,
                        min_path_confidence=self.min_edge_confidence,
                    ))
                    all_paths.extend(paths)

        # If only source, explore outgoing paths
        elif source_entities:
            for source in source_entities:
                paths = list(self.kg.find_causal_paths(
                    source_id=source.id,
                    max_depth=self.max_path_depth,
                    time_point=time_point,
                    min_path_confidence=self.min_edge_confidence,
                ))
                all_paths.extend(paths)

        # If only target, this is trickier - we'd need to search backwards
        elif target_entities:
            # For now, we'll search from all nodes to target
            logger.warning("Target-only search not fully implemented")

        logger.info(f"Found {len(all_paths)} causal paths")
        return all_paths

    def _score_paths(
        self,
        paths: list[CausalPath],
        time_point: datetime,
    ) -> list[CausalPath]:
        """Score and rank causal paths."""
        scored = []

        for path in paths:
            score = self.path_scorer.score(path, time_point)
            # Attach score as metadata (bit of a hack, but keeps CausalPath clean)
            path.edges[0].metadata["path_score"] = score
            scored.append((score, path))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        return [path for _, path in scored]

    def _retrieve_supporting_documents(
        self,
        paths: list[CausalPath],
    ) -> list[dict]:
        """Retrieve documents that support the causal paths."""
        documents = []
        seen_sources = set()

        for path in paths:
            for edge in path.edges:
                for source_id in edge.evidence_sources:
                    if source_id not in seen_sources:
                        seen_sources.add(source_id)

                        # Retrieve document from vector store
                        doc = self.vector_store.get_document(source_id)
                        if doc:
                            doc["supporting_edge"] = edge.to_dict()
                            documents.append(doc)

        return documents

    def _aggregate_evidence(
        self,
        paths: list[CausalPath],
        documents: list[dict],
    ) -> dict:
        """Aggregate evidence across paths and documents."""
        aggregated = {
            "num_paths": len(paths),
            "num_documents": len(documents),
            "relation_types": {},
            "evidence_types": {},
            "time_range": {"earliest": None, "latest": None},
            "key_entities": {},
            "consensus_mechanisms": [],
        }

        for path in paths:
            for edge in path.edges:
                # Count relation types
                rel_type = edge.relation_type.value
                aggregated["relation_types"][rel_type] = (
                    aggregated["relation_types"].get(rel_type, 0) + 1
                )

                # Count evidence types
                ev_type = edge.evidence_type.value
                aggregated["evidence_types"][ev_type] = (
                    aggregated["evidence_types"].get(ev_type, 0) + 1
                )

                # Track time range
                if (aggregated["time_range"]["earliest"] is None or
                    edge.t_start < aggregated["time_range"]["earliest"]):
                    aggregated["time_range"]["earliest"] = edge.t_start

                if (aggregated["time_range"]["latest"] is None or
                    edge.t_start > aggregated["time_range"]["latest"]):
                    aggregated["time_range"]["latest"] = edge.t_start

                # Count key entities
                for entity in [edge.source, edge.target]:
                    entity_key = f"{entity.node_type.value}:{entity.name}"
                    aggregated["key_entities"][entity_key] = (
                        aggregated["key_entities"].get(entity_key, 0) + 1
                    )

        # Find consensus mechanisms (paths that appear multiple times)
        mechanism_counts = {}
        for path in paths:
            mechanism = path.get_mechanism_description()
            mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1

        aggregated["consensus_mechanisms"] = sorted(
            mechanism_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return aggregated

    def _calculate_uncertainty(
        self,
        paths: list[CausalPath],
    ) -> tuple[float, float]:
        """Calculate uncertainty bounds using bootstrap."""
        if not paths:
            return (0.0, 0.0)

        scores = []
        for path in paths:
            if path.edges and "path_score" in path.edges[0].metadata:
                scores.append(path.edges[0].metadata["path_score"])

        if not scores:
            return (0.0, 0.0)

        # Bootstrap confidence interval
        n_bootstrap = 1000
        bootstrap_means = []

        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        return (ci_low, ci_high)

    def _get_confidence_breakdown(
        self,
        paths: list[CausalPath],
    ) -> dict:
        """Get breakdown of confidence scores by factor."""
        if not paths:
            return {}

        return {
            "mean_path_confidence": np.mean([
                path.get_path_confidence() for path in paths
            ]) if paths else 0.0,
            "path_length_distribution": {
                length: sum(1 for p in paths if p.length == length)
                for length in range(1, self.max_path_depth + 1)
            },
            "temporal_consistency_rate": (
                sum(1 for p in paths if p.is_temporally_consistent()) / len(paths)
                if paths else 0.0
            ),
        }

    def retrieve_for_hypothesis(
        self,
        hypothesis: str,
        entities: list[Node] = None,
    ) -> RetrievalResult:
        """
        Retrieve causal evidence for a hypothesis statement.

        This is a convenience method that parses a natural language
        hypothesis and retrieves supporting/contradicting evidence.

        Args:
            hypothesis: Natural language hypothesis
            entities: Optional pre-extracted entities

        Returns:
            RetrievalResult with evidence
        """
        # Parse hypothesis into structured query
        query = self.query_processor.parse_hypothesis(hypothesis, entities)

        return self.retrieve(query)

    def find_contradicting_evidence(
        self,
        path: CausalPath,
        time_point: datetime = None,
    ) -> list[CausalPath]:
        """
        Find evidence that contradicts a causal path.

        For counterfactual reasoning - what evidence would suggest
        the opposite causal relationship?

        Args:
            path: The causal path to find contradictions for
            time_point: Time point for temporal filtering

        Returns:
            List of contradicting causal paths
        """
        time_point = time_point or datetime.now()
        contradictions = []

        # For each edge in the path, look for opposite relations
        opposite_relations = {
            CausalRelationType.ACTIVATES: CausalRelationType.INHIBITS,
            CausalRelationType.INHIBITS: CausalRelationType.ACTIVATES,
            CausalRelationType.UPREGULATES: CausalRelationType.DOWNREGULATES,
            CausalRelationType.DOWNREGULATES: CausalRelationType.UPREGULATES,
            CausalRelationType.CAUSES: CausalRelationType.PREVENTS,
            CausalRelationType.PREVENTS: CausalRelationType.CAUSES,
        }

        for edge in path.edges:
            opposite = opposite_relations.get(edge.relation_type)
            if opposite:
                # Search for paths with opposite relation
                for contra_path in self.kg.find_causal_paths(
                    source_id=edge.source.id,
                    target_id=edge.target.id,
                    max_depth=1,
                    time_point=time_point,
                ):
                    for contra_edge in contra_path.edges:
                        if contra_edge.relation_type == opposite:
                            contradictions.append(contra_path)

        return contradictions
