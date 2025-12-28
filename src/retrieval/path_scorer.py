"""
Causal Path Scorer

Scores causal paths based on multiple factors:
- Evidence strength
- Causal depth (path length)
- Temporal consistency
- Source diversity
"""

import logging
import math
from datetime import datetime

from ..knowledge_graph.schemas import CausalPath, EvidenceType

logger = logging.getLogger(__name__)


class CausalPathScorer:
    """
    Score causal paths for ranking in retrieval.

    The scoring function combines multiple factors with configurable weights.

    Formula:
        Score = w1*Evidence + w2*CausalDepth + w3*Temporal + w4*Diversity

    where:
        - Evidence: Aggregated confidence from edge weights
        - CausalDepth: Penalty for longer paths (uncertainty accumulates)
        - Temporal: Bonus for temporally consistent paths
        - Diversity: Bonus for evidence from diverse sources
    """

    # Evidence type weights (higher for more reliable evidence)
    EVIDENCE_TYPE_WEIGHTS = {
        EvidenceType.RANDOMIZED_CONTROLLED_TRIAL: 1.0,
        EvidenceType.CLINICAL_OBSERVATIONAL: 0.85,
        EvidenceType.PRECLINICAL_IN_VIVO: 0.7,
        EvidenceType.PRECLINICAL_IN_VITRO: 0.6,
        EvidenceType.EXPERT_CURATED: 0.9,
        EvidenceType.LITERATURE_MINING: 0.5,
        EvidenceType.COMPUTATIONAL: 0.4,
    }

    def __init__(
        self,
        evidence_weight: float = 0.35,
        causal_depth_weight: float = 0.25,
        temporal_weight: float = 0.20,
        diversity_weight: float = 0.20,
        temporal_decay_lambda: float = 0.1,
    ):
        """
        Initialize the path scorer.

        Args:
            evidence_weight: Weight for evidence strength factor
            causal_depth_weight: Weight for causal depth penalty
            temporal_weight: Weight for temporal consistency bonus
            diversity_weight: Weight for source diversity bonus
            temporal_decay_lambda: Decay rate for temporal confidence
        """
        self.evidence_weight = evidence_weight
        self.causal_depth_weight = causal_depth_weight
        self.temporal_weight = temporal_weight
        self.diversity_weight = diversity_weight
        self.temporal_decay_lambda = temporal_decay_lambda

        # Normalize weights
        total = (evidence_weight + causal_depth_weight +
                 temporal_weight + diversity_weight)
        self.evidence_weight /= total
        self.causal_depth_weight /= total
        self.temporal_weight /= total
        self.diversity_weight /= total

    def score(
        self,
        path: CausalPath,
        time_point: datetime = None,
    ) -> float:
        """
        Calculate overall score for a causal path.

        Args:
            path: The causal path to score
            time_point: Reference time point for temporal calculations

        Returns:
            Normalized score in [0, 1]
        """
        if not path.edges:
            return 0.0

        time_point = time_point or datetime.now()

        # Calculate individual factors
        evidence_score = self._calculate_evidence_score(path, time_point)
        depth_score = self._calculate_depth_score(path)
        temporal_score = self._calculate_temporal_score(path)
        diversity_score = self._calculate_diversity_score(path)

        # Combine with weights
        total_score = (
            self.evidence_weight * evidence_score +
            self.causal_depth_weight * depth_score +
            self.temporal_weight * temporal_score +
            self.diversity_weight * diversity_score
        )

        return total_score

    def _calculate_evidence_score(
        self,
        path: CausalPath,
        time_point: datetime,
    ) -> float:
        """
        Calculate evidence strength score.

        Combines edge confidences with evidence type weights and temporal decay.
        """
        if not path.edges:
            return 0.0

        # Aggregate evidence across edges
        edge_scores = []

        for edge in path.edges:
            # Base confidence
            base_conf = edge.confidence

            # Evidence type multiplier
            ev_weight = self.EVIDENCE_TYPE_WEIGHTS.get(
                edge.evidence_type, 0.5
            )

            # Temporal decay
            years_old = (time_point - edge.t_start).days / 365.25
            temporal_decay = math.exp(-self.temporal_decay_lambda * years_old)

            # Number of sources bonus
            source_bonus = min(len(edge.evidence_sources), 5) / 5.0

            edge_score = base_conf * ev_weight * temporal_decay * (0.7 + 0.3 * source_bonus)
            edge_scores.append(edge_score)

        # Product of edge scores (confidence decreases through chain)
        # But we use geometric mean to not penalize long paths too heavily
        if edge_scores:
            geometric_mean = math.exp(
                sum(math.log(max(s, 0.01)) for s in edge_scores) / len(edge_scores)
            )
            return geometric_mean

        return 0.0

    def _calculate_depth_score(self, path: CausalPath) -> float:
        """
        Calculate causal depth score.

        Shorter paths are preferred as they involve fewer assumptions.
        Uses exponential decay with path length.
        """
        if not path.edges:
            return 0.0

        # Exponential decay with path length
        # At length 1: score = 1.0
        # At length 4: score ≈ 0.22
        depth_penalty = math.exp(-0.5 * (path.length - 1))

        return depth_penalty

    def _calculate_temporal_score(self, path: CausalPath) -> float:
        """
        Calculate temporal consistency score.

        Rewards paths where causes precede effects in time.
        """
        if not path.edges:
            return 0.0

        # Check if path is temporally consistent
        if path.is_temporally_consistent():
            return 1.0

        # Partial credit for mostly consistent paths
        consistent_edges = 0
        for i in range(len(path.edges) - 1):
            if path.edges[i].t_start <= path.edges[i + 1].t_start:
                consistent_edges += 1

        if len(path.edges) > 1:
            return consistent_edges / (len(path.edges) - 1)

        return 1.0

    def _calculate_diversity_score(self, path: CausalPath) -> float:
        """
        Calculate source diversity score.

        Rewards paths supported by evidence from diverse sources.
        """
        if not path.edges:
            return 0.0

        # Collect all evidence sources
        all_sources = set()
        evidence_types = set()

        for edge in path.edges:
            all_sources.update(edge.evidence_sources)
            evidence_types.add(edge.evidence_type)

        # Diversity metrics
        source_diversity = min(len(all_sources), 10) / 10.0
        type_diversity = len(evidence_types) / len(EvidenceType)

        # Combined diversity score
        return 0.6 * source_diversity + 0.4 * type_diversity

    def get_score_breakdown(
        self,
        path: CausalPath,
        time_point: datetime = None,
    ) -> dict:
        """
        Get detailed breakdown of score components.

        Useful for interpretability and debugging.
        """
        time_point = time_point or datetime.now()

        evidence = self._calculate_evidence_score(path, time_point)
        depth = self._calculate_depth_score(path)
        temporal = self._calculate_temporal_score(path)
        diversity = self._calculate_diversity_score(path)

        total = (
            self.evidence_weight * evidence +
            self.causal_depth_weight * depth +
            self.temporal_weight * temporal +
            self.diversity_weight * diversity
        )

        return {
            "total_score": total,
            "evidence_score": evidence,
            "evidence_contribution": self.evidence_weight * evidence,
            "depth_score": depth,
            "depth_contribution": self.causal_depth_weight * depth,
            "temporal_score": temporal,
            "temporal_contribution": self.temporal_weight * temporal,
            "diversity_score": diversity,
            "diversity_contribution": self.diversity_weight * diversity,
            "path_length": path.length,
            "num_sources": sum(len(e.evidence_sources) for e in path.edges),
        }
