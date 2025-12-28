"""
Uncertainty Quantification for Hypotheses

Provides calibrated uncertainty estimates for drug discovery hypotheses
using multiple methods: ensemble, conformal prediction, and bootstrap.
"""

import logging
import math
from typing import Optional

import numpy as np

from ..knowledge_graph.schemas import CausalPath, EvidenceType

logger = logging.getLogger(__name__)


class UncertaintyQuantifier:
    """
    Quantify uncertainty in drug discovery hypotheses.

    Combines multiple factors:
    - Evidence strength and diversity
    - Causal path properties
    - Source reliability
    - Temporal consistency

    Provides calibrated confidence intervals using bootstrap.
    """

    # Reliability weights for different evidence types
    EVIDENCE_RELIABILITY = {
        EvidenceType.RANDOMIZED_CONTROLLED_TRIAL: 1.0,
        EvidenceType.CLINICAL_OBSERVATIONAL: 0.8,
        EvidenceType.PRECLINICAL_IN_VIVO: 0.6,
        EvidenceType.PRECLINICAL_IN_VITRO: 0.5,
        EvidenceType.EXPERT_CURATED: 0.85,
        EvidenceType.LITERATURE_MINING: 0.4,
        EvidenceType.COMPUTATIONAL: 0.3,
    }

    def __init__(
        self,
        method: str = "bootstrap",
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ):
        """
        Initialize the uncertainty quantifier.

        Args:
            method: UQ method (bootstrap, ensemble, conformal)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
        """
        self.method = method
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level

    def adjust_confidence(
        self,
        base_confidence: float,
        retrieval_result,
    ) -> float:
        """
        Adjust base confidence based on retrieval evidence.

        Factors:
        - Number of supporting paths (more = higher confidence)
        - Evidence quality (RCT > observational > computational)
        - Source diversity (diverse sources = more robust)
        - Temporal consistency of evidence

        Args:
            base_confidence: Initial confidence from LLM
            retrieval_result: Result from causal retrieval

        Returns:
            Adjusted confidence score
        """
        paths = retrieval_result.causal_paths
        docs = retrieval_result.supporting_documents

        if not paths:
            return base_confidence * 0.5  # Penalize lack of evidence

        # Factor 1: Evidence quantity (log scale to diminish returns)
        quantity_factor = min(math.log(1 + len(paths)) / math.log(20), 1.0)

        # Factor 2: Evidence quality
        quality_scores = []
        for path in paths:
            for edge in path.edges:
                quality = self.EVIDENCE_RELIABILITY.get(edge.evidence_type, 0.5)
                quality_scores.append(quality)

        quality_factor = np.mean(quality_scores) if quality_scores else 0.5

        # Factor 3: Source diversity
        all_sources = set()
        for path in paths:
            for edge in path.edges:
                all_sources.update(edge.evidence_sources)

        diversity_factor = min(len(all_sources) / 10, 1.0)

        # Factor 4: Temporal consistency
        consistent_paths = sum(1 for p in paths if p.is_temporally_consistent())
        temporal_factor = consistent_paths / len(paths) if paths else 0.5

        # Combine factors
        adjustment = (
            0.25 * quantity_factor +
            0.35 * quality_factor +
            0.20 * diversity_factor +
            0.20 * temporal_factor
        )

        # Adjust base confidence
        adjusted = base_confidence * (0.5 + 0.5 * adjustment)

        return min(max(adjusted, 0.0), 1.0)

    def calculate_confidence_interval(
        self,
        point_estimate: float,
        paths: list[CausalPath],
        method: str = None,
    ) -> tuple[float, float]:
        """
        Calculate confidence interval for the hypothesis.

        Uses bootstrap resampling of supporting evidence to
        estimate uncertainty.

        Args:
            point_estimate: The point estimate of confidence
            paths: Supporting causal paths
            method: Override default method

        Returns:
            (lower_bound, upper_bound) tuple
        """
        method = method or self.method

        if method == "bootstrap":
            return self._bootstrap_ci(point_estimate, paths)
        elif method == "analytical":
            return self._analytical_ci(point_estimate, len(paths))
        else:
            # Default: simple heuristic based on evidence
            return self._heuristic_ci(point_estimate, len(paths))

    def _bootstrap_ci(
        self,
        point_estimate: float,
        paths: list[CausalPath],
    ) -> tuple[float, float]:
        """Calculate CI using bootstrap resampling."""
        if not paths:
            # Wide interval when no evidence
            return (max(0, point_estimate - 0.3), min(1, point_estimate + 0.3))

        # Collect all edge confidences
        confidences = []
        for path in paths:
            path_conf = path.get_path_confidence()
            confidences.append(path_conf)

        if not confidences:
            return (max(0, point_estimate - 0.2), min(1, point_estimate + 0.2))

        confidences = np.array(confidences)

        # Bootstrap
        bootstrap_estimates = []
        for _ in range(self.n_bootstrap):
            sample = np.random.choice(confidences, size=len(confidences), replace=True)
            bootstrap_estimates.append(np.mean(sample))

        bootstrap_estimates = np.array(bootstrap_estimates)

        # Calculate percentiles
        alpha = 1 - self.confidence_level
        lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
        upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))

        # Scale to match point estimate
        estimate_mean = np.mean(bootstrap_estimates)
        if estimate_mean > 0:
            scale = point_estimate / estimate_mean
            lower = lower * scale
            upper = upper * scale

        return (max(0, lower), min(1, upper))

    def _analytical_ci(
        self,
        point_estimate: float,
        n_samples: int,
    ) -> tuple[float, float]:
        """Calculate CI using normal approximation."""
        if n_samples == 0:
            return (max(0, point_estimate - 0.3), min(1, point_estimate + 0.3))

        # Standard error using Bernoulli approximation
        se = math.sqrt(point_estimate * (1 - point_estimate) / n_samples)

        # Z-score for confidence level
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        lower = point_estimate - z * se
        upper = point_estimate + z * se

        return (max(0, lower), min(1, upper))

    def _heuristic_ci(
        self,
        point_estimate: float,
        n_samples: int,
    ) -> tuple[float, float]:
        """Calculate CI using simple heuristic."""
        # Width decreases with more evidence
        base_width = 0.3
        width = base_width / math.sqrt(max(1, n_samples))

        lower = point_estimate - width
        upper = point_estimate + width

        return (max(0, lower), min(1, upper))

    def get_uncertainty_decomposition(
        self,
        paths: list[CausalPath],
    ) -> dict:
        """
        Decompose uncertainty into sources.

        Useful for interpretability - understanding why
        uncertainty is high or low.

        Returns dict with:
        - aleatoric: irreducible uncertainty from data
        - epistemic: reducible uncertainty from limited knowledge
        - sources: breakdown by factor
        """
        if not paths:
            return {
                "aleatoric": 0.5,
                "epistemic": 0.5,
                "sources": {
                    "evidence_quantity": 1.0,
                    "evidence_quality": 0.5,
                    "source_diversity": 0.5,
                    "temporal_consistency": 0.5,
                },
            }

        # Evidence quantity uncertainty (epistemic - reducible with more data)
        quantity_uncertainty = 1.0 / math.sqrt(max(1, len(paths)))

        # Evidence quality uncertainty (mix of aleatoric and epistemic)
        quality_scores = []
        for path in paths:
            for edge in path.edges:
                quality = self.EVIDENCE_RELIABILITY.get(edge.evidence_type, 0.5)
                quality_scores.append(quality)

        quality_mean = np.mean(quality_scores) if quality_scores else 0.5
        quality_var = np.var(quality_scores) if quality_scores else 0.25
        quality_uncertainty = 1.0 - quality_mean

        # Source diversity uncertainty (epistemic)
        all_sources = set()
        for path in paths:
            for edge in path.edges:
                all_sources.update(edge.evidence_sources)
        diversity_uncertainty = 1.0 - min(len(all_sources) / 10, 1.0)

        # Temporal consistency uncertainty (aleatoric)
        consistent = sum(1 for p in paths if p.is_temporally_consistent())
        temporal_uncertainty = 1.0 - (consistent / len(paths))

        # Decompose into aleatoric (inherent) and epistemic (knowledge)
        aleatoric = (quality_var + temporal_uncertainty) / 2
        epistemic = (quantity_uncertainty + diversity_uncertainty) / 2

        return {
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "total": aleatoric + epistemic,
            "sources": {
                "evidence_quantity": quantity_uncertainty,
                "evidence_quality": quality_uncertainty,
                "source_diversity": diversity_uncertainty,
                "temporal_consistency": temporal_uncertainty,
            },
        }

    def calibrate(
        self,
        predictions: list[float],
        actuals: list[bool],
    ) -> callable:
        """
        Calibrate confidence scores using validation data.

        Uses temperature scaling to calibrate predicted probabilities.

        Args:
            predictions: Predicted confidence scores
            actuals: Actual outcomes (True = hypothesis correct)

        Returns:
            Calibration function to apply to new predictions
        """
        if not predictions or not actuals:
            return lambda x: x

        predictions = np.array(predictions)
        actuals = np.array(actuals, dtype=float)

        # Find optimal temperature using grid search
        best_temp = 1.0
        best_ece = float("inf")

        for temp in np.linspace(0.1, 5.0, 50):
            calibrated = self._apply_temperature(predictions, temp)
            ece = self._calculate_ece(calibrated, actuals)
            if ece < best_ece:
                best_ece = ece
                best_temp = temp

        logger.info(f"Optimal temperature: {best_temp:.2f}, ECE: {best_ece:.4f}")

        return lambda x: self._apply_temperature(x, best_temp)

    def _apply_temperature(
        self,
        predictions: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        # Convert to logits, scale, convert back
        eps = 1e-7
        predictions = np.clip(predictions, eps, 1 - eps)
        logits = np.log(predictions / (1 - predictions))
        scaled_logits = logits / temperature
        return 1 / (1 + np.exp(-scaled_logits))

    def _calculate_ece(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (predictions >= bin_boundaries[i]) & (predictions < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = actuals[mask].mean()
                bin_confidence = predictions[mask].mean()
                bin_weight = mask.sum() / len(predictions)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece
