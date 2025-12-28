#!/usr/bin/env python3
"""
Evaluation Script for TC-DrugRAG

Calculates all metrics for paper: precision, recall, calibration, etc.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate TC-DrugRAG hypotheses against ground truth."""

    def __init__(self, ground_truth_path: str = None):
        """Initialize evaluator with ground truth data."""
        self.ground_truth = self._load_ground_truth(ground_truth_path)

    def _load_ground_truth(self, path: str) -> dict:
        """Load ground truth drug-disease associations."""
        if path and Path(path).exists():
            with open(path) as f:
                return json.load(f)

        # Default ground truth from TTD/DrugCentral
        return {
            # Format: (drug, disease) -> True/False
            ("metformin", "type 2 diabetes"): True,
            ("metformin", "glioblastoma"): True,  # Emerging evidence
            ("aspirin", "colorectal cancer"): True,
            ("sildenafil", "pulmonary hypertension"): True,
            ("thalidomide", "multiple myeloma"): True,
            ("rapamycin", "lymphangioleiomyomatosis"): True,
        }

    def precision_at_k(
        self,
        hypotheses: list,
        k: int = 10,
    ) -> float:
        """Calculate precision at k."""
        top_k = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:k]

        if not top_k:
            return 0.0

        correct = 0
        for h in top_k:
            key = (h.drug.lower(), h.disease.lower())
            if self.ground_truth.get(key, False):
                correct += 1

        return correct / len(top_k)

    def recall_at_k(
        self,
        hypotheses: list,
        k: int = 10,
    ) -> float:
        """Calculate recall at k."""
        top_k = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:k]

        # Get all true positives in ground truth
        true_positives = sum(1 for v in self.ground_truth.values() if v)

        if true_positives == 0:
            return 0.0

        found = 0
        for h in top_k:
            key = (h.drug.lower(), h.disease.lower())
            if self.ground_truth.get(key, False):
                found += 1

        return found / true_positives

    def ndcg_at_k(
        self,
        hypotheses: list,
        k: int = 10,
    ) -> float:
        """Calculate Normalized Discounted Cumulative Gain at k."""
        top_k = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)[:k]

        if not top_k:
            return 0.0

        # DCG
        dcg = 0.0
        for i, h in enumerate(top_k):
            key = (h.drug.lower(), h.disease.lower())
            rel = 1.0 if self.ground_truth.get(key, False) else 0.0
            dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0

        # Ideal DCG
        ideal_relevances = sorted(
            [1.0 if v else 0.0 for v in self.ground_truth.values()],
            reverse=True,
        )[:k]
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    def novelty_score(
        self,
        hypotheses: list,
        known_pairs: set = None,
    ) -> float:
        """Calculate novelty as fraction of hypotheses not in known pairs."""
        if not hypotheses:
            return 0.0

        known_pairs = known_pairs or set(self.ground_truth.keys())

        novel = 0
        for h in hypotheses:
            key = (h.drug.lower(), h.disease.lower())
            if key not in known_pairs:
                novel += 1

        return novel / len(hypotheses)

    def calibration_error(
        self,
        hypotheses: list,
        n_bins: int = 10,
    ) -> float:
        """Calculate Expected Calibration Error."""
        if not hypotheses:
            return 0.0

        confidences = np.array([h.confidence for h in hypotheses])
        actuals = np.array([
            1.0 if self.ground_truth.get(
                (h.drug.lower(), h.disease.lower()), False
            ) else 0.0
            for h in hypotheses
        ])

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
            if mask.sum() > 0:
                bin_accuracy = actuals[mask].mean()
                bin_confidence = confidences[mask].mean()
                bin_weight = mask.sum() / len(confidences)
                ece += bin_weight * abs(bin_accuracy - bin_confidence)

        return ece

    def coverage_95ci(
        self,
        hypotheses: list,
    ) -> float:
        """Calculate coverage of 95% confidence intervals."""
        if not hypotheses:
            return 0.0

        covered = 0
        total = 0

        for h in hypotheses:
            key = (h.drug.lower(), h.disease.lower())
            actual = 1.0 if self.ground_truth.get(key, False) else 0.0

            ci_low, ci_high = h.confidence_interval
            if ci_low <= actual <= ci_high:
                covered += 1
            total += 1

        return covered / total if total > 0 else 0.0

    def brier_score(
        self,
        hypotheses: list,
    ) -> float:
        """Calculate Brier score (lower is better)."""
        if not hypotheses:
            return 1.0

        scores = []
        for h in hypotheses:
            key = (h.drug.lower(), h.disease.lower())
            actual = 1.0 if self.ground_truth.get(key, False) else 0.0
            scores.append((h.confidence - actual) ** 2)

        return float(np.mean(scores))

    def evaluate_all(
        self,
        hypotheses: list,
        k_values: list = None,
    ) -> dict:
        """Calculate all metrics."""
        k_values = k_values or [5, 10, 20]

        metrics = {
            "num_hypotheses": len(hypotheses),
        }

        # Ranking metrics at different k
        for k in k_values:
            metrics[f"precision@{k}"] = self.precision_at_k(hypotheses, k)
            metrics[f"recall@{k}"] = self.recall_at_k(hypotheses, k)
            metrics[f"ndcg@{k}"] = self.ndcg_at_k(hypotheses, k)

        # Novelty
        metrics["novelty_score"] = self.novelty_score(hypotheses)

        # Calibration
        metrics["expected_calibration_error"] = self.calibration_error(hypotheses)
        metrics["coverage_95ci"] = self.coverage_95ci(hypotheses)
        metrics["brier_score"] = self.brier_score(hypotheses)

        # Confidence statistics
        if hypotheses:
            confs = [h.confidence for h in hypotheses]
            metrics["mean_confidence"] = float(np.mean(confs))
            metrics["std_confidence"] = float(np.std(confs))

        return metrics


def main():
    """Run evaluation on saved results."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--ground-truth", type=str)
    parser.add_argument("--output", type=str)

    args = parser.parse_args()

    # Load results
    with open(args.results) as f:
        results = json.load(f)

    # Initialize evaluator
    evaluator = Evaluator(args.ground_truth)

    # Convert results to hypothesis objects (simplified)
    # In practice, deserialize properly
    hypotheses = results.get("hypotheses", [])

    # Calculate metrics
    metrics = evaluator.evaluate_all(hypotheses)

    # Print metrics
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")
    print("=" * 60)

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
