#!/usr/bin/env python3
"""
Main Experiment Runner for TC-DrugRAG

Runs the complete pipeline and evaluates against baselines.
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import yaml

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_experiment(config: dict, output_dir: Path):
    """Run the full TC-DrugRAG experiment."""
    from src.knowledge_graph import TemporalCausalKG, GraphBuilder
    from src.retrieval import CausalRetrievalEngine, VectorStore
    from src.hypothesis import HypothesisGenerator
    from src.validation import ValidationPipeline, FeedbackLoop

    results = {
        "config": config,
        "start_time": datetime.now().isoformat(),
        "metrics": {},
    }

    # Step 1: Build Knowledge Graph
    logger.info("Building Temporal Causal Knowledge Graph...")
    kg_start = time.time()

    builder = GraphBuilder(
        output_dir=config["data"]["processed_dir"],
    )

    kg = builder.build(
        sources=["drugbank", "disgenet", "string"],
    )

    results["kg_build_time"] = time.time() - kg_start
    results["kg_stats"] = kg.get_statistics()
    logger.info(f"KG built in {results['kg_build_time']:.2f}s: {results['kg_stats']}")

    # Step 2: Initialize Retrieval Engine
    logger.info("Initializing Causal Retrieval Engine...")
    vector_store = VectorStore(
        backend=config["retrieval"]["vector_store"]["backend"],
    )

    retrieval_engine = CausalRetrievalEngine(
        knowledge_graph=kg,
        vector_store=vector_store,
        config=config["retrieval"],
    )

    # Step 3: Initialize Hypothesis Generator
    logger.info("Initializing Hypothesis Generator...")
    hypothesis_generator = HypothesisGenerator(
        retrieval_engine=retrieval_engine,
        llm_provider=config["hypothesis"]["llm"]["provider"],
        model=config["hypothesis"]["llm"]["model"],
    )

    # Step 4: Initialize Validation Pipeline
    logger.info("Initializing Validation Pipeline...")
    feedback_loop = FeedbackLoop(knowledge_graph=kg)
    validation_pipeline = ValidationPipeline(
        feedback_loop=feedback_loop,
    )

    # Step 5: Run on Test Cases
    logger.info("Running on test cases...")
    test_cases = load_test_cases()

    all_hypotheses = []
    all_validations = []

    for test_case in test_cases:
        logger.info(f"Processing: {test_case['drug']} -> {test_case['disease']}")

        # Generate hypotheses
        batch = hypothesis_generator.generate(
            drug=test_case["drug"],
            disease=test_case["disease"],
            num_hypotheses=5,
        )

        all_hypotheses.extend(batch.hypotheses)

        # Validate top hypothesis
        if batch.best_hypothesis:
            validation = validation_pipeline.validate(batch.best_hypothesis)
            all_validations.append(validation)

    # Step 6: Calculate Metrics
    logger.info("Calculating metrics...")
    results["metrics"] = calculate_metrics(all_hypotheses, all_validations, test_cases)

    # Save results
    results["end_time"] = datetime.now().isoformat()
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {results_file}")

    return results


def load_test_cases() -> list:
    """Load test cases for evaluation."""
    # Example test cases for drug repurposing
    return [
        {"drug": "Metformin", "disease": "Glioblastoma", "known": True},
        {"drug": "Aspirin", "disease": "Colorectal Cancer", "known": True},
        {"drug": "Sildenafil", "disease": "Pulmonary Hypertension", "known": True},
        {"drug": "Thalidomide", "disease": "Multiple Myeloma", "known": True},
        {"drug": "Rapamycin", "disease": "Lymphangioleiomyomatosis", "known": True},
    ]


def calculate_metrics(hypotheses: list, validations: list, test_cases: list) -> dict:
    """Calculate evaluation metrics."""
    import numpy as np

    metrics = {}

    # Hypothesis quality metrics
    if hypotheses:
        confidences = [h.confidence for h in hypotheses]
        novelties = [h.novelty_score for h in hypotheses]

        metrics["mean_confidence"] = float(np.mean(confidences))
        metrics["std_confidence"] = float(np.std(confidences))
        metrics["mean_novelty"] = float(np.mean(novelties))
        metrics["num_hypotheses"] = len(hypotheses)

    # Validation metrics
    if validations:
        passed = sum(1 for v in validations if v.validation_passed)
        metrics["validation_pass_rate"] = passed / len(validations)
        metrics["mean_validation_score"] = float(
            np.mean([v.overall_score for v in validations])
        )

    # Precision at known cases
    known_cases = [tc for tc in test_cases if tc.get("known")]
    if known_cases and hypotheses:
        # Check if top hypothesis matches known relationship
        matches = 0
        for tc in known_cases:
            for h in hypotheses:
                if (h.drug.lower() == tc["drug"].lower() and
                    h.disease.lower() == tc["disease"].lower() and
                    h.confidence > 0.5):
                    matches += 1
                    break
        metrics["recall_at_known"] = matches / len(known_cases)

    return metrics


def run_baseline_comparison(config: dict, output_dir: Path):
    """Run comparison with baseline methods."""
    baselines = config["evaluation"]["baselines"]
    results = {}

    for baseline in baselines:
        logger.info(f"Running baseline: {baseline}")
        results[baseline] = run_baseline(baseline, config)

    return results


def run_baseline(baseline_name: str, config: dict) -> dict:
    """Run a single baseline method."""
    # Placeholder for baseline implementations
    return {
        "baseline": baseline_name,
        "metrics": {},
        "note": "Baseline not yet implemented",
    }


def main():
    parser = argparse.ArgumentParser(description="Run TC-DrugRAG experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/results",
        help="Output directory",
    )
    parser.add_argument(
        "--run-baselines",
        action="store_true",
        help="Also run baseline comparisons",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create output directory
    output_dir = Path(args.output) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    results = run_experiment(config, output_dir)

    # Run baselines if requested
    if args.run_baselines:
        baseline_results = run_baseline_comparison(config, output_dir)
        results["baselines"] = baseline_results

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"Hypotheses generated: {results['metrics'].get('num_hypotheses', 0)}")
    print(f"Mean confidence: {results['metrics'].get('mean_confidence', 0):.3f}")
    print(f"Validation pass rate: {results['metrics'].get('validation_pass_rate', 0):.3f}")
    print(f"Recall at known: {results['metrics'].get('recall_at_known', 0):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
