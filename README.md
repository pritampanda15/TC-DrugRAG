# TC-DrugRAG: Temporal Causal Retrieval-Augmented Generation for Drug Discovery

> A novel framework that combines temporal causal reasoning with RAG for hypothesis-driven drug discovery with uncertainty quantification.

## Overview

TC-DrugRAG addresses critical limitations in current AI-driven drug discovery:

| Current Limitation | TC-DrugRAG Solution |
|-------------------|---------------------|
| Static knowledge retrieval | Temporal causal knowledge graph with time-varying relationships |
| Correlation-based predictions | Causal path traversal and counterfactual reasoning |
| Black-box predictions | Interpretable mechanisms with evidence chains |
| Uncalibrated confidence | Uncertainty quantification with confidence intervals |
| No learning from validation | Closed-loop feedback updates knowledge graph |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TC-DrugRAG FRAMEWORK                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │
│  │  Module 1   │    │  Module 2   │    │  Module 3   │             │
│  │  Temporal   │───▶│   Causal    │───▶│ Hypothesis  │             │
│  │  Causal KG  │    │  Retrieval  │    │ Generation  │             │
│  └─────────────┘    └─────────────┘    └─────────────┘             │
│         ▲                                     │                     │
│         │                                     ▼                     │
│         │           ┌─────────────────────────────┐                │
│         └───────────│     Module 4: Validation    │                │
│                     │     & Feedback Loop         │                │
│                     └─────────────────────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tc-drugrag.git
cd tc-drugrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Copy environment file and add your API keys
cp .env.example .env
# Edit .env with your keys
```

## Quick Start

```python
from tc_drugrag import (
    TemporalCausalKG,
    CausalRetrievalEngine,
    HypothesisGenerator,
    ValidationPipeline,
)

# 1. Build Knowledge Graph
from tc_drugrag.knowledge_graph import GraphBuilder

builder = GraphBuilder()
kg = builder.build(sources=["drugbank", "disgenet"])

# 2. Initialize Retrieval Engine
from tc_drugrag.retrieval import VectorStore

vector_store = VectorStore()
retrieval_engine = CausalRetrievalEngine(kg, vector_store)

# 3. Generate Hypotheses
generator = HypothesisGenerator(retrieval_engine)
hypotheses = generator.generate(
    drug="Metformin",
    disease="Glioblastoma",
    num_hypotheses=5,
)

# 4. Validate Top Hypothesis
pipeline = ValidationPipeline()
result = pipeline.validate(hypotheses.best_hypothesis)

# Print results
for h in hypotheses.hypotheses:
    print(h.get_summary())
```

## Project Structure

```
tc_drugrag/
├── configs/
│   └── config.yaml          # Main configuration
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data
│   └── external/             # External databases
├── experiments/
│   ├── run_experiment.py     # Main experiment runner
│   └── evaluate.py           # Evaluation metrics
├── notebooks/
│   └── demo.ipynb            # Demo notebook
├── src/
│   ├── knowledge_graph/      # Module 1: Temporal Causal KG
│   │   ├── schemas.py        # Node, Edge, Path definitions
│   │   ├── temporal_kg.py    # Main KG implementation
│   │   ├── entity_extraction.py
│   │   ├── relation_extraction.py
│   │   └── graph_builder.py
│   ├── retrieval/            # Module 2: Causal Retrieval
│   │   ├── causal_retriever.py
│   │   ├── vector_store.py
│   │   ├── path_scorer.py
│   │   └── query_processor.py
│   ├── hypothesis/           # Module 3: Hypothesis Generation
│   │   ├── generator.py
│   │   ├── uncertainty.py
│   │   └── structured_output.py
│   ├── validation/           # Module 4: Validation
│   │   ├── pipeline.py
│   │   ├── docking.py
│   │   ├── admet.py
│   │   └── feedback.py
│   └── utils/
├── tests/
├── requirements.txt
└── pyproject.toml
```

## Key Features

### 1. Temporal Causal Knowledge Graph

```python
from tc_drugrag.knowledge_graph import TemporalEdge, CausalRelationType

# Edges have temporal validity and confidence
edge = TemporalEdge(
    source=drug_node,
    target=target_node,
    relation_type=CausalRelationType.INHIBITS,
    confidence=0.95,
    t_start=datetime(2010, 1, 1),
    t_end=None,  # Still valid
    evidence_type=EvidenceType.RANDOMIZED_CONTROLLED_TRIAL,
)

# Confidence decays over time
current_conf = edge.get_temporal_confidence(datetime.now(), decay_lambda=0.1)
```

### 2. Causal Path Retrieval

```python
# Find causal paths between entities
paths = kg.find_causal_paths(
    source_id="DB00001",  # Metformin
    target_id="DOID:3068",  # Glioblastoma
    max_depth=4,
    min_path_confidence=0.3,
)

# Paths are scored by:
# - Evidence strength (RCT > observational > computational)
# - Causal depth (shorter = more certain)
# - Temporal consistency (causes precede effects)
# - Source diversity (multiple independent sources)
```

### 3. Uncertainty Quantification

```python
# Hypotheses come with calibrated confidence intervals
hypothesis = DrugDiscoveryHypothesis(
    drug="Metformin",
    target="AMPK",
    disease="Glioblastoma",
    mechanism="Metformin activates AMPK → inhibits mTOR → reduces proliferation",
    confidence=0.73,
    confidence_interval=(0.61, 0.85),  # 95% CI via bootstrap
    novelty_score=0.65,
)

# Uncertainty decomposition
decomposition = uncertainty_quantifier.get_uncertainty_decomposition(paths)
# Returns: aleatoric, epistemic, and per-factor breakdown
```

### 4. Closed-Loop Validation

```python
# Validation updates knowledge graph
feedback_loop = FeedbackLoop(kg)
validation_result = pipeline.validate(hypothesis)

if validation_result.validation_passed:
    feedback_loop.update(hypothesis, validation_result)
    # KG now contains new evidence with updated confidence
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Precision@k | Fraction of top-k hypotheses that are correct |
| Recall@k | Fraction of known associations found in top-k |
| NDCG@k | Ranking quality accounting for position |
| Novelty Score | Fraction of hypotheses not in training data |
| ECE | Expected Calibration Error (lower is better) |
| Coverage | 95% CI coverage rate |
| Brier Score | Mean squared prediction error |

## Running Experiments

```bash
# Run main experiment
python experiments/run_experiment.py --config configs/config.yaml

# Run with baseline comparisons
python experiments/run_experiment.py --config configs/config.yaml --run-baselines

# Evaluate results
python experiments/evaluate.py --results experiments/results/latest/results.json
```

## Configuration

Key configuration options in `configs/config.yaml`:

```yaml
# Knowledge Graph
knowledge_graph:
  backend: "neo4j"  # or "networkx"
  temporal:
    enable_temporal_edges: true
    temporal_decay_lambda: 0.1

# Retrieval
retrieval:
  max_path_depth: 4
  min_edge_confidence: 0.3
  ranking:
    evidence_weight: 0.35
    causal_depth_weight: 0.25

# Hypothesis Generation
hypothesis:
  llm:
    provider: "openai"
    model: "gpt-4-turbo-preview"

# Uncertainty
uncertainty:
  method: "bootstrap"
  confidence_level: 0.95
```

## Data Sources

| Source | Description | Size |
|--------|-------------|------|
| DrugBank | Drug-target interactions | ~15K drugs |
| DisGeNET | Gene-disease associations | ~1.1M associations |
| STRING | Protein-protein interactions | ~12K proteins |
| PharmGKB | Pharmacogenomics | ~50K annotations |
| PubMed | Literature (mined) | Configurable |
| ClinicalTrials.gov | Trial outcomes | ~450K trials |

## Citation

If you use TC-DrugRAG in your research, please cite:

```bibtex
@article{tc-drugrag2026,
  title={TC-DrugRAG: Temporal Causal Retrieval-Augmented Generation
         for Hypothesis-Driven Drug Discovery},
  author={Your Name},
  journal={TBD},
  year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Acknowledgments

- Inspired by [CausalRAG](https://arxiv.org/abs/2503.19878), [DrugAgent](https://arxiv.org/abs/2411.15692)
- Built on top of: LangChain, Neo4j, RDKit, PubMedBERT
