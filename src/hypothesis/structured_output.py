"""
Structured Output Definitions for Hypotheses.

Defines the schema for drug discovery hypotheses with
all required fields for downstream validation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class DrugDiscoveryHypothesis:
    """
    A structured drug discovery hypothesis.

    Contains all information needed for evaluation and validation:
    - Drug, target, disease triplet
    - Proposed mechanism of action
    - Supporting evidence summary
    - Confidence with uncertainty bounds
    - Suggested validation experiments

    Example:
        >>> hypothesis = DrugDiscoveryHypothesis(
        ...     drug="Metformin",
        ...     target="AMPK",
        ...     disease="Glioblastoma",
        ...     mechanism="Metformin activates AMPK, leading to mTOR inhibition...",
        ...     confidence=0.73,
        ...     confidence_interval=(0.61, 0.85),
        ... )
    """
    # Core hypothesis fields
    drug: str
    target: str
    disease: str
    mechanism: str

    # Confidence and uncertainty
    confidence: float
    confidence_interval: tuple[float, float]
    novelty_score: float = 0.0

    # Evidence
    evidence_summary: list[str] = field(default_factory=list)
    supporting_paths: list[dict] = field(default_factory=list)
    num_supporting_sources: int = 0

    # Suggested experiments
    suggested_experiments: list[str] = field(default_factory=list)

    # Metadata
    hypothesis_id: str = ""
    generated_at: datetime = field(default_factory=datetime.now)
    generation_method: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.hypothesis_id:
            import hashlib
            content = f"{self.drug}:{self.target}:{self.disease}"
            self.hypothesis_id = hashlib.md5(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "drug": self.drug,
            "target": self.target,
            "disease": self.disease,
            "mechanism": self.mechanism,
            "confidence": self.confidence,
            "confidence_interval": list(self.confidence_interval),
            "novelty_score": self.novelty_score,
            "evidence_summary": self.evidence_summary,
            "num_supporting_sources": self.num_supporting_sources,
            "suggested_experiments": self.suggested_experiments,
            "generated_at": self.generated_at.isoformat(),
            "generation_method": self.generation_method,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DrugDiscoveryHypothesis":
        """Create from dictionary."""
        return cls(
            hypothesis_id=data.get("hypothesis_id", ""),
            drug=data["drug"],
            target=data["target"],
            disease=data["disease"],
            mechanism=data["mechanism"],
            confidence=data["confidence"],
            confidence_interval=tuple(data["confidence_interval"]),
            novelty_score=data.get("novelty_score", 0.0),
            evidence_summary=data.get("evidence_summary", []),
            num_supporting_sources=data.get("num_supporting_sources", 0),
            suggested_experiments=data.get("suggested_experiments", []),
            generated_at=datetime.fromisoformat(data["generated_at"])
            if "generated_at" in data else datetime.now(),
            generation_method=data.get("generation_method", ""),
        )

    def get_summary(self) -> str:
        """Get a human-readable summary of the hypothesis."""
        return (
            f"Hypothesis {self.hypothesis_id}:\n"
            f"  Drug: {self.drug}\n"
            f"  Target: {self.target}\n"
            f"  Disease: {self.disease}\n"
            f"  Confidence: {self.confidence:.2f} "
            f"(95% CI: {self.confidence_interval[0]:.2f}-{self.confidence_interval[1]:.2f})\n"
            f"  Novelty: {self.novelty_score:.2f}\n"
            f"  Mechanism: {self.mechanism}\n"
        )


@dataclass
class HypothesisBatch:
    """
    A batch of related hypotheses for a query.
    """
    query: dict
    hypotheses: list[DrugDiscoveryHypothesis]
    generation_time: float = 0.0
    retrieval_stats: dict = field(default_factory=dict)

    @property
    def best_hypothesis(self) -> Optional[DrugDiscoveryHypothesis]:
        """Get the highest confidence hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.confidence)

    @property
    def most_novel(self) -> Optional[DrugDiscoveryHypothesis]:
        """Get the most novel hypothesis."""
        if not self.hypotheses:
            return None
        return max(self.hypotheses, key=lambda h: h.novelty_score)

    def filter_by_confidence(
        self,
        min_confidence: float = 0.5,
    ) -> list[DrugDiscoveryHypothesis]:
        """Filter hypotheses by minimum confidence."""
        return [h for h in self.hypotheses if h.confidence >= min_confidence]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "query": self.query,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "generation_time": self.generation_time,
            "retrieval_stats": self.retrieval_stats,
        }
