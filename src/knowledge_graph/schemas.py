"""
Schema definitions for the Temporal Causal Knowledge Graph.

Defines the structure of nodes, edges, and causal relationship types
with temporal annotations.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class NodeType(Enum):
    """Types of entities in the knowledge graph."""
    DRUG = "Drug"
    GENE = "Gene"
    PROTEIN = "Protein"
    DISEASE = "Disease"
    PATHWAY = "Pathway"
    CLINICAL_TRIAL = "ClinicalTrial"
    PUBLICATION = "Publication"
    SIDE_EFFECT = "SideEffect"
    VARIANT = "Variant"
    CELL_LINE = "CellLine"
    TISSUE = "Tissue"


class CausalRelationType(Enum):
    """Types of causal relationships between entities."""
    # Direct causal
    ACTIVATES = "ACTIVATES"
    INHIBITS = "INHIBITS"
    PHOSPHORYLATES = "PHOSPHORYLATES"
    BINDS_TO = "BINDS_TO"
    CLEAVES = "CLEAVES"

    # Indirect causal
    UPREGULATES = "UPREGULATES"
    DOWNREGULATES = "DOWNREGULATES"
    MODULATES = "MODULATES"

    # Temporal causal
    PRECEDES = "PRECEDES"
    LEADS_TO = "LEADS_TO"
    PREVENTS = "PREVENTS"
    DELAYS = "DELAYS"
    CAUSES_RESISTANCE = "CAUSES_RESISTANCE"

    # Disease-specific
    TREATS = "TREATS"
    CAUSES = "CAUSES"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    PROTECTIVE_AGAINST = "PROTECTIVE_AGAINST"

    # Evidence-based
    MENTIONED_IN = "MENTIONED_IN"
    TESTED_IN = "TESTED_IN"


class EvidenceType(Enum):
    """Types of evidence supporting a relationship."""
    RANDOMIZED_CONTROLLED_TRIAL = "RCT"
    CLINICAL_OBSERVATIONAL = "clinical_observational"
    PRECLINICAL_IN_VIVO = "preclinical_in_vivo"
    PRECLINICAL_IN_VITRO = "preclinical_in_vitro"
    COMPUTATIONAL = "computational"
    LITERATURE_MINING = "literature_mining"
    EXPERT_CURATED = "expert_curated"


@dataclass
class Node:
    """
    Represents an entity in the knowledge graph.

    Attributes:
        id: Unique identifier (e.g., DrugBank ID, HGNC symbol)
        name: Human-readable name
        node_type: Type of entity
        properties: Additional properties (e.g., SMILES for drugs)
        external_ids: Cross-references to external databases
        embedding: Optional vector embedding for similarity search
    """
    id: str
    name: str
    node_type: NodeType
    properties: dict[str, Any] = field(default_factory=dict)
    external_ids: dict[str, str] = field(default_factory=dict)
    embedding: Optional[list[float]] = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "node_type": self.node_type.value,
            "properties": self.properties,
            "external_ids": self.external_ids,
        }


@dataclass
class TemporalEdge:
    """
    Represents a temporal causal relationship between two entities.

    The key innovation: edges have temporal validity windows and
    confidence scores that decay over time.

    Attributes:
        source: Source node
        target: Target node
        relation_type: Type of causal relationship
        confidence: Confidence score [0, 1]
        t_start: Start of temporal validity (when relationship was established)
        t_end: End of temporal validity (None if still valid)
        evidence_type: Type of evidence supporting this relationship
        evidence_sources: List of source identifiers (PMIDs, trial IDs, etc.)
        context: Optional disease/tissue context for the relationship
        metadata: Additional metadata
    """
    source: Node
    target: Node
    relation_type: CausalRelationType
    confidence: float
    t_start: datetime
    t_end: Optional[datetime] = None
    evidence_type: EvidenceType = EvidenceType.LITERATURE_MINING
    evidence_sources: list[str] = field(default_factory=list)
    context: Optional[dict[str, str]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate edge properties."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.t_end and self.t_end < self.t_start:
            raise ValueError("t_end cannot be before t_start")

    def is_valid_at(self, time_point: datetime) -> bool:
        """Check if the edge is valid at a given time point."""
        if time_point < self.t_start:
            return False
        if self.t_end and time_point > self.t_end:
            return False
        return True

    def get_temporal_confidence(
        self,
        time_point: datetime,
        decay_lambda: float = 0.1
    ) -> float:
        """
        Calculate time-decayed confidence score.

        Newer evidence is weighted more heavily than older evidence.

        Args:
            time_point: The time at which to evaluate confidence
            decay_lambda: Decay rate parameter (higher = faster decay)

        Returns:
            Time-adjusted confidence score
        """
        import math

        if not self.is_valid_at(time_point):
            return 0.0

        # Calculate time difference in years
        time_diff = (time_point - self.t_start).days / 365.25

        # Exponential decay
        decay_factor = math.exp(-decay_lambda * time_diff)

        return self.confidence * decay_factor

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_id": self.source.id,
            "target_id": self.target.id,
            "relation_type": self.relation_type.value,
            "confidence": self.confidence,
            "t_start": self.t_start.isoformat(),
            "t_end": self.t_end.isoformat() if self.t_end else None,
            "evidence_type": self.evidence_type.value,
            "evidence_sources": self.evidence_sources,
            "context": self.context,
            "metadata": self.metadata,
        }


@dataclass
class CausalPath:
    """
    Represents a causal path through the knowledge graph.

    A path consists of multiple edges connecting a source to a target
    through intermediate nodes.
    """
    edges: list[TemporalEdge]

    @property
    def source(self) -> Node:
        """Get the source node of the path."""
        return self.edges[0].source if self.edges else None

    @property
    def target(self) -> Node:
        """Get the target node of the path."""
        return self.edges[-1].target if self.edges else None

    @property
    def length(self) -> int:
        """Get the number of edges in the path."""
        return len(self.edges)

    def get_path_confidence(self, time_point: datetime = None) -> float:
        """
        Calculate overall path confidence as product of edge confidences.

        Longer paths naturally have lower confidence.
        """
        if not self.edges:
            return 0.0

        time_point = time_point or datetime.now()

        confidence = 1.0
        for edge in self.edges:
            confidence *= edge.get_temporal_confidence(time_point)

        return confidence

    def is_temporally_consistent(self) -> bool:
        """
        Check if the path is temporally consistent.

        For causal paths, causes should precede effects.
        """
        for i in range(len(self.edges) - 1):
            current_edge = self.edges[i]
            next_edge = self.edges[i + 1]

            # The effect of current edge should precede or coincide with cause of next
            if current_edge.t_start > next_edge.t_start:
                return False

        return True

    def get_mechanism_description(self) -> str:
        """Generate a human-readable description of the causal mechanism."""
        if not self.edges:
            return ""

        parts = [self.edges[0].source.name]
        for edge in self.edges:
            parts.append(f"--[{edge.relation_type.value}]-->")
            parts.append(edge.target.name)

        return " ".join(parts)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "edges": [e.to_dict() for e in self.edges],
            "length": self.length,
            "path_confidence": self.get_path_confidence(),
            "temporally_consistent": self.is_temporally_consistent(),
            "mechanism": self.get_mechanism_description(),
        }


@dataclass
class DrugDiscoveryQuery:
    """
    Structured query for drug discovery hypothesis generation.
    """
    query_type: str  # "repurposing", "target_discovery", "mechanism", "safety"
    drug: Optional[str] = None
    disease: Optional[str] = None
    target: Optional[str] = None
    time_point: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
