"""
Feedback Loop

Updates the knowledge graph based on validation results,
implementing a closed-loop learning system.
"""

import logging
from datetime import datetime
from typing import Optional

from ..hypothesis.structured_output import DrugDiscoveryHypothesis
from ..knowledge_graph import TemporalCausalKG
from ..knowledge_graph.schemas import (
    CausalRelationType,
    EvidenceType,
    Node,
    NodeType,
    TemporalEdge,
)

logger = logging.getLogger(__name__)


class FeedbackLoop:
    """
    Closed-loop feedback for knowledge graph updates.

    When hypotheses are validated (or invalidated), this component
    updates the knowledge graph with the new evidence, enabling
    the system to learn and improve over time.

    Key operations:
    1. Add new edges based on validated hypotheses
    2. Update edge confidences using Bayesian inference
    3. Mark edges as invalidated if hypothesis fails validation
    4. Track validation history for reproducibility

    Example:
        >>> feedback = FeedbackLoop(knowledge_graph)
        >>> feedback.update(hypothesis, validation_result)
        >>> # Knowledge graph now contains updated evidence
    """

    def __init__(
        self,
        knowledge_graph: TemporalCausalKG,
        update_threshold: float = 0.7,
        prior_weight: float = 0.3,
    ):
        """
        Initialize the feedback loop.

        Args:
            knowledge_graph: The knowledge graph to update
            update_threshold: Minimum confidence to add new edges
            prior_weight: Weight given to prior beliefs in Bayesian update
        """
        self.kg = knowledge_graph
        self.update_threshold = update_threshold
        self.prior_weight = prior_weight

        # Track updates for auditing
        self.update_history = []

    def update(
        self,
        hypothesis: DrugDiscoveryHypothesis,
        validation_result,
    ) -> dict:
        """
        Update knowledge graph based on validation result.

        Args:
            hypothesis: The validated hypothesis
            validation_result: Result from validation pipeline

        Returns:
            Dict describing the updates made
        """
        updates = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "timestamp": datetime.now().isoformat(),
            "edges_added": 0,
            "edges_updated": 0,
            "actions": [],
        }

        if not validation_result.validation_passed:
            logger.info(f"Hypothesis {hypothesis.hypothesis_id} did not pass validation")
            updates["actions"].append("no_update_validation_failed")
            return updates

        # Get or create nodes
        drug_node = self._get_or_create_node(
            hypothesis.drug,
            NodeType.DRUG,
        )
        target_node = self._get_or_create_node(
            hypothesis.target,
            NodeType.PROTEIN,
        )
        disease_node = self._get_or_create_node(
            hypothesis.disease,
            NodeType.DISEASE,
        )

        # Determine relation type from mechanism
        relation_type = self._infer_relation_type(hypothesis.mechanism)

        # Calculate new confidence from validation
        new_confidence = self._calculate_validated_confidence(
            hypothesis,
            validation_result,
        )

        if new_confidence >= self.update_threshold:
            # Add/update drug -> target edge
            drug_target_updated = self._update_edge(
                source=drug_node,
                target=target_node,
                relation_type=relation_type,
                new_confidence=new_confidence,
                evidence_source=f"TC-DrugRAG:{hypothesis.hypothesis_id}",
            )

            if drug_target_updated:
                updates["edges_updated" if drug_target_updated == "updated" else "edges_added"] += 1
                updates["actions"].append(f"drug_target_{drug_target_updated}")

            # Add/update target -> disease edge
            target_disease_updated = self._update_edge(
                source=target_node,
                target=disease_node,
                relation_type=CausalRelationType.ASSOCIATED_WITH,
                new_confidence=new_confidence * 0.9,  # Slightly lower for indirect
                evidence_source=f"TC-DrugRAG:{hypothesis.hypothesis_id}",
            )

            if target_disease_updated:
                updates["edges_updated" if target_disease_updated == "updated" else "edges_added"] += 1
                updates["actions"].append(f"target_disease_{target_disease_updated}")

        # Record in history
        self.update_history.append(updates)

        logger.info(
            f"Feedback loop updated KG: {updates['edges_added']} added, "
            f"{updates['edges_updated']} updated"
        )

        return updates

    def _get_or_create_node(
        self,
        name: str,
        node_type: NodeType,
    ) -> Node:
        """Get existing node or create new one."""
        # Try to find existing node
        # In practice, you'd have better entity resolution
        node_id = f"{node_type.value[:3].upper()}_{name.upper().replace(' ', '_')}"

        existing = self.kg.get_node(node_id)
        if existing:
            return existing

        # Create new node
        node = Node(
            id=node_id,
            name=name,
            node_type=node_type,
        )
        self.kg.add_node(node)

        return node

    def _update_edge(
        self,
        source: Node,
        target: Node,
        relation_type: CausalRelationType,
        new_confidence: float,
        evidence_source: str,
    ) -> Optional[str]:
        """
        Update or create an edge in the knowledge graph.

        Returns:
            "added", "updated", or None
        """
        # Check for existing edge
        existing_edges = self.kg.get_outgoing_edges(
            source.id,
            relation_types=[relation_type],
        )

        for edge in existing_edges:
            if edge.target.id == target.id:
                # Update existing edge using Bayesian update
                self.kg.update_edge_confidence(
                    source_id=source.id,
                    target_id=target.id,
                    relation_type=relation_type,
                    new_confidence=new_confidence,
                    evidence_source=evidence_source,
                )
                return "updated"

        # Create new edge
        edge = TemporalEdge(
            source=source,
            target=target,
            relation_type=relation_type,
            confidence=new_confidence,
            t_start=datetime.now(),
            evidence_type=EvidenceType.COMPUTATIONAL,
            evidence_sources=[evidence_source],
            metadata={"source": "tc_drugrag_feedback"},
        )
        self.kg.add_edge(edge)

        return "added"

    def _infer_relation_type(self, mechanism: str) -> CausalRelationType:
        """Infer relation type from mechanism description."""
        mechanism_lower = mechanism.lower()

        if "inhibit" in mechanism_lower or "block" in mechanism_lower:
            return CausalRelationType.INHIBITS
        elif "activat" in mechanism_lower or "stimulat" in mechanism_lower:
            return CausalRelationType.ACTIVATES
        elif "bind" in mechanism_lower:
            return CausalRelationType.BINDS_TO
        elif "upregulat" in mechanism_lower:
            return CausalRelationType.UPREGULATES
        elif "downregulat" in mechanism_lower:
            return CausalRelationType.DOWNREGULATES
        else:
            return CausalRelationType.MODULATES

    def _calculate_validated_confidence(
        self,
        hypothesis: DrugDiscoveryHypothesis,
        validation_result,
    ) -> float:
        """
        Calculate confidence after validation.

        Combines prior hypothesis confidence with validation score.
        """
        prior = hypothesis.confidence
        likelihood = validation_result.overall_score

        # Bayesian update
        # P(H|V) = P(V|H) * P(H) / P(V)
        # Simplified: weighted average with prior
        posterior = (
            self.prior_weight * prior +
            (1 - self.prior_weight) * likelihood
        )

        return posterior

    def rollback_update(self, update_record: dict) -> bool:
        """
        Rollback a previous update.

        Useful for correcting errors or testing.
        """
        # TODO: Implement rollback logic
        logger.warning("Rollback not yet implemented")
        return False

    def get_update_statistics(self) -> dict:
        """Get statistics about feedback loop updates."""
        if not self.update_history:
            return {"total_updates": 0}

        total_added = sum(u["edges_added"] for u in self.update_history)
        total_updated = sum(u["edges_updated"] for u in self.update_history)

        return {
            "total_updates": len(self.update_history),
            "total_edges_added": total_added,
            "total_edges_updated": total_updated,
            "last_update": self.update_history[-1]["timestamp"],
        }
