"""
Query Processor

Processes drug discovery queries to identify entities and
structure the retrieval task.
"""

import logging
import re
from typing import Optional

from ..knowledge_graph import TemporalCausalKG
from ..knowledge_graph.schemas import DrugDiscoveryQuery, Node, NodeType

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Process drug discovery queries for causal retrieval.

    Handles:
    - Entity resolution (mapping names to KG nodes)
    - Query type classification
    - Natural language hypothesis parsing
    """

    # Keywords for query type classification
    QUERY_TYPE_KEYWORDS = {
        "repurposing": ["repurpos", "reposition", "new indication", "off-label"],
        "target_discovery": ["target", "mechanism", "pathway", "binding"],
        "safety": ["toxicity", "adverse", "side effect", "safety", "interaction"],
        "resistance": ["resistance", "resistant", "escape", "refractory"],
    }

    def __init__(self):
        """Initialize the query processor."""
        self._entity_cache = {}

    def process(
        self,
        query: DrugDiscoveryQuery,
        kg: TemporalCausalKG,
    ) -> dict:
        """
        Process a structured query to identify entities in the KG.

        Args:
            query: The drug discovery query
            kg: The knowledge graph for entity lookup

        Returns:
            Dict with 'sources' and 'targets' entity lists
        """
        sources = []
        targets = []

        # Resolve drug entity
        if query.drug:
            drug_node = self._resolve_entity(query.drug, NodeType.DRUG, kg)
            if drug_node:
                sources.append(drug_node)

        # Resolve target entity
        if query.target:
            target_node = self._resolve_entity(
                query.target, NodeType.PROTEIN, kg
            ) or self._resolve_entity(query.target, NodeType.GENE, kg)
            if target_node:
                targets.append(target_node)

        # Resolve disease entity
        if query.disease:
            disease_node = self._resolve_entity(query.disease, NodeType.DISEASE, kg)
            if disease_node:
                targets.append(disease_node)

        return {
            "sources": sources,
            "targets": targets,
            "query_type": query.query_type,
        }

    def _resolve_entity(
        self,
        name: str,
        node_type: NodeType,
        kg: TemporalCausalKG,
    ) -> Optional[Node]:
        """
        Resolve an entity name to a node in the knowledge graph.

        Uses fuzzy matching if exact match not found.
        """
        cache_key = f"{name}:{node_type.value}"
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        # Try exact match first
        node = kg.get_node(name)
        if node and node.node_type == node_type:
            self._entity_cache[cache_key] = node
            return node

        # Try case-insensitive match
        name_lower = name.lower()

        # Search through nodes (in practice, you'd want an index)
        if kg.backend == "networkx":
            for node_id, node in kg._nodes.items():
                if node.node_type == node_type:
                    if node.name.lower() == name_lower:
                        self._entity_cache[cache_key] = node
                        return node

        # Try partial/fuzzy matching
        best_match = None
        best_score = 0

        if kg.backend == "networkx":
            for node_id, node in kg._nodes.items():
                if node.node_type == node_type:
                    score = self._calculate_match_score(name_lower, node.name.lower())
                    if score > best_score and score > 0.7:
                        best_score = score
                        best_match = node

        if best_match:
            self._entity_cache[cache_key] = best_match
            logger.info(f"Fuzzy matched '{name}' to '{best_match.name}'")
            return best_match

        logger.warning(f"Could not resolve entity: {name} ({node_type.value})")
        return None

    def _calculate_match_score(self, query: str, candidate: str) -> float:
        """Calculate simple string similarity score."""
        # Longest common substring ratio
        if not query or not candidate:
            return 0.0

        # Check if one contains the other
        if query in candidate or candidate in query:
            return 0.9

        # Simple character overlap ratio
        query_set = set(query)
        candidate_set = set(candidate)
        intersection = query_set & candidate_set
        union = query_set | candidate_set

        if not union:
            return 0.0

        return len(intersection) / len(union)

    def parse_hypothesis(
        self,
        hypothesis: str,
        entities: list[Node] = None,
    ) -> DrugDiscoveryQuery:
        """
        Parse a natural language hypothesis into a structured query.

        Examples:
            "Metformin could treat glioblastoma by inhibiting mTOR"
            → drug=Metformin, disease=glioblastoma, target=mTOR

        Args:
            hypothesis: Natural language hypothesis
            entities: Optional pre-extracted entities

        Returns:
            Structured DrugDiscoveryQuery
        """
        hypothesis_lower = hypothesis.lower()

        # Classify query type
        query_type = "general"
        for qtype, keywords in self.QUERY_TYPE_KEYWORDS.items():
            if any(kw in hypothesis_lower for kw in keywords):
                query_type = qtype
                break

        # Extract entities if not provided
        drug = None
        disease = None
        target = None

        if entities:
            for entity in entities:
                if entity.node_type == NodeType.DRUG:
                    drug = entity.name
                elif entity.node_type == NodeType.DISEASE:
                    disease = entity.name
                elif entity.node_type in (NodeType.GENE, NodeType.PROTEIN):
                    target = entity.name
        else:
            # Simple pattern-based extraction
            drug, disease, target = self._extract_from_text(hypothesis)

        return DrugDiscoveryQuery(
            query_type=query_type,
            drug=drug,
            disease=disease,
            target=target,
            context={"original_hypothesis": hypothesis},
        )

    def _extract_from_text(self, text: str) -> tuple:
        """Extract drug, disease, target from text using patterns."""
        # These are simplified patterns - in practice, use NER
        drug = None
        disease = None
        target = None

        # Pattern: "X could treat Y"
        treat_match = re.search(
            r"(\w+)\s+(?:could|may|might|can)\s+treat\s+(\w+)",
            text,
            re.IGNORECASE,
        )
        if treat_match:
            drug = treat_match.group(1)
            disease = treat_match.group(2)

        # Pattern: "by inhibiting/activating X"
        mechanism_match = re.search(
            r"by\s+(?:inhibiting|activating|targeting|blocking)\s+(\w+)",
            text,
            re.IGNORECASE,
        )
        if mechanism_match:
            target = mechanism_match.group(1)

        return drug, disease, target

    def expand_query_with_synonyms(
        self,
        query: DrugDiscoveryQuery,
        kg: TemporalCausalKG,
    ) -> list[DrugDiscoveryQuery]:
        """
        Expand a query with entity synonyms for broader retrieval.

        Uses the knowledge graph's external IDs and aliases.
        """
        expanded = [query]

        # Get synonyms for drug
        if query.drug:
            drug_node = self._resolve_entity(query.drug, NodeType.DRUG, kg)
            if drug_node and drug_node.external_ids:
                for db, ext_id in drug_node.external_ids.items():
                    new_query = DrugDiscoveryQuery(
                        query_type=query.query_type,
                        drug=ext_id,
                        disease=query.disease,
                        target=query.target,
                        time_point=query.time_point,
                        context={**query.context, "synonym_source": db},
                    )
                    expanded.append(new_query)

        return expanded
