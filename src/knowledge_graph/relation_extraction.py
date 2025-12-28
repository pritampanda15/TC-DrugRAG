"""
Causal Relation Extraction from Biomedical Text.

Extracts causal relationships between entities with temporal information
and confidence scores.
"""

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .schemas import (
    CausalRelationType,
    EvidenceType,
    Node,
    TemporalEdge,
)

logger = logging.getLogger(__name__)


@dataclass
class RelationCandidate:
    """A candidate relation between two entities."""
    source: Node
    target: Node
    sentence: str
    relation_type: Optional[CausalRelationType] = None
    confidence: float = 0.0


class CausalRelationExtractor:
    """
    Extract causal relations from text between identified entities.

    Uses a combination of:
    1. Pattern-based extraction for high-precision relations
    2. Transformer-based classification for broader coverage

    Example:
        >>> extractor = CausalRelationExtractor()
        >>> drug = Node(id="DB00001", name="Metformin", node_type=NodeType.DRUG)
        >>> target = Node(id="PRKAA1", name="AMPK", node_type=NodeType.PROTEIN)
        >>> edges = extractor.extract_relations(
        ...     "Metformin activates AMPK in hepatocytes",
        ...     [drug, target]
        ... )
    """

    # Pattern-based causal indicators (high precision)
    CAUSAL_PATTERNS = {
        CausalRelationType.ACTIVATES: [
            r"(\w+)\s+activates?\s+(\w+)",
            r"(\w+)\s+stimulates?\s+(\w+)",
            r"(\w+)\s+induces?\s+(\w+)",
            r"activation\s+of\s+(\w+)\s+by\s+(\w+)",
        ],
        CausalRelationType.INHIBITS: [
            r"(\w+)\s+inhibits?\s+(\w+)",
            r"(\w+)\s+blocks?\s+(\w+)",
            r"(\w+)\s+suppresses?\s+(\w+)",
            r"(\w+)\s+reduces?\s+(\w+)",
            r"inhibition\s+of\s+(\w+)\s+by\s+(\w+)",
        ],
        CausalRelationType.BINDS_TO: [
            r"(\w+)\s+binds?\s+(?:to\s+)?(\w+)",
            r"(\w+)\s+interacts?\s+with\s+(\w+)",
            r"binding\s+of\s+(\w+)\s+to\s+(\w+)",
        ],
        CausalRelationType.UPREGULATES: [
            r"(\w+)\s+upregulates?\s+(\w+)",
            r"(\w+)\s+increases?\s+(?:expression\s+of\s+)?(\w+)",
            r"upregulation\s+of\s+(\w+)\s+by\s+(\w+)",
        ],
        CausalRelationType.DOWNREGULATES: [
            r"(\w+)\s+downregulates?\s+(\w+)",
            r"(\w+)\s+decreases?\s+(?:expression\s+of\s+)?(\w+)",
            r"downregulation\s+of\s+(\w+)\s+by\s+(\w+)",
        ],
        CausalRelationType.TREATS: [
            r"(\w+)\s+treats?\s+(\w+)",
            r"(\w+)\s+(?:is\s+)?used\s+(?:for|to\s+treat)\s+(\w+)",
            r"treatment\s+of\s+(\w+)\s+with\s+(\w+)",
        ],
        CausalRelationType.CAUSES: [
            r"(\w+)\s+causes?\s+(\w+)",
            r"(\w+)\s+leads?\s+to\s+(\w+)",
            r"(\w+)\s+results?\s+in\s+(\w+)",
        ],
        CausalRelationType.PREVENTS: [
            r"(\w+)\s+prevents?\s+(\w+)",
            r"(\w+)\s+protects?\s+against\s+(\w+)",
            r"prevention\s+of\s+(\w+)\s+by\s+(\w+)",
        ],
        CausalRelationType.CAUSES_RESISTANCE: [
            r"(\w+)\s+(?:mutation\s+)?(?:causes?|confers?)\s+resistance\s+to\s+(\w+)",
            r"resistance\s+to\s+(\w+)\s+(?:caused\s+)?by\s+(\w+)",
        ],
    }

    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        device: str = None,
        use_patterns: bool = True,
        use_model: bool = True,
    ):
        """
        Initialize the relation extractor.

        Args:
            model_name: HuggingFace model for relation classification
            device: Device to run model on
            use_patterns: Whether to use pattern-based extraction
            use_model: Whether to use model-based extraction
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_patterns = use_patterns
        self.use_model = use_model

        # Compile regex patterns
        self._compiled_patterns = {}
        for rel_type, patterns in self.CAUSAL_PATTERNS.items():
            self._compiled_patterns[rel_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        # Load transformer model for relation classification
        if use_model:
            self._load_relation_model(model_name)
        else:
            self.relation_model = None
            self.relation_tokenizer = None

    def _load_relation_model(self, model_name: str):
        """Load the relation classification model."""
        try:
            logger.info(f"Loading relation model: {model_name}")
            self.relation_tokenizer = AutoTokenizer.from_pretrained(model_name)

            # For now, we'll use the base model - in practice you'd fine-tune
            # a relation extraction model
            self.relation_model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=len(CausalRelationType),
                problem_type="single_label_classification",
            )
            self.relation_model.to(self.device)
            self.relation_model.eval()
        except Exception as e:
            logger.warning(f"Failed to load relation model: {e}")
            self.relation_model = None
            self.relation_tokenizer = None

    def extract_relations(
        self,
        text: str,
        entities: list[Node],
        publication_date: datetime = None,
        source_id: str = None,
        evidence_type: EvidenceType = EvidenceType.LITERATURE_MINING,
    ) -> list[TemporalEdge]:
        """
        Extract causal relations between entities in text.

        Args:
            text: Input text containing entities
            entities: List of entities found in the text
            publication_date: Date of the source document
            source_id: Identifier of the source (e.g., PMID)
            evidence_type: Type of evidence

        Returns:
            List of TemporalEdge objects representing extracted relations
        """
        publication_date = publication_date or datetime.now()
        edges = []

        # Create entity lookup by name
        entity_lookup = {e.name.lower(): e for e in entities}

        # Pattern-based extraction
        if self.use_patterns:
            pattern_edges = self._extract_with_patterns(
                text, entity_lookup, publication_date, source_id, evidence_type
            )
            edges.extend(pattern_edges)

        # Model-based extraction for entity pairs not found by patterns
        if self.use_model and self.relation_model:
            model_edges = self._extract_with_model(
                text, entities, entity_lookup, publication_date, source_id, evidence_type
            )
            # Only add if not already found by patterns
            existing_pairs = {(e.source.id, e.target.id) for e in edges}
            for edge in model_edges:
                if (edge.source.id, edge.target.id) not in existing_pairs:
                    edges.append(edge)

        return edges

    def _extract_with_patterns(
        self,
        text: str,
        entity_lookup: dict,
        publication_date: datetime,
        source_id: str,
        evidence_type: EvidenceType,
    ) -> list[TemporalEdge]:
        """Extract relations using regex patterns."""
        edges = []

        for rel_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    groups = match.groups()
                    if len(groups) >= 2:
                        source_name = groups[0].lower()
                        target_name = groups[1].lower()

                        # Look up entities
                        source_entity = entity_lookup.get(source_name)
                        target_entity = entity_lookup.get(target_name)

                        if source_entity and target_entity:
                            edge = TemporalEdge(
                                source=source_entity,
                                target=target_entity,
                                relation_type=rel_type,
                                confidence=0.85,  # High confidence for pattern match
                                t_start=publication_date,
                                evidence_type=evidence_type,
                                evidence_sources=[source_id] if source_id else [],
                                metadata={"extraction_method": "pattern"},
                            )
                            edges.append(edge)

        return edges

    def _extract_with_model(
        self,
        text: str,
        entities: list[Node],
        entity_lookup: dict,
        publication_date: datetime,
        source_id: str,
        evidence_type: EvidenceType,
    ) -> list[TemporalEdge]:
        """Extract relations using transformer model."""
        edges = []

        if not self.relation_model:
            return edges

        # Generate all entity pairs
        for i, source in enumerate(entities):
            for target in entities[i+1:]:
                # Skip same-type pairs that are unlikely to have relations
                if source.node_type == target.node_type:
                    continue

                # Create input with entity markers
                marked_text = self._mark_entities(text, source, target)

                # Get model prediction
                rel_type, confidence = self._predict_relation(marked_text)

                if rel_type and confidence > 0.5:
                    edge = TemporalEdge(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        confidence=confidence,
                        t_start=publication_date,
                        evidence_type=evidence_type,
                        evidence_sources=[source_id] if source_id else [],
                        metadata={"extraction_method": "model"},
                    )
                    edges.append(edge)

        return edges

    def _mark_entities(self, text: str, source: Node, target: Node) -> str:
        """Mark entities in text for relation classification."""
        # Simple entity marking - in practice, use more sophisticated method
        marked = text.replace(
            source.name, f"[E1]{source.name}[/E1]"
        ).replace(
            target.name, f"[E2]{target.name}[/E2]"
        )
        return marked

    def _predict_relation(
        self,
        marked_text: str,
    ) -> tuple[Optional[CausalRelationType], float]:
        """Predict relation type using the model."""
        inputs = self.relation_tokenizer(
            marked_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.relation_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            confidence, pred_idx = torch.max(probs, dim=-1)

        # Map prediction to relation type
        rel_types = list(CausalRelationType)
        if pred_idx.item() < len(rel_types):
            return rel_types[pred_idx.item()], confidence.item()

        return None, 0.0


class LLMRelationExtractor:
    """
    Use LLM for relation extraction with structured output.

    This approach leverages GPT-4/Claude for more nuanced relation
    extraction, especially for complex causal chains.
    """

    EXTRACTION_PROMPT = """Extract causal relationships from the following biomedical text.

For each relationship, identify:
1. Source entity (drug, gene, protein, etc.)
2. Target entity
3. Relationship type (activates, inhibits, treats, causes, etc.)
4. Confidence (high, medium, low)

Text: {text}

Entities found: {entities}

Output as JSON array:
[{{"source": "...", "target": "...", "relation": "...", "confidence": "...", "evidence": "..."}}]
"""

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4-turbo-preview",
    ):
        """Initialize LLM-based extractor."""
        self.llm_provider = llm_provider
        self.model = model
        self._client = None

    def _get_client(self):
        """Lazy load LLM client."""
        if self._client is None:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            elif self.llm_provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic()
        return self._client

    def extract_relations(
        self,
        text: str,
        entities: list[Node],
        publication_date: datetime = None,
    ) -> list[TemporalEdge]:
        """Extract relations using LLM."""
        publication_date = publication_date or datetime.now()

        entity_names = [e.name for e in entities]
        entity_lookup = {e.name: e for e in entities}

        prompt = self.EXTRACTION_PROMPT.format(
            text=text,
            entities=", ".join(entity_names),
        )

        client = self._get_client()

        if self.llm_provider == "openai":
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = response.choices[0].message.content
        else:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text

        # Parse and convert to edges
        return self._parse_llm_output(
            result, entity_lookup, publication_date
        )

    def _parse_llm_output(
        self,
        output: str,
        entity_lookup: dict,
        publication_date: datetime,
    ) -> list[TemporalEdge]:
        """Parse LLM output into TemporalEdge objects."""
        import json

        edges = []
        try:
            relations = json.loads(output)
            if isinstance(relations, dict):
                relations = relations.get("relations", [])

            for rel in relations:
                source = entity_lookup.get(rel["source"])
                target = entity_lookup.get(rel["target"])

                if source and target:
                    rel_type = self._map_relation_type(rel["relation"])
                    confidence = self._map_confidence(rel.get("confidence", "medium"))

                    edge = TemporalEdge(
                        source=source,
                        target=target,
                        relation_type=rel_type,
                        confidence=confidence,
                        t_start=publication_date,
                        evidence_type=EvidenceType.LITERATURE_MINING,
                        metadata={"extraction_method": "llm", "evidence": rel.get("evidence")},
                    )
                    edges.append(edge)
        except Exception as e:
            logger.error(f"Failed to parse LLM output: {e}")

        return edges

    def _map_relation_type(self, relation_str: str) -> CausalRelationType:
        """Map string relation to CausalRelationType."""
        relation_str = relation_str.lower()

        mapping = {
            "activates": CausalRelationType.ACTIVATES,
            "activate": CausalRelationType.ACTIVATES,
            "inhibits": CausalRelationType.INHIBITS,
            "inhibit": CausalRelationType.INHIBITS,
            "blocks": CausalRelationType.INHIBITS,
            "treats": CausalRelationType.TREATS,
            "treat": CausalRelationType.TREATS,
            "causes": CausalRelationType.CAUSES,
            "cause": CausalRelationType.CAUSES,
            "binds": CausalRelationType.BINDS_TO,
            "bind": CausalRelationType.BINDS_TO,
        }

        return mapping.get(relation_str, CausalRelationType.ASSOCIATED_WITH)

    def _map_confidence(self, confidence_str: str) -> float:
        """Map confidence string to float."""
        mapping = {"high": 0.9, "medium": 0.7, "low": 0.5}
        return mapping.get(confidence_str.lower(), 0.7)
