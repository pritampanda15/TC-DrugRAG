"""
Entity Extraction for Biomedical Text.

Extracts drug, gene, protein, disease, and other biomedical entities
from text using domain-specific NER models.
"""

import logging
from typing import Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from .schemas import Node, NodeType

logger = logging.getLogger(__name__)


class EntityExtractor:
    """
    Extract biomedical entities from text using pre-trained NER models.

    Uses PubMedBERT-based models fine-tuned for biomedical NER.

    Example:
        >>> extractor = EntityExtractor()
        >>> entities = extractor.extract("Metformin inhibits AMPK in diabetes patients")
        >>> print([e.name for e in entities])
        ['Metformin', 'AMPK', 'diabetes']
    """

    # Mapping from NER labels to NodeType
    LABEL_TO_NODE_TYPE = {
        "DRUG": NodeType.DRUG,
        "CHEMICAL": NodeType.DRUG,
        "GENE": NodeType.GENE,
        "PROTEIN": NodeType.PROTEIN,
        "DISEASE": NodeType.DISEASE,
        "SPECIES": None,  # Skip species
        "CELL_LINE": NodeType.CELL_LINE,
        "CELL_TYPE": NodeType.CELL_LINE,
        "DNA": NodeType.GENE,
        "RNA": NodeType.GENE,
    }

    def __init__(
        self,
        model_name: str = "alvaroalon2/biobert_diseases_ner",
        device: str = None,
        batch_size: int = 16,
    ):
        """
        Initialize the entity extractor.

        Args:
            model_name: HuggingFace model name for NER
            device: Device to run model on (cuda, cpu, mps)
            batch_size: Batch size for processing
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        logger.info(f"Loading NER model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.to(self.device)

        self.ner_pipeline = pipeline(
            "ner",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1,
            aggregation_strategy="simple",
        )

        # Entity normalization databases (lazy loaded)
        self._drug_normalizer = None
        self._gene_normalizer = None
        self._disease_normalizer = None

    def extract(
        self,
        text: str,
        normalize: bool = True,
        min_score: float = 0.5,
    ) -> list[Node]:
        """
        Extract entities from text.

        Args:
            text: Input text
            normalize: Whether to normalize entities to standard IDs
            min_score: Minimum confidence score for entities

        Returns:
            List of extracted Node objects
        """
        if not text or not text.strip():
            return []

        # Run NER
        ner_results = self.ner_pipeline(text)

        entities = []
        seen_entities = set()

        for result in ner_results:
            if result["score"] < min_score:
                continue

            entity_text = result["word"].strip()
            entity_label = result["entity_group"]

            # Skip if already seen
            if entity_text.lower() in seen_entities:
                continue
            seen_entities.add(entity_text.lower())

            # Map to NodeType
            node_type = self.LABEL_TO_NODE_TYPE.get(entity_label.upper())
            if node_type is None:
                continue

            # Create node
            node = Node(
                id=self._generate_temp_id(entity_text, node_type),
                name=entity_text,
                node_type=node_type,
                properties={"ner_score": result["score"]},
            )

            # Normalize to standard IDs
            if normalize:
                node = self._normalize_entity(node)

            entities.append(node)

        return entities

    def extract_batch(
        self,
        texts: list[str],
        normalize: bool = True,
        min_score: float = 0.5,
    ) -> list[list[Node]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            normalize: Whether to normalize entities
            min_score: Minimum confidence score

        Returns:
            List of entity lists, one per input text
        """
        all_entities = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            for text in batch:
                entities = self.extract(text, normalize, min_score)
                all_entities.append(entities)

        return all_entities

    def _generate_temp_id(self, name: str, node_type: NodeType) -> str:
        """Generate a temporary ID for an entity."""
        prefix = node_type.value[:3].upper()
        clean_name = name.replace(" ", "_").upper()[:20]
        return f"TEMP_{prefix}_{clean_name}"

    def _normalize_entity(self, node: Node) -> Node:
        """
        Normalize entity to standard database identifiers.

        Maps entity names to DrugBank, HGNC, MeSH, etc.
        """
        if node.node_type == NodeType.DRUG:
            normalized = self._normalize_drug(node.name)
        elif node.node_type in (NodeType.GENE, NodeType.PROTEIN):
            normalized = self._normalize_gene(node.name)
        elif node.node_type == NodeType.DISEASE:
            normalized = self._normalize_disease(node.name)
        else:
            normalized = None

        if normalized:
            node.id = normalized["id"]
            node.external_ids = normalized.get("external_ids", {})

        return node

    def _normalize_drug(self, name: str) -> Optional[dict]:
        """Normalize drug name to DrugBank ID."""
        # Lazy load drug database
        if self._drug_normalizer is None:
            self._drug_normalizer = self._load_drug_database()

        name_lower = name.lower()
        if name_lower in self._drug_normalizer:
            return self._drug_normalizer[name_lower]

        return None

    def _normalize_gene(self, name: str) -> Optional[dict]:
        """Normalize gene/protein name to HGNC symbol."""
        if self._gene_normalizer is None:
            self._gene_normalizer = self._load_gene_database()

        name_upper = name.upper()
        if name_upper in self._gene_normalizer:
            return self._gene_normalizer[name_upper]

        return None

    def _normalize_disease(self, name: str) -> Optional[dict]:
        """Normalize disease name to MeSH/DOID."""
        if self._disease_normalizer is None:
            self._disease_normalizer = self._load_disease_database()

        name_lower = name.lower()
        if name_lower in self._disease_normalizer:
            return self._disease_normalizer[name_lower]

        return None

    def _load_drug_database(self) -> dict:
        """Load drug name to ID mapping."""
        # TODO: Implement loading from DrugBank
        logger.warning("Drug database not loaded - using empty mapping")
        return {}

    def _load_gene_database(self) -> dict:
        """Load gene name to ID mapping."""
        # TODO: Implement loading from HGNC
        logger.warning("Gene database not loaded - using empty mapping")
        return {}

    def _load_disease_database(self) -> dict:
        """Load disease name to ID mapping."""
        # TODO: Implement loading from MeSH/DOID
        logger.warning("Disease database not loaded - using empty mapping")
        return {}


class MultiModelEntityExtractor:
    """
    Combine multiple NER models for comprehensive entity extraction.

    Uses specialized models for different entity types:
    - BioBERT for diseases
    - ChemBERT for chemicals/drugs
    - PubMedBERT for genes/proteins
    """

    def __init__(self, device: str = None):
        """Initialize with multiple specialized extractors."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Model configurations
        self.models = {
            "disease": "alvaroalon2/biobert_diseases_ner",
            "chemical": "alvaroalon2/biobert_chemical_ner",
            "gene": "alvaroalon2/biobert_genetic_ner",
        }

        self.extractors = {}
        self._loaded = False

    def load_models(self):
        """Lazy load all models."""
        if self._loaded:
            return

        for entity_type, model_name in self.models.items():
            logger.info(f"Loading {entity_type} model: {model_name}")
            try:
                self.extractors[entity_type] = EntityExtractor(
                    model_name=model_name,
                    device=self.device,
                )
            except Exception as e:
                logger.warning(f"Failed to load {entity_type} model: {e}")

        self._loaded = True

    def extract(
        self,
        text: str,
        normalize: bool = True,
        min_score: float = 0.5,
    ) -> list[Node]:
        """
        Extract all entity types from text.

        Args:
            text: Input text
            normalize: Whether to normalize entities
            min_score: Minimum confidence score

        Returns:
            Combined list of extracted entities
        """
        self.load_models()

        all_entities = []
        seen_ids = set()

        for extractor in self.extractors.values():
            entities = extractor.extract(text, normalize, min_score)
            for entity in entities:
                if entity.id not in seen_ids:
                    all_entities.append(entity)
                    seen_ids.add(entity.id)

        return all_entities
