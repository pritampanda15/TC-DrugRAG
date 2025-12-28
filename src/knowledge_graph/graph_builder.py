"""
Graph Builder: Constructs the Temporal Causal Knowledge Graph from multiple data sources.

Integrates data from:
- DrugBank (drug-target interactions)
- DisGeNET (gene-disease associations)
- STRING (protein-protein interactions)
- PubMed (literature-mined relations)
- ClinicalTrials.gov (trial outcomes)
- PharmGKB (pharmacogenomics)
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .entity_extraction import EntityExtractor
from .relation_extraction import CausalRelationExtractor
from .schemas import (
    CausalRelationType,
    EvidenceType,
    Node,
    NodeType,
    TemporalEdge,
)
from .temporal_kg import TemporalCausalKG

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Build the Temporal Causal Knowledge Graph from multiple sources.

    This class orchestrates the ingestion of data from various biomedical
    databases and literature sources to construct a comprehensive knowledge
    graph with temporal causal edges.

    Example:
        >>> builder = GraphBuilder(output_dir="data/processed")
        >>> kg = builder.build(
        ...     sources=["drugbank", "disgenet", "pubmed"],
        ...     pubmed_query="cancer drug resistance",
        ...     max_pubmed_articles=1000,
        ... )
    """

    def __init__(
        self,
        output_dir: str = "data/processed",
        neo4j_config: dict = None,
        use_cache: bool = True,
    ):
        """
        Initialize the graph builder.

        Args:
            output_dir: Directory to save processed data
            neo4j_config: Neo4j connection configuration
            use_cache: Whether to use cached data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.neo4j_config = neo4j_config
        self.use_cache = use_cache

        # Initialize extractors
        self.entity_extractor = None
        self.relation_extractor = None

    def _init_extractors(self):
        """Lazy initialization of extractors."""
        if self.entity_extractor is None:
            self.entity_extractor = EntityExtractor()
        if self.relation_extractor is None:
            self.relation_extractor = CausalRelationExtractor()

    def build(
        self,
        sources: list[str] = None,
        pubmed_query: str = None,
        max_pubmed_articles: int = 10000,
        clinical_trials_query: str = None,
        max_trials: int = 5000,
        n_workers: int = 4,
    ) -> TemporalCausalKG:
        """
        Build the knowledge graph from specified sources.

        Args:
            sources: List of source names to include
            pubmed_query: Query for PubMed literature mining
            max_pubmed_articles: Maximum articles to process
            clinical_trials_query: Query for clinical trials
            max_trials: Maximum trials to process
            n_workers: Number of parallel workers

        Returns:
            Constructed TemporalCausalKG
        """
        sources = sources or ["drugbank", "disgenet", "string"]

        # Initialize knowledge graph
        if self.neo4j_config:
            kg = TemporalCausalKG(
                backend="neo4j",
                **self.neo4j_config,
            )
        else:
            kg = TemporalCausalKG(backend="networkx")

        logger.info(f"Building knowledge graph from sources: {sources}")

        # Process each source
        if "drugbank" in sources:
            self._load_drugbank(kg)

        if "disgenet" in sources:
            self._load_disgenet(kg)

        if "string" in sources:
            self._load_string(kg)

        if "pharmgkb" in sources:
            self._load_pharmgkb(kg)

        if "pubmed" in sources and pubmed_query:
            self._load_pubmed(kg, pubmed_query, max_pubmed_articles, n_workers)

        if "clinicaltrials" in sources and clinical_trials_query:
            self._load_clinical_trials(kg, clinical_trials_query, max_trials)

        # Log statistics
        stats = kg.get_statistics()
        logger.info(f"Knowledge graph built: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

        return kg

    def _load_drugbank(self, kg: TemporalCausalKG) -> None:
        """Load drug-target interactions from DrugBank."""
        logger.info("Loading DrugBank data...")

        drugbank_path = self.output_dir / "drugbank_processed.parquet"

        if self.use_cache and drugbank_path.exists():
            import pandas as pd
            df = pd.read_parquet(drugbank_path)
        else:
            df = self._download_and_process_drugbank()
            if df is not None:
                df.to_parquet(drugbank_path)

        if df is None or df.empty:
            logger.warning("No DrugBank data available")
            return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="DrugBank"):
            # Create drug node
            drug = Node(
                id=row["drugbank_id"],
                name=row["drug_name"],
                node_type=NodeType.DRUG,
                properties={
                    "smiles": row.get("smiles"),
                    "drug_type": row.get("drug_type"),
                },
                external_ids={"drugbank": row["drugbank_id"]},
            )
            kg.add_node(drug)

            # Create target node
            if row.get("target_id"):
                target = Node(
                    id=row["target_id"],
                    name=row.get("target_name", row["target_id"]),
                    node_type=NodeType.PROTEIN,
                    external_ids={"uniprot": row.get("uniprot_id")},
                )
                kg.add_node(target)

                # Create edge
                action = row.get("action", "").lower()
                if "inhibit" in action:
                    rel_type = CausalRelationType.INHIBITS
                elif "activat" in action or "agonist" in action:
                    rel_type = CausalRelationType.ACTIVATES
                else:
                    rel_type = CausalRelationType.BINDS_TO

                edge = TemporalEdge(
                    source=drug,
                    target=target,
                    relation_type=rel_type,
                    confidence=0.95,  # High confidence for curated data
                    t_start=datetime(2000, 1, 1),  # DrugBank has been around since 2006
                    evidence_type=EvidenceType.EXPERT_CURATED,
                    evidence_sources=[f"DrugBank:{row['drugbank_id']}"],
                )
                kg.add_edge(edge)

    def _load_disgenet(self, kg: TemporalCausalKG) -> None:
        """Load gene-disease associations from DisGeNET."""
        logger.info("Loading DisGeNET data...")

        disgenet_path = self.output_dir / "disgenet_processed.parquet"

        if self.use_cache and disgenet_path.exists():
            import pandas as pd
            df = pd.read_parquet(disgenet_path)
        else:
            df = self._download_and_process_disgenet()
            if df is not None:
                df.to_parquet(disgenet_path)

        if df is None or df.empty:
            logger.warning("No DisGeNET data available")
            return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="DisGeNET"):
            # Create gene node
            gene = Node(
                id=row["gene_id"],
                name=row["gene_symbol"],
                node_type=NodeType.GENE,
                external_ids={"ncbi_gene": str(row.get("ncbi_gene_id"))},
            )
            kg.add_node(gene)

            # Create disease node
            disease = Node(
                id=row["disease_id"],
                name=row["disease_name"],
                node_type=NodeType.DISEASE,
                external_ids={"umls": row.get("umls_cui")},
            )
            kg.add_node(disease)

            # Create edge with GDA score as confidence
            edge = TemporalEdge(
                source=gene,
                target=disease,
                relation_type=CausalRelationType.ASSOCIATED_WITH,
                confidence=min(row.get("gda_score", 0.5), 1.0),
                t_start=datetime(2010, 1, 1),
                evidence_type=EvidenceType.LITERATURE_MINING,
                evidence_sources=row.get("pmids", "").split(";")[:5],
            )
            kg.add_edge(edge)

    def _load_string(self, kg: TemporalCausalKG) -> None:
        """Load protein-protein interactions from STRING."""
        logger.info("Loading STRING data...")

        string_path = self.output_dir / "string_processed.parquet"

        if self.use_cache and string_path.exists():
            import pandas as pd
            df = pd.read_parquet(string_path)
        else:
            df = self._download_and_process_string()
            if df is not None:
                df.to_parquet(string_path)

        if df is None or df.empty:
            logger.warning("No STRING data available")
            return

        for _, row in tqdm(df.iterrows(), total=len(df), desc="STRING"):
            # Create protein nodes
            protein1 = Node(
                id=row["protein1"],
                name=row.get("protein1_name", row["protein1"]),
                node_type=NodeType.PROTEIN,
            )
            protein2 = Node(
                id=row["protein2"],
                name=row.get("protein2_name", row["protein2"]),
                node_type=NodeType.PROTEIN,
            )
            kg.add_node(protein1)
            kg.add_node(protein2)

            # Create edge (STRING scores are 0-1000, normalize to 0-1)
            confidence = row.get("combined_score", 500) / 1000.0

            # Determine relation type based on action
            action = row.get("action", "").lower()
            if "activation" in action:
                rel_type = CausalRelationType.ACTIVATES
            elif "inhibition" in action:
                rel_type = CausalRelationType.INHIBITS
            else:
                rel_type = CausalRelationType.BINDS_TO

            edge = TemporalEdge(
                source=protein1,
                target=protein2,
                relation_type=rel_type,
                confidence=confidence,
                t_start=datetime(2015, 1, 1),
                evidence_type=EvidenceType.COMPUTATIONAL,
                evidence_sources=["STRING"],
            )
            kg.add_edge(edge)

    def _load_pharmgkb(self, kg: TemporalCausalKG) -> None:
        """Load pharmacogenomic data from PharmGKB."""
        logger.info("Loading PharmGKB data...")
        # TODO: Implement PharmGKB loading
        pass

    def _load_pubmed(
        self,
        kg: TemporalCausalKG,
        query: str,
        max_articles: int,
        n_workers: int,
    ) -> None:
        """Mine relations from PubMed literature."""
        logger.info(f"Mining PubMed for: {query}")

        self._init_extractors()

        # Fetch article abstracts
        abstracts = self._fetch_pubmed_abstracts(query, max_articles)

        logger.info(f"Processing {len(abstracts)} abstracts...")

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(
                    self._process_abstract, abstract, kg
                ): abstract
                for abstract in abstracts
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="PubMed mining",
            ):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Error processing abstract: {e}")

    def _process_abstract(
        self,
        abstract: dict,
        kg: TemporalCausalKG,
    ) -> None:
        """Process a single PubMed abstract."""
        text = abstract.get("abstract", "")
        pmid = abstract.get("pmid")
        pub_date = abstract.get("pub_date", datetime.now())

        if not text:
            return

        # Extract entities
        entities = self.entity_extractor.extract(text)

        if len(entities) < 2:
            return

        # Extract relations
        edges = self.relation_extractor.extract_relations(
            text=text,
            entities=entities,
            publication_date=pub_date,
            source_id=f"PMID:{pmid}",
            evidence_type=EvidenceType.LITERATURE_MINING,
        )

        # Add to knowledge graph
        for entity in entities:
            kg.add_node(entity)
        for edge in edges:
            kg.add_edge(edge)

    def _load_clinical_trials(
        self,
        kg: TemporalCausalKG,
        query: str,
        max_trials: int,
    ) -> None:
        """Load clinical trial data."""
        logger.info(f"Loading clinical trials for: {query}")
        # TODO: Implement clinical trials loading
        pass

    def _download_and_process_drugbank(self):
        """Download and process DrugBank data."""
        logger.info("DrugBank requires license - please download manually")
        return None

    def _download_and_process_disgenet(self):
        """Download and process DisGeNET data."""
        logger.info("DisGeNET requires registration - please download manually")
        return None

    def _download_and_process_string(self):
        """Download and process STRING data."""
        logger.info("STRING download - please download manually")
        return None

    def _fetch_pubmed_abstracts(
        self,
        query: str,
        max_results: int,
    ) -> list[dict]:
        """Fetch abstracts from PubMed."""
        from Bio import Entrez

        # Set your email for NCBI
        Entrez.email = "your.email@institution.edu"

        try:
            # Search for articles
            handle = Entrez.esearch(
                db="pubmed",
                term=query,
                retmax=max_results,
                sort="relevance",
            )
            record = Entrez.read(handle)
            handle.close()

            pmids = record["IdList"]
            logger.info(f"Found {len(pmids)} articles")

            # Fetch abstracts in batches
            abstracts = []
            batch_size = 100

            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i+batch_size]
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="abstract",
                    retmode="xml",
                )
                records = Entrez.read(handle)
                handle.close()

                for article in records.get("PubmedArticle", []):
                    try:
                        medline = article["MedlineCitation"]
                        pmid = str(medline["PMID"])
                        abstract_text = " ".join(
                            medline["Article"].get("Abstract", {}).get("AbstractText", [])
                        )

                        # Get publication date
                        pub_date_dict = medline["Article"]["Journal"]["JournalIssue"].get(
                            "PubDate", {}
                        )
                        year = int(pub_date_dict.get("Year", 2020))
                        month = pub_date_dict.get("Month", "Jan")

                        abstracts.append({
                            "pmid": pmid,
                            "abstract": abstract_text,
                            "pub_date": datetime(year, 1, 1),
                        })
                    except Exception:
                        continue

            return abstracts

        except Exception as e:
            logger.error(f"Error fetching PubMed abstracts: {e}")
            return []
