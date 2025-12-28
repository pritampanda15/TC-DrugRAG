#!/usr/bin/env python3
"""
Data Download Script for TC-DrugRAG

Downloads and processes data from various biomedical databases.
Some sources require registration - see instructions below.

Usage:
    python scripts/download_data.py --source all
    python scripts/download_data.py --source drugbank --drugbank-username YOUR_USER --drugbank-password YOUR_PASS
    python scripts/download_data.py --source pubmed --query "cancer drug resistance"
"""

import argparse
import gzip
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EXTERNAL_DIR = DATA_DIR / "external"
PROCESSED_DIR = DATA_DIR / "processed"

# Create directories
for d in [RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def download_file(url: str, output_path: Path, chunk_size: int = 8192) -> bool:
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get("content-length", 0))

        with open(output_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True, desc=output_path.name) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(len(chunk))

        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


# =============================================================================
# 1. STRING Database (Protein-Protein Interactions)
# =============================================================================

def download_string(species: int = 9606):
    """
    Download STRING protein interactions.

    STRING is freely available without registration.
    Species 9606 = Human

    Website: https://string-db.org/
    """
    logger.info("Downloading STRING database...")

    output_dir = EXTERNAL_DIR / "string"
    output_dir.mkdir(exist_ok=True)

    # Protein links (interactions)
    links_url = f"https://stringdb-downloads.org/download/protein.links.v12.0/{species}.protein.links.v12.0.txt.gz"
    links_file = output_dir / f"{species}.protein.links.txt.gz"

    if not links_file.exists():
        logger.info("Downloading protein links...")
        download_file(links_url, links_file)

    # Protein info (names)
    info_url = f"https://stringdb-downloads.org/download/protein.info.v12.0/{species}.protein.info.v12.0.txt.gz"
    info_file = output_dir / f"{species}.protein.info.txt.gz"

    if not info_file.exists():
        logger.info("Downloading protein info...")
        download_file(info_url, info_file)

    # Protein actions (activation, inhibition, etc.)
    actions_url = f"https://stringdb-downloads.org/download/protein.actions.v12.0/{species}.protein.actions.v12.0.txt.gz"
    actions_file = output_dir / f"{species}.protein.actions.txt.gz"

    if not actions_file.exists():
        logger.info("Downloading protein actions...")
        download_file(actions_url, actions_file)

    logger.info(f"STRING data downloaded to {output_dir}")
    return output_dir


def process_string(min_score: int = 700):
    """Process STRING data into usable format."""
    import pandas as pd

    string_dir = EXTERNAL_DIR / "string"
    output_file = PROCESSED_DIR / "string_processed.parquet"

    if output_file.exists():
        logger.info("STRING already processed")
        return

    # Load protein info
    info_file = string_dir / "9606.protein.info.txt.gz"
    if info_file.exists():
        info_df = pd.read_csv(info_file, sep="\t", compression="gzip")
        protein_names = dict(zip(info_df["#string_protein_id"], info_df["preferred_name"]))
    else:
        protein_names = {}

    # Load interactions
    links_file = string_dir / "9606.protein.links.txt.gz"
    if not links_file.exists():
        logger.error("STRING links file not found. Run download first.")
        return

    logger.info("Processing STRING interactions...")
    df = pd.read_csv(links_file, sep=" ", compression="gzip")

    # Filter by score
    df = df[df["combined_score"] >= min_score]

    # Add protein names
    df["protein1_name"] = df["protein1"].map(protein_names)
    df["protein2_name"] = df["protein2"].map(protein_names)

    # Load actions if available
    actions_file = string_dir / "9606.protein.actions.txt.gz"
    if actions_file.exists():
        actions_df = pd.read_csv(actions_file, sep="\t", compression="gzip")
        # Merge actions
        df = df.merge(
            actions_df[["item_id_a", "item_id_b", "action", "score"]],
            left_on=["protein1", "protein2"],
            right_on=["item_id_a", "item_id_b"],
            how="left"
        )

    df.to_parquet(output_file)
    logger.info(f"Processed {len(df)} STRING interactions -> {output_file}")


# =============================================================================
# 2. DisGeNET (Gene-Disease Associations)
# =============================================================================

def download_disgenet(api_key: str = None):
    """
    Download DisGeNET gene-disease associations.

    REGISTRATION REQUIRED (free for academic use):
    1. Go to https://www.disgenet.org/signup/
    2. Register for an account
    3. Get your API key from profile

    Alternative: Download manually from https://www.disgenet.org/downloads
    """
    logger.info("Downloading DisGeNET database...")

    output_dir = EXTERNAL_DIR / "disgenet"
    output_dir.mkdir(exist_ok=True)

    if api_key:
        # Use API
        headers = {"Authorization": f"Bearer {api_key}"}

        # Download curated GDAs
        url = "https://www.disgenet.org/api/gda/gene/"
        # Note: API has rate limits, full download is better via web interface
        logger.info("For full DisGeNET data, download manually from https://www.disgenet.org/downloads")

    # Check for manually downloaded files
    manual_files = [
        "curated_gene_disease_associations.tsv.gz",
        "all_gene_disease_associations.tsv.gz",
    ]

    found = False
    for f in manual_files:
        if (output_dir / f).exists():
            found = True
            logger.info(f"Found {f}")

    if not found:
        logger.warning("""
DisGeNET requires registration. Please:
1. Go to https://www.disgenet.org/signup/
2. Register (free for academic use)
3. Download 'curated_gene_disease_associations.tsv.gz' from Downloads page
4. Place it in: data/external/disgenet/
        """)

    return output_dir


def process_disgenet():
    """Process DisGeNET data."""
    import pandas as pd

    disgenet_dir = EXTERNAL_DIR / "disgenet"
    output_file = PROCESSED_DIR / "disgenet_processed.parquet"

    if output_file.exists():
        logger.info("DisGeNET already processed")
        return

    # Try different file names
    possible_files = [
        "curated_gene_disease_associations.tsv.gz",
        "all_gene_disease_associations.tsv.gz",
        "curated_gene_disease_associations.tsv",
    ]

    input_file = None
    for f in possible_files:
        if (disgenet_dir / f).exists():
            input_file = disgenet_dir / f
            break

    if input_file is None:
        logger.error("DisGeNET file not found. Please download manually.")
        return

    logger.info(f"Processing {input_file}...")

    compression = "gzip" if str(input_file).endswith(".gz") else None
    df = pd.read_csv(input_file, sep="\t", compression=compression)

    # Standardize column names
    df = df.rename(columns={
        "geneId": "ncbi_gene_id",
        "geneSymbol": "gene_symbol",
        "diseaseId": "disease_id",
        "diseaseName": "disease_name",
        "score": "gda_score",
    })

    # Create gene ID
    df["gene_id"] = df["gene_symbol"]

    # Get PMIDs if available
    if "pmid" in df.columns:
        df["pmids"] = df.groupby(["gene_id", "disease_id"])["pmid"].transform(
            lambda x: ";".join(x.astype(str).unique()[:10])
        )

    df.to_parquet(output_file)
    logger.info(f"Processed {len(df)} DisGeNET associations -> {output_file}")


# =============================================================================
# 3. DrugBank (Drug-Target Interactions)
# =============================================================================

def download_drugbank(username: str = None, password: str = None):
    """
    Download DrugBank database.

    REGISTRATION REQUIRED (free for academic use):
    1. Go to https://go.drugbank.com/releases/latest
    2. Register for academic access
    3. Download 'drugbank_all_full_database.xml.zip'

    Note: DrugBank requires approval for academic access.
    """
    logger.info("DrugBank download instructions...")

    output_dir = EXTERNAL_DIR / "drugbank"
    output_dir.mkdir(exist_ok=True)

    # Check for manually downloaded files
    xml_files = list(output_dir.glob("*.xml*"))

    if xml_files:
        logger.info(f"Found DrugBank files: {xml_files}")
    else:
        logger.warning("""
DrugBank requires academic registration. Please:
1. Go to https://go.drugbank.com/releases/latest
2. Click 'Download' and register for academic access
3. Wait for approval (usually 1-2 days)
4. Download 'drugbank_all_full_database.xml.zip'
5. Extract and place XML file in: data/external/drugbank/
        """)

    return output_dir


def process_drugbank():
    """Process DrugBank XML data."""
    import xml.etree.ElementTree as ET
    import pandas as pd

    drugbank_dir = EXTERNAL_DIR / "drugbank"
    output_file = PROCESSED_DIR / "drugbank_processed.parquet"

    if output_file.exists():
        logger.info("DrugBank already processed")
        return

    # Find XML file
    xml_files = list(drugbank_dir.glob("*.xml")) + list(drugbank_dir.glob("*.xml.gz"))

    if not xml_files:
        logger.error("DrugBank XML not found. Please download manually.")
        return

    xml_file = xml_files[0]
    logger.info(f"Processing {xml_file}...")

    # Parse XML (this can take a while for full DrugBank)
    ns = {"db": "http://www.drugbank.ca"}

    records = []

    # Use iterparse for memory efficiency
    for event, elem in ET.iterparse(xml_file, events=["end"]):
        if elem.tag == "{http://www.drugbank.ca}drug":
            try:
                drugbank_id = elem.find("db:drugbank-id[@primary='true']", ns)
                if drugbank_id is not None:
                    drugbank_id = drugbank_id.text

                name = elem.find("db:name", ns)
                name = name.text if name is not None else ""

                # Get targets
                targets = elem.findall(".//db:target", ns)
                for target in targets:
                    target_id = target.find("db:id", ns)
                    target_name = target.find("db:name", ns)
                    polypeptide = target.find(".//db:polypeptide", ns)

                    uniprot_id = None
                    if polypeptide is not None:
                        uniprot_id = polypeptide.get("id")

                    actions = target.findall(".//db:action", ns)
                    action = actions[0].text if actions else ""

                    records.append({
                        "drugbank_id": drugbank_id,
                        "drug_name": name,
                        "target_id": target_id.text if target_id is not None else "",
                        "target_name": target_name.text if target_name is not None else "",
                        "uniprot_id": uniprot_id,
                        "action": action,
                    })
            except Exception as e:
                pass

            elem.clear()

    df = pd.DataFrame(records)
    df.to_parquet(output_file)
    logger.info(f"Processed {len(df)} DrugBank records -> {output_file}")


# =============================================================================
# 4. PubMed (Literature Mining)
# =============================================================================

def download_pubmed(query: str, max_results: int = 10000, email: str = "your.email@example.com"):
    """
    Download PubMed abstracts for a specific query.

    NO REGISTRATION REQUIRED (but email recommended for NCBI tracking)

    Uses NCBI E-utilities API.
    """
    from Bio import Entrez

    logger.info(f"Downloading PubMed abstracts for: {query}")

    output_dir = EXTERNAL_DIR / "pubmed"
    output_dir.mkdir(exist_ok=True)

    Entrez.email = email

    # Search
    logger.info("Searching PubMed...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    handle.close()

    pmids = record["IdList"]
    logger.info(f"Found {len(pmids)} articles")

    # Fetch abstracts in batches
    abstracts = []
    batch_size = 100

    for i in tqdm(range(0, len(pmids), batch_size), desc="Fetching abstracts"):
        batch_pmids = pmids[i:i + batch_size]

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=",".join(batch_pmids),
                rettype="xml",
                retmode="xml",
            )
            records = Entrez.read(handle)
            handle.close()

            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    pmid = str(medline["PMID"])

                    article_data = medline["Article"]
                    title = article_data.get("ArticleTitle", "")

                    abstract_parts = article_data.get("Abstract", {}).get("AbstractText", [])
                    abstract = " ".join(str(p) for p in abstract_parts)

                    # Get publication date
                    pub_date = article_data.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                    year = pub_date.get("Year", "")

                    abstracts.append({
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                    })
                except Exception:
                    continue

            time.sleep(0.34)  # Respect NCBI rate limit (3 requests/sec)

        except Exception as e:
            logger.warning(f"Error fetching batch: {e}")
            time.sleep(1)

    # Save
    query_slug = query.replace(" ", "_")[:50]
    output_file = output_dir / f"pubmed_{query_slug}.json"

    with open(output_file, "w") as f:
        json.dump(abstracts, f, indent=2)

    logger.info(f"Downloaded {len(abstracts)} abstracts -> {output_file}")
    return output_file


# =============================================================================
# 5. PharmGKB (Pharmacogenomics)
# =============================================================================

def download_pharmgkb():
    """
    Download PharmGKB data.

    REGISTRATION REQUIRED (free):
    1. Go to https://www.pharmgkb.org/downloads
    2. Register for an account
    3. Download relationships.zip
    """
    logger.info("PharmGKB download instructions...")

    output_dir = EXTERNAL_DIR / "pharmgkb"
    output_dir.mkdir(exist_ok=True)

    # Check for downloaded files
    if list(output_dir.glob("*.tsv")) or list(output_dir.glob("*.zip")):
        logger.info("Found PharmGKB files")
    else:
        logger.warning("""
PharmGKB requires registration. Please:
1. Go to https://www.pharmgkb.org/downloads
2. Register (free)
3. Download 'relationships.zip'
4. Extract to: data/external/pharmgkb/
        """)

    return output_dir


# =============================================================================
# 6. ClinicalTrials.gov
# =============================================================================

def download_clinicaltrials(query: str, max_results: int = 1000):
    """
    Download clinical trials data.

    NO REGISTRATION REQUIRED - uses public API.
    """
    logger.info(f"Downloading clinical trials for: {query}")

    output_dir = EXTERNAL_DIR / "clinicaltrials"
    output_dir.mkdir(exist_ok=True)

    # Use ClinicalTrials.gov API v2
    base_url = "https://clinicaltrials.gov/api/v2/studies"

    trials = []
    page_token = None

    with tqdm(total=max_results, desc="Fetching trials") as pbar:
        while len(trials) < max_results:
            params = {
                "query.term": query,
                "pageSize": min(100, max_results - len(trials)),
                "format": "json",
            }
            if page_token:
                params["pageToken"] = page_token

            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()

                studies = data.get("studies", [])
                if not studies:
                    break

                for study in studies:
                    protocol = study.get("protocolSection", {})
                    identification = protocol.get("identificationModule", {})
                    status = protocol.get("statusModule", {})
                    description = protocol.get("descriptionModule", {})

                    trials.append({
                        "nct_id": identification.get("nctId"),
                        "title": identification.get("briefTitle"),
                        "status": status.get("overallStatus"),
                        "start_date": status.get("startDateStruct", {}).get("date"),
                        "completion_date": status.get("completionDateStruct", {}).get("date"),
                        "brief_summary": description.get("briefSummary"),
                    })

                pbar.update(len(studies))
                page_token = data.get("nextPageToken")

                if not page_token:
                    break

                time.sleep(0.1)

            except Exception as e:
                logger.warning(f"Error fetching trials: {e}")
                break

    # Save
    query_slug = query.replace(" ", "_")[:50]
    output_file = output_dir / f"trials_{query_slug}.json"

    with open(output_file, "w") as f:
        json.dump(trials, f, indent=2)

    logger.info(f"Downloaded {len(trials)} trials -> {output_file}")
    return output_file


# =============================================================================
# 7. TTD - Therapeutic Target Database (for evaluation)
# =============================================================================

def download_ttd():
    """
    Download TTD for evaluation.

    Freely available from http://db.idrblab.net/ttd/
    """
    logger.info("Downloading TTD database...")

    output_dir = EXTERNAL_DIR / "ttd"
    output_dir.mkdir(exist_ok=True)

    # TTD download URLs (check website for latest)
    files = {
        "drug_target": "http://db.idrblab.net/ttd/sites/default/files/ttd_database/P1-01-TTD_target_download.txt",
        "drug_disease": "http://db.idrblab.net/ttd/sites/default/files/ttd_database/P1-05-Drug_disease.txt",
    }

    for name, url in files.items():
        output_file = output_dir / f"{name}.txt"
        if not output_file.exists():
            logger.info(f"Downloading {name}...")
            download_file(url, output_file)

    return output_dir


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Download data for TC-DrugRAG")
    parser.add_argument(
        "--source",
        type=str,
        choices=["all", "string", "disgenet", "drugbank", "pubmed", "pharmgkb", "clinicaltrials", "ttd"],
        default="all",
        help="Data source to download",
    )
    parser.add_argument("--pubmed-query", type=str, default="drug repurposing cancer")
    parser.add_argument("--pubmed-max", type=int, default=5000)
    parser.add_argument("--ct-query", type=str, default="cancer drug")
    parser.add_argument("--email", type=str, default="your.email@example.com")
    parser.add_argument("--process", action="store_true", help="Also process downloaded data")

    args = parser.parse_args()

    if args.source in ["all", "string"]:
        download_string()
        if args.process:
            process_string()

    if args.source in ["all", "disgenet"]:
        download_disgenet()
        if args.process:
            process_disgenet()

    if args.source in ["all", "drugbank"]:
        download_drugbank()
        if args.process:
            process_drugbank()

    if args.source in ["all", "pubmed"]:
        download_pubmed(args.pubmed_query, args.pubmed_max, args.email)

    if args.source in ["all", "pharmgkb"]:
        download_pharmgkb()

    if args.source in ["all", "clinicaltrials"]:
        download_clinicaltrials(args.ct_query)

    if args.source in ["all", "ttd"]:
        download_ttd()

    print("\n" + "=" * 60)
    print("DATA DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Raw data:       {RAW_DIR}")
    print(f"External data:  {EXTERNAL_DIR}")
    print(f"Processed data: {PROCESSED_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
