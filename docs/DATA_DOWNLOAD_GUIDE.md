# Data Download Guide for TC-DrugRAG

This guide explains how to download all required datasets for TC-DrugRAG.

## Quick Start

```bash
cd tc_drugrag

# Install dependencies first
pip install requests tqdm biopython pandas pyarrow

# Download freely available data
python scripts/download_data.py --source string
python scripts/download_data.py --source pubmed --pubmed-query "drug repurposing cancer" --email your@email.com
python scripts/download_data.py --source clinicaltrials --ct-query "cancer drug"
python scripts/download_data.py --source ttd

# Process downloaded data
python scripts/download_data.py --source string --process
```

---

## Data Sources Overview

| Source | Registration | Size | Use Case |
|--------|--------------|------|----------|
| STRING | **None** | ~2GB | Protein-protein interactions |
| DisGeNET | Academic (free) | ~500MB | Gene-disease associations |
| DrugBank | Academic (free, approval) | ~1GB | Drug-target interactions |
| PubMed | None (email) | Variable | Literature mining |
| PharmGKB | Free | ~100MB | Pharmacogenomics |
| ClinicalTrials.gov | **None** | Variable | Clinical trial outcomes |
| TTD | **None** | ~50MB | Evaluation (drug-target) |

---

## 1. STRING Database (No Registration)

**What it provides**: Protein-protein interactions with confidence scores

### Automatic Download
```bash
python scripts/download_data.py --source string --process
```

### Manual Download
1. Go to https://string-db.org/cgi/download
2. Select organism: **Homo sapiens (9606)**
3. Download:
   - `9606.protein.links.v12.0.txt.gz` (~400MB)
   - `9606.protein.info.v12.0.txt.gz` (~10MB)
   - `9606.protein.actions.v12.0.txt.gz` (~200MB)
4. Place in `data/external/string/`

---

## 2. DisGeNET (Free Academic Registration)

**What it provides**: Gene-disease associations with evidence scores

### Registration Steps
1. Go to https://www.disgenet.org/signup/
2. Fill in your academic information
3. Verify your email
4. Login and go to **Downloads**

### Download
1. After login, go to https://www.disgenet.org/downloads
2. Download **"Curated gene-disease associations"** (`curated_gene_disease_associations.tsv.gz`)
3. Place in `data/external/disgenet/`

### Process
```bash
python scripts/download_data.py --source disgenet --process
```

---

## 3. DrugBank (Academic License Required)

**What it provides**: Comprehensive drug information, targets, interactions

### Registration Steps (Takes 1-3 days for approval)
1. Go to https://go.drugbank.com/public_users/sign_up
2. Select **"Academic/Non-Profit"**
3. Fill in your institution details
4. Wait for approval email (usually 1-2 business days)

### Download
1. After approval, login at https://go.drugbank.com/releases/latest
2. Click **"Download"** on the latest release
3. Download **"All drugs, with targets, enzymes, carriers, and transporters"**
   - File: `drugbank_all_full_database.xml.zip` (~150MB compressed)
4. Extract the XML file
5. Place in `data/external/drugbank/`

### Process
```bash
python scripts/download_data.py --source drugbank --process
```

---

## 4. PubMed (No Registration, Email Recommended)

**What it provides**: Scientific literature abstracts for mining

### Automatic Download
```bash
# Cancer drug resistance literature
python scripts/download_data.py --source pubmed \
    --pubmed-query "cancer drug resistance mechanism" \
    --pubmed-max 10000 \
    --email your.email@institution.edu

# Drug repurposing literature
python scripts/download_data.py --source pubmed \
    --pubmed-query "drug repurposing" \
    --pubmed-max 10000 \
    --email your.email@institution.edu
```

### Get NCBI API Key (Optional, for higher rate limits)
1. Go to https://www.ncbi.nlm.nih.gov/account/
2. Create/login to NCBI account
3. Go to Settings → API Key Management
4. Generate API key
5. Add to `.env`: `PUBMED_API_KEY=your_key`

---

## 5. PharmGKB (Free Registration)

**What it provides**: Pharmacogenomic relationships (gene-drug-phenotype)

### Registration Steps
1. Go to https://www.pharmgkb.org/signUp
2. Register with your email
3. Confirm your email

### Download
1. Login and go to https://www.pharmgkb.org/downloads
2. Download:
   - `relationships.zip` (clinical annotations)
   - `drugs.zip` (drug information)
   - `genes.zip` (gene information)
3. Extract to `data/external/pharmgkb/`

---

## 6. ClinicalTrials.gov (No Registration)

**What it provides**: Clinical trial data with outcomes

### Automatic Download
```bash
# Cancer trials
python scripts/download_data.py --source clinicaltrials --ct-query "cancer drug"

# Specific disease
python scripts/download_data.py --source clinicaltrials --ct-query "glioblastoma"
```

### Manual Download
1. Go to https://clinicaltrials.gov/
2. Search for your topic
3. Click "Download" → "JSON format"

---

## 7. TTD - Therapeutic Target Database (No Registration)

**What it provides**: Validated drug-target pairs (for evaluation)

### Automatic Download
```bash
python scripts/download_data.py --source ttd
```

### Manual Download
1. Go to http://db.idrblab.net/ttd/
2. Navigate to "Download"
3. Download target and drug files
4. Place in `data/external/ttd/`

---

## Directory Structure After Download

```
data/
├── external/
│   ├── string/
│   │   ├── 9606.protein.links.txt.gz
│   │   ├── 9606.protein.info.txt.gz
│   │   └── 9606.protein.actions.txt.gz
│   ├── disgenet/
│   │   └── curated_gene_disease_associations.tsv.gz
│   ├── drugbank/
│   │   └── full_database.xml
│   ├── pubmed/
│   │   └── pubmed_cancer_drug_resistance.json
│   ├── pharmgkb/
│   │   └── relationships.tsv
│   ├── clinicaltrials/
│   │   └── trials_cancer_drug.json
│   └── ttd/
│       └── P1-01-TTD_target_download.txt
├── processed/
│   ├── string_processed.parquet
│   ├── disgenet_processed.parquet
│   └── drugbank_processed.parquet
└── raw/
```

---

## Minimum Data for Testing

For quick testing, you only need:

1. **STRING** (automatic, no registration) - for protein interactions
2. **PubMed** (automatic) - for literature
3. **TTD** (automatic) - for evaluation

```bash
# Minimal setup
python scripts/download_data.py --source string --process
python scripts/download_data.py --source pubmed --pubmed-query "metformin cancer" --pubmed-max 1000
python scripts/download_data.py --source ttd
```

---

## Full Research Setup

For complete research:

```bash
# 1. Automatic downloads
python scripts/download_data.py --source string --process
python scripts/download_data.py --source pubmed --pubmed-query "drug repurposing" --pubmed-max 20000
python scripts/download_data.py --source clinicaltrials --ct-query "cancer drug"
python scripts/download_data.py --source ttd

# 2. Manual downloads (after registration)
# - DisGeNET: Place in data/external/disgenet/
# - DrugBank: Place in data/external/drugbank/
# - PharmGKB: Place in data/external/pharmgkb/

# 3. Process all
python scripts/download_data.py --source disgenet --process
python scripts/download_data.py --source drugbank --process
```

---

## Troubleshooting

### "Rate limit exceeded" (PubMed)
- Add your email: `--email your@email.com`
- Get API key from NCBI

### "Access denied" (DrugBank)
- Registration takes 1-3 days for approval
- Make sure you selected "Academic/Non-Profit"

### "File not found" errors
- Check the exact filename downloaded (versions may differ)
- Update paths in the processing code if needed

### Memory issues with DrugBank XML
- The full DrugBank XML is ~1GB
- Processing requires ~8GB RAM
- Use the streaming parser (already implemented)

---

## Data Size Estimates

| Dataset | Raw Size | Processed Size | Records |
|---------|----------|----------------|---------|
| STRING (human) | ~600MB | ~200MB | ~12M interactions |
| DisGeNET (curated) | ~100MB | ~50MB | ~100K associations |
| DrugBank | ~1GB | ~100MB | ~15K drugs |
| PubMed (10K abstracts) | ~50MB | ~50MB | 10K documents |
| PharmGKB | ~50MB | ~20MB | ~50K relationships |
| TTD | ~10MB | ~5MB | ~3K targets |

**Total**: ~2GB raw, ~500MB processed

---

## Next Steps

After downloading data:

1. Build the knowledge graph:
```python
from tc_drugrag.knowledge_graph import GraphBuilder

builder = GraphBuilder(output_dir="data/processed")
kg = builder.build(sources=["string", "disgenet", "drugbank"])
```

2. Run the demo notebook:
```bash
jupyter notebook notebooks/demo.ipynb
```

3. Run experiments:
```bash
python experiments/run_experiment.py --config configs/config.yaml
```
