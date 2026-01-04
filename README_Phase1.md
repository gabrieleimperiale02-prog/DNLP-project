# Deep NLP Project - Phase 1

## Prompt-Based Abstractive Summarization with Semantic Coverage Control


## 📁 Repository Structure

```
├── Deep_NLP_Phase1.ipynb    # Main notebook for Phase 1
├── README.md                 # This file
├── data/                     # Generated data files
│   ├── validation_samples.json
│   ├── ground_truth_analysis.json
│   ├── extracted_phrases.json
│   ├── grouped_phrases.json
│   ├── grouped_phrases_improved.json
│   ├── extraction_stats.json
│   └── extraction_stats_improved.json
└── results/                  # Analysis results
```

---

## 🎯 Phase 1 Objectives

Phase 1 focuses on **data preparation and semantic extraction pipeline**:

| Task | Description | Status |
|------|-------------|--------|
| Data Loading | Load CNN/DailyMail validation set | ✅ |
| Ground Truth Analysis | Analyze coverage in reference summaries | ✅ |
| Phrase Extraction | Implement SigExt-based extraction | ✅ |
| Semantic Grouping | Group phrases into WHO/WHAT/WHEN/WHERE/NUMERIC | ✅ |
| Improved Extraction | Fix WHAT extraction gap (18% → 100%) | ✅ |
| Statistics | Compute extraction rates per category | ✅ |

---

## 🔬 Methodology

### 1. Dataset
- **CNN/DailyMail** dataset (validation split)
- 200 samples for development/testing
- Articles: avg ~3000 characters
- Highlights: avg ~300 characters (reference summaries)

### 2. Phrase Extraction (SigExt)
We use spaCy for:
- **Named Entity Recognition (NER)**: PERSON, ORG, GPE, DATE, MONEY, etc.
- **Noun Chunks**: Multi-word expressions
- **Verb Phrases**: ROOT verb + direct object

### 3. Semantic Grouping
Extracted phrases are mapped to semantic categories:

| Category | Entity Types | Example |
|----------|-------------|---------|
| WHO | PERSON, ORG, NORP | "President Biden", "Google" |
| WHAT | EVENT, verb phrases | "announced deal", "investigation" |
| WHEN | DATE, TIME | "Monday", "2024" |
| WHERE | GPE, LOC, FAC | "New York", "hospital" |
| NUMERIC | MONEY, PERCENT, CARDINAL | "$5 million", "50%" |

### 4. Improved WHAT Extraction
The baseline SigExt had poor WHAT extraction (18%). We improved it by:
- Extracting verb + object patterns (not just ROOT verbs)
- Adding phrasal verbs (verb + particle)
- Including passive constructions
- Detecting event-related noun phrases

**Result: WHAT extraction improved from 18% → 100%**

---

## 📊 Phase 1 Results

### Ground Truth Coverage Analysis
Category presence in reference summaries:

| Category | % of Documents |
|----------|---------------|
| WHO | 95.5% |
| WHAT | 98.0% |
| WHEN | 54.0% |
| WHERE | 73.5% |
| NUMERIC | 67.0% |

### Extraction Statistics

| Category | Original | Improved | Change |
|----------|----------|----------|--------|
| WHO | 99.5% | 99.5% | — |
| **WHAT** | **18.0%** | **100.0%** | **+82%** ✅ |
| WHEN | 98.0% | 98.0% | — |
| WHERE | 93.0% | 93.5% | +0.5% |
| NUMERIC | 89.0% | 90.0% | +1.0% |

