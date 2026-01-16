
## Key Features

- **SigExt-inspired phrase extraction**: Extracts salient phrases from news articles using NER and noun chunking
- **Semantic grouping**: Categorizes phrases into WHO/WHAT/WHEN/WHERE/NUMERIC categories
- **Coverage-aware prompting**: Guides summarization models to include key factual information
- **Multi-model comparison**: Evaluates GPT-3.5-turbo and BART-large-CNN
- **Adaptive strategy**: ML-based classifier to select optimal prompting strategy per document
- **Comprehensive evaluation**: ROUGE scores, exact coverage, semantic coverage, number recall, entity recall

## Results

| Model | Strategy | ROUGE-1 | Coverage | Significant? |
|-------|----------|---------|----------|--------------|
| GPT-3.5 | Baseline | 0.339 | 33.0% | - |
| GPT-3.5 | Coverage | 0.334 | 40.1% | ✅ p<10⁻¹² |
| GPT-3.5 | **Smart** | **0.364** | - | **+7.5%** |
| BART | Baseline | 0.351 | 22.3% | - |
| BART | Coverage | 0.337 | 24.7% | - |

**Key Finding**: Coverage-aware prompting significantly improves factual coverage (+21.3%) while maintaining summary quality (ROUGE difference not significant).

## Configuration

Key parameters can be adjusted in Cell 3:

```python
TRAIN_SIZE = 200        # Training documents
TEST_SIZE = 100         # Test documents
TOPK_BUDGET = {         # Phrases per category
    "what": 5, "who": 4, "where": 2,
    "when": 2, "numeric": 2
}
```

## Methodology

1. **Data Split**: Train/test split (200/100) to avoid data leakage
2. **TF-IDF**: Fitted on training corpus only
3. **Phrase Extraction**: NER + noun chunks, no early truncation
4. **Scoring**: Position, TF-IDF, frequency-based relevance scores
5. **Prompting**: Soft constraints (paraphrase OK) + hard constraints (numbers exact)
6. **Evaluation**: Paired t-tests on 100 test documents

## Evaluation Metrics

- **ROUGE-1/2/L**: N-gram overlap with reference summaries
- **Exact Coverage**: Substring matching of extracted phrases
- **Semantic Coverage**: Embedding-based phrase-to-sentence matching (τ=0.45, 0.55)
- **Number Recall**: Numeric values preserved from source
- **Entity Recall**: Named entities preserved from source
