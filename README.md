# Project 29: Contextual Preference Extraction from Food Reviews

NLP pipeline for a food recommender that extracts **who** likes/dislikes **what**, under **which circumstances**, from Food.com reviews (HUMMUS dataset).

Example input:

> "I love spicy food, but this was too hot for my children. Great for weekend dinners, but too time-consuming for weekdays."

Expected output (holder, aspect, sentiment, context, evidence span):

| Holder   | Aspect          | Sentiment | Context  |
|----------|-----------------|-----------|----------|
| reviewer | spicy food      | positive  | general  |
| children | spiciness       | negative  | general  |
| reviewer | suitability     | positive  | weekend  |
| reviewer | preparation time| negative  | weekdays |

The research question is whether preserving the **holder** and **condition** prevents incorrect generalisation in user profiles and improves recommendation and explanation accuracy.

## Dataset

HUMMUS (Bölz et al., RecSys 2023), loaded from Kaggle (`raiyanahmed/hummus`) via `kagglehub`:

- `pp_reviews.csv`: review `text`, `rating`, `submit_date`, member and recipe ids.
- `pp_recipes.csv`: structured attributes (duration, ingredients, tags, nutrition, healthiness scores).
- `pp_members.csv`: member metadata.

Dataset caveats that affect the analysis:

- Authorship is added as pseudo-reviews with rating `6` and no text.
- Reviews without a rating are set to `2.5`. Only ratings 1-5 are real stars.
- Use only historical reviews (by `submit_date`) when building profiles for evaluation.
- Food.com data is for educational and scientific use only.

## Files

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `load-dataset.py` | Downloads and loads the three HUMMUS tables with `kagglehub`.  |
| `task-1.py`       | Task 1: EDA and text preparation (runs `load-dataset.py`).     |

## Setup

```bash
pip install kagglehub pandas matplotlib seaborn nltk
python task-1.py
```

Kaggle access may require credentials (see the `kagglehub` documentation). `task-1.py` downloads the NLTK tokenizer data on first run.

## Task status

| # | Task                                              | Status |
|---|---------------------------------------------------|--------|
| 1 | EDA and text preparation                          | Done   |
| 2 | NER for food and context (dictionary vs. model)   | To do  |
| 3 | Topic modelling (LDA, NMF, BERTopic)              | To do  |
| 4 | Aspect-based sentiment analysis                   | To do  |
| 5 | Holder and relation extraction                    | To do  |
| 6 | Negation and conditional preference detection     | To do  |
| 7 | Contextual user profile construction              | To do  |
| 8 | Context-aware retrieval and recommendation        | To do  |
| 9 | Evidence-grounded summaries and explanations      | To do  |
| 10| Annotation and evaluation protocol                | To do  |

## Task 1 details

1. Missing values for reviews, members and recipes; drops reviews without text (including authorship rows).
2. Review length (words) statistics and histogram.
3. Rating distribution, reported separately for all rows and for explicit 1-5 star ratings.
4. Frequency of contextual expressions (weekday, weekend, children, partner, guests, special occasion, time-constrained).
5. Sentence and clause splitting with NLTK. Clauses are split **before** contrast words (`but`, `although`, `however`, `yet`, `though`), so contrast and negation words are kept. `df_clauses` keeps ids and a clause index for later evidence spans.

## Planned evaluation

- NER: entity-level P/R/F1.
- Sentiment and condition detection: macro-F1.
- Relation extraction: tuple-level F1.
- Topics: coherence, diversity, human interpretation.
- Recommendation: NDCG@10 and Recall@10 with chronological splits.
- Explanations: evidence support and incorrect-attribution rate.
- Annotation: 200-review pilot, then 800-1,200 reviews, with double annotation and agreement reporting.
