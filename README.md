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

- `pp_reviews.csv`: review `text`, `rating`, `last_modified_date`, member and recipe ids.
- `pp_recipes.csv`: structured attributes (duration, ingredients, tags, nutrition, healthiness scores).
- `pp_members.csv`: member metadata.

Dataset caveats that affect the analysis:

- Authorship is added as pseudo-reviews with rating `6` and no text.
- Reviews without a rating are set to `2.5`. Only ratings 1-5 are real stars.
- Use only historical reviews (by `last_modified_date`) when building profiles for evaluation. There is no separate submission date, and an edit moves this date later, so a chronological split is approximate.
- Food.com data is for educational and scientific use only.

## Files

| File              | Purpose                                                        |
|-------------------|----------------------------------------------------------------|
| `load-dataset.py` | Downloads and loads the three HUMMUS tables with `kagglehub`.  |
| `task-1.py`       | Task 1: EDA and text preparation (needs `load-dataset.py` run first). |
| `task-2.py`       | Task 2: NER for food and context (needs Tasks 1 and loader run first). |
| `task-3.py`       | Task 3: topic modelling of cooking experiences. |
| `task-4.py`       | Task 4: aspect-based sentiment analysis. |
| `task-5.py`       | Task 5: preference-holder and relation extraction. |
| `task-6.py`       | Task 6: negation and conditional preference detection. |
| `task-7.py`       | Task 7: contextual user profile construction. |
| `task-8.py`       | Task 8: context-aware recipe retrieval and recommendation. |
| `task-9.py`       | Task 9: evidence-grounded summaries and explanations. |
| `task-10.py`      | Task 10: annotation sheets, agreement, evaluation and the protocol's status. |
| `ANNOTATION.md`   | Guidelines for the annotators and raters. |
| `LITERATURE.md`   | Related work and the case for the project's novelty (a starting point; check every reference). |
| `annotation_checked.csv` | LLM's pre-annotation of the 1,200 annotation reviews, checked and corrected by the author (clauses marked `checked` = y); the gold for Task 10. |
| `topic_ratings_A.csv` | The author's interpretability ratings of Task 3's topics. These two are the only CSVs tracked in git; upload both to Kaggle for Task 10. |

## Setup

```bash
pip install kagglehub pandas matplotlib seaborn nltk spacy scikit-learn bertopic gensim transformers torch sentencepiece bitsandbytes
python task-1.py
```

Kaggle access may require credentials (see the `kagglehub` documentation). `task-1.py` downloads the NLTK tokenizer data on first run.

## Task status

| # | Task                                              | Status |
|---|---------------------------------------------------|--------|
| 1 | EDA and text preparation                          | Done   |
| 2 | NER for food and context (dictionary vs. model)   | Done (pilot) |
| 3 | Topic modelling (LDA, NMF, BERTopic)              | Done   |
| 4 | Aspect-based sentiment analysis                   | Done (pilot) |
| 5 | Holder and relation extraction                    | Done (pilot) |
| 6 | Negation and conditional preference detection     | Done (pilot) |
| 7 | Contextual user profile construction              | Done (pilot) |
| 8 | Context-aware retrieval and recommendation        | Done (pilot) |
| 9 | Evidence-grounded summaries and explanations      | Done (pilot) |
| 10| Annotation and evaluation protocol                | Done |

## Task 1 details

1. Missing values for reviews, members and recipes; drops reviews without text (including authorship rows). Review text is cleaned of HTML: escaped characters are unescaped ("&#039;" to "'"), `<br/>` tags (some escaped) become line breaks and formatting tags are removed. Line breaks are kept, since 105k reviews use them instead of full stops.
2. Review length (words) statistics and histogram.
3. Rating distribution, reported separately for all rows and for explicit 1-5 star ratings.
4. Frequency of contextual expressions (weekday, weekend, children, partner, guests, special occasion, time-constrained).
5. Sentence and clause splitting with NLTK, line by line. Clauses are split **before** contrast words (`but`, `although`, `however`, `yet`, `though`), so contrast and negation words are kept. `df_clauses` keeps ids and a clause index for later evidence spans.

## Task 2 details

Run order in a Kaggle notebook (one cell per file, same session): `load-dataset.py`, `task-1.py`, `task-2.py`. spaCy is preinstalled on Kaggle; no internet is needed.

- Labels: `INGREDIENT`, `DISH`, `COOKING_EQUIPMENT`, `DURATION`, `OCCASION`, `DINER_GROUP`.
- **Dictionary NER:** spaCy `PhraseMatcher` over lexicons (frequent FoodKG ingredient names and dish phrases from recipe titles, plus hand-written equipment, occasion and diner-group lists) and a regex for durations. Overlaps keep the longest match.
- **Domain-adapted NER:** a spaCy NER model trained on 20k review clauses labelled by the dictionary (weak supervision), so it can tag terms missing from the lexicons. The 200 pilot reviews are held out of training, and the frequency analysis uses clauses not seen in training. Training takes about 5-10 minutes on a CPU.
- **General baseline:** pretrained `en_core_web_sm`, mapped to the project labels where possible (`TIME` to `DURATION`, `DATE`/`EVENT` to `OCCASION`); skipped if the model is not installed.
- **Evaluation:** entity-level exact-match P/R/F1 on a small hand-written gold set (about 40 sentences, in `task-2.py`). This is a pilot only; extend it with the Task 10 annotations. The model is trained on dictionary labels, so a gain over the dictionary on gold reflects generalisation, not agreement with human annotation.
- **Outputs** (in `/kaggle/working`, or the current folder locally): `hummus_ner_model/`, `task2_entities.csv` (entities with offsets, for later tasks), `task2_pilot_annotation.csv` (200 reviews for manual correction).
- Meal times (breakfast, dinner) are not labelled as `OCCASION`; the holder ("I", "we") is left to Task 5.

## Task 3 details

Run after `load-dataset.py` and `task-1.py` in the same session. The cell starts with `!pip install -q bertopic gensim`, so internet must be on (the MiniLM embedding model is also downloaded on first use); a GPU speeds up BERTopic. Outputs are written to `/kaggle/working`.

- **Corpus:** 50,000 random reviews with at least 8 words, the same sample for all three models. HTML, links and Food.com forum-game tags ("made for ZWT") are removed.
- **Vocabulary:** an experience view. Stop words are scikit-learn's English list, generic praise and recipe boilerplate ("great", "recipe", "thanks"), units, and food words (the 1,500 most frequent FoodKG ingredient words and common dish nouns), so topics describe the cooking experience rather than the dish. Context words that also occur in food names are protected ("quick", "family", "party", "crock pot", "healthy"). Unigrams and bigrams, `min_df` 20, `max_df` 0.5.
- **Methods:** LDA (counts), NMF (TF-IDF, KL divergence) and BERTopic (MiniLM embeddings of the original text, UMAP, KMeans instead of HDBSCAN so every review gets a topic, class-TF-IDF with frequent words reduced), all with 15 topics.
- **Evaluation:** c_v and NPMI coherence (gensim, top 10 words, on the tokens the models see), topic diversity (unique share of the top 25 words), representative reviews per topic, and a seed-word check of which topic best matches each expected theme (quick meals, family cooking, meal preparation, celebrations, cooking difficulty; a reading aid, not validation).
- **Human interpretation:** every topic, with its top words and example reviews, is exported for raters to label and score from 1 to 3 (see ANNOTATION.md); Task 10 reads the ratings. The author's ratings (`topic_ratings_A.csv`, one rater): NMF 2.67 (12 of 15 topics a clear theme), BERTopic 2.60 (10 of 15) and LDA 2.33 (6 of 15).
- **Findings:** coherence is similar for the three models (c_v 0.53 for LDA, 0.50 for BERTopic and 0.49 for NMF; NPMI 0.077, 0.077 and 0.060), but NMF's topics are far more distinct (diversity 0.94, against 0.67 for LDA and 0.36 for BERTopic, whose topics share generic words such as "time", "little" and "instead") and the author rated them the most interpretable. BERTopic clusters the embeddings of the full review text, so several of its topics are dish types (cupcakes, scones, chowder, burgers) despite the experience vocabulary; LDA and NMF find experience themes such as quick and easy meals, family meals, substitutions and low-fat versions.
- **Contrast:** NMF topics of recipe directions, to show that the experience themes come from reviews rather than recipes; the per-recipe topic mix is saved.
- **Outputs:** `topic_scores_experience_vocab.csv` (coherence and diversity per model; Task 10 reads it), `topics_for_human_rating_experience_vocab.csv` (for raters) and `recipe_direction_topics.parquet`.
- Topics are exploratory themes, not verified user attributes; they are not copied into user profiles.

## Task 4 details

Run after the earlier cells in the same notebook (it uses `df_clauses`, `split_into_sentences`, `df_recipes`, Task 2's `LEXICONS` and earlier imports). Internet is needed to download the ABSA model; a GPU makes it faster.

- **Aspects:** `TASTE`, `TEXTURE`, `PREP_EFFORT`, `TIME`, `INGREDIENT_AVAILABILITY`, `HEALTHINESS` (perceived, from the text only), and `SUITABILITY`: how well the recipe fits an occasion the clause names ("great for weekend dinners", "a hit at our Christmas party"), so that the brief's "Recipe → suitable for → weekend dinners" becomes a preference. It is found by a judgement word followed within a few words by one of Task 2's occasions; praise without an occasion and an occasion without a judgement ("made it for Christmas") are not fits.
- **Sentiment per aspect per clause**, using Task 1's clauses (split before "but"), so "delicious but difficult" gives positive taste and negative effort. On the whole sentence, the model rates "difficult" positive; per clause it is negative.
- **Aspect detection:** a cue lexicon (e.g. "bland", "mushy", "time consuming", "on hand", "healthy"). Generic praise ("great recipe") names no aspect and produces no row. Cues that are part of a product name before an ingredient ("heavy cream", "low fat cheese", "dry white wine") are skipped.
- **Two sentiment methods:** a lexicon baseline (cue polarity, flipped by a negator in the three words before it; VADER for neutral cues such as "texture") and the pretrained ABSA model `yangheng/deberta-v3-base-absa-v1.1` (clause plus cue as the aspect term).
- **Evaluation:** 45 hand-written reviews (aspect P/R/F1, aspect+sentiment F1, sentiment macro-F1), lexicon-model agreement (Cohen's kappa) and mean star rating per aspect sentiment on 5,000 real reviews.
- **Perceived healthiness vs nutrition:** compared with the recipe's WHO, FSA and Nutri-Score (0-1, higher is healthier), but stored in separate columns and never combined.
- **Outputs:** `task4_aspect_sentiments.csv` (one row per aspect mention with ids, clause, evidence offsets and both sentiments; Tasks 5 and 6 add holder and conditions to these rows) and `task4_pilot_annotation.csv` (the same 200 pilot reviews as Task 2).
- Known limits: aspects without a cue are missed ("took me all afternoon", "melted in your mouth"); "I didn't have X so I used Y" is neutral for the model and negative for the lexicon; the model reads some occasion fits as neutral ("Perfect for a potluck"), so on the 45 hand-written reviews its sentiment macro-F1 is 0.87 against 0.97 for the lexicon.

## Task 5 details

Run after the earlier cells in the same notebook (it uses `df_aspects` and `analyse` from Task 4, `df_clauses` and `split_into_sentences` from Task 1, and `OCCASIONS` and `plural_variants` from Task 2).

- **Tuples:** each Task 4 aspect mention becomes `(holder, aspect, sentiment, context, evidence span)`. Aspect, sentiment and the evidence clause (with cue offsets) come from Task 4; this task adds the holder and the context.
- **Holder:** `reviewer`, `children`, `partner`, `family` (the household as a group, "we", or other relatives), `guests`, or `unknown` when it cannot be told.
- **Context:** the occasion the opinion is tied to, grouped from Task 2's occasion list: `weekday`, `weekend`, `holiday`, `gathering`, `special occasion`; `general` when none is named, `unknown` when a clause names two different ones.
- **Dependency rules (spaCy `en_core_web_sm`):** an experiencer phrase on the opinion word ("too hot for my children", "spicy enough for my husband", "for my taste"), else the subject of the verb above it, climbing through complement clauses ("my husband said it was too salty", "the kids thought it was bland") but not through coordinated verbs. Effort, time and availability go to the reviewer (the cook) unless an experiencer is named. For taste, texture and healthiness, a person named in the clause but not linked to the opinion makes the holder `unknown`; for an occasion fit, people named are its audience ("great for a party with friends"), so it stays the reviewer's.
- **LLM with a fixed output schema:** a free, open model, `Qwen/Qwen3-4B-Instruct-2507` (Apache-2.0, no account or key), run locally with `transformers`. One prompt per mention; the answer must be a JSON object `{"holder": ..., "context": ...}` using only the label lists, and anything else counts as a schema violation and becomes `unknown`. Greedy decoding, so results are repeatable. The model is downloaded once (8 GB, internet on); a GPU (Kaggle: Settings > Accelerator) is much faster. Without it, the rules alone produce the tuples.
- **Evaluation:** holder accuracy and macro-F1, context accuracy and tuple accuracy (holder and context both right) on 40 hand-written reviews; on 40 real reviews, agreement and Cohen's kappa between rules and LLM. The gold set was written alongside the rules and includes cases they cannot solve ("they" referring back to the kids, "everyone" at a party).
- **Output:** `task5_preference_tuples.csv` (one tuple per Task 4 mention, rule-based, with the LLM's labels where it ran).

## Task 6 details

Run after the earlier cells in the same notebook (it uses `df_tuples`, `mentions_for` and `nlp_dep` from Task 5, `CUE_INDEX` and `normalise` from Task 4, and `df_clauses` and `split_into_sentences` from Task 1).

- **Negation scope (spaCy rules):** an opinion word is negated by a negator before it on the word itself or its degree word ("not too spicy"), on the words up to its verb ("wasn't bland", "don't like spicy food"), or on a verb of thinking above its clause ("I don't think it was bland"). Coordinated and subordinate clauses are out of scope ("I didn't add salt and it was bland", "because it was too spicy"), and so are lists ("not too sweet, and fluffy"), compounds ("no-fuss") and surprise ("I never thought I could get it this crispy").
- **Sentiment:** Task 4's model sentiment, except that a cue with its own polarity is reversed when the negation is on the word, a noun it describes, a copula or linking verb, or a verb of thinking ("not too spicy" is positive; "not great for a weeknight" is negative). A negated verb of liking keeps the polarity ("I don't like things too spicy" stays negative), a negated comparative after "could" or "can" is praise ("couldn't have been easier"), and a cue with its own polarity keeps it when it is not negated and the model is unsure (probability below 0.5, as for a clause that is just "Difficult"). The model alone reads "not too spicy" and "never bland" as negative. The model's value is kept in `model_sentiment`.
- **Contrast:** the clause starts with a contrast word (Task 1 splits clauses there).
- **Condition:** an *if*-clause attached to the opinion ("quick to make if you use canned beans"), not indirect questions ("asked if...") or "even if".
- **Exception:** "except/unless/other than ..." after the opinion word, or a following clause "but not at/for/when/if ..." that has no opinion of its own. Each exception adds a tuple with the opposite sentiment and the situation as its condition ("I enjoy spicy food, except at breakfast" gives positive in general and negative "at breakfast"). "except I left out the onions" (a change note) and "but not overly so" (a degree) are not exceptions.
- **Challenge set:** 32 hand-written, manually checked sentences in minimal pairs (negation, negation scope, exceptions, conditions, contrasts), scored by whether every extracted `(aspect, sentiment, condition, negated)` preference matches. Some pairs come from failure patterns found in real reviews.
- **Output:** `task6_preference_tuples.csv` (Task 5's tuples with `negated`, `negator`, `contrast`, `condition`, `condition_type` and `model_sentiment`, plus one row per exception).

## Task 7 details

Run after the earlier cells in the same notebook (it uses `df_reviews` and `DATE_COL` from Task 1, `rules_label` from Task 5 and `preferences_for` from Task 6).

- **Members:** Task 4 sampled random reviews, so most members appear once. This task runs the same extraction (Tasks 4-6) on the 30 most recent reviews of 200 members with at least 10 reviews.
- **General profile:** the reviewer's own opinions with no occasion or condition. **Context profiles:** opinions held by someone else (children, partner, family, guests), or tied to an occasion (weekday, weekend, holiday, gathering, special occasion) or a condition ("if ...", "except ..."). Context opinions are never merged into the general profile, and opinions with an unknown holder are left out.
- **Entries:** one per member, holder, scope and aspect, with the majority preference (likes, dislikes, mixed, neutral), counts, the number of recipes behind it, consistency, mean extraction confidence, the most common cue words, and up to three evidence spans (review id, clause, cue offsets, confidence).
- **Stable vs tentative:** a preference is stable when it holds across at least two recipes with at least two-thirds agreement; a single comment stays tentative, since "too salty" is usually about that recipe.
- **Confidence:** 0.9 where Task 6's negation rule set the sentiment (about 19 of 20 sampled reversals were right), otherwise Task 4's model probability (low where Task 6 overrode an unsure model).
- **Check:** a profile that ignores holders and contexts is built for comparison; the script reports the stable general preferences it would wrongly claim (for example, a child's "yummy" turning into the reviewer's taste preference).
- **Outputs:** `task7_profiles.csv` (profile entries with evidence) and `task7_profile_tuples.csv` (every tuple with ids, date, holder, scope, confidence and evidence offsets). `build_profiles` can be rerun on tuples before a cut-off date for Task 8's chronological evaluation.
- Profiles are per aspect, so "likes TASTE" is common and says little on its own; the cue words and evidence show what was liked.

## Task 8 details

Run after the earlier cells in the same notebook (it uses `df_reviews` and `DATE_COL` from Task 1, `df_recipes`, Task 5's holder and occasion word lists, and `extract` from Task 7). About 3 minutes with a GPU.

- **Requests:** "Recommend a quick dinner for my children." is parsed into a holder (Task 5's holder words: children), an occasion (Task 5's occasion list), and checkable constraints on structured attributes: "quick" means `duration` <= 30 minutes ("under N minutes" sets N), and a meal word requires a tag ("dinner" -> `main-dish`).
- **Score:** recipes that pass the constraints are ranked by the sum of three standardised scores: content (TF-IDF of the request against the recipe's title and tags, without Food.com's tag-category headers), collaborative filtering (truncated SVD with 64 factors on 4-5 star reviews), and the relevant contextual profile (recipes that profile liked minus those it disliked, weighted by extraction confidence; the reviewer's general profile when that holder or occasion has no opinions). The weights are equal and untuned.
- **Evaluation (chronological):** everything is built from reviews before the 90th-percentile date (2014-10-03). For 300 random members with at least 5 earlier reviews, their first 4-5 star recipe after the cut-off is ranked among 49 catalog recipes they never reviewed (catalog: at least 5 earlier reviews), sampled uniformly and in proportion to popularity. Nothing from the held-out review is used, so the request is context-free; Food.com does not record the request behind a review, so contextual matching is evaluated in Task 10 with authored requests. Metrics: NDCG@10 and Recall@10 (random Recall@10 = 0.2).
- **Baselines:** popularity, CF alone, review embeddings (MiniLM; a member is the mean of their earlier reviews, a recipe is up to three earlier reviews by others), and the same extracted-fact profile built without holders and contexts, alone and in the combination.
- **Findings:** with uniform negatives popularity wins (NDCG@10 0.47), because the held-out recipes are mostly popular ones. With popularity-matched negatives, the combination ranks best (NDCG@10 0.18, against 0.17 for CF, 0.15-0.16 for the profiles, 0.14 for popularity, 0.12 for review embeddings and about 0.09 for random); the contextual and context-independent profiles score about the same (0.16 and 0.15), as expected for context-free requests. The differences are small and untested for significance.
- **Output:** `task8_evaluation.csv` (rank of the held-out recipe per member, sampling scheme and method).

## Task 9 details

Run after the earlier cells in the same notebook (it uses Task 8's members, facts and recommender, Task 7's `build_profiles`, and Task 5's local LLM). About 2 minutes for the LLM part with a GPU.

- **Citable facts:** a member's profile entries (likes or dislikes), each with its holder, occasion or condition, how many recipes it rests on, and one verbatim quote from an earlier review (before Task 8's cut-off), numbered E1, E2, ... The recipe's structured record (time, steps, tags) is [R]. Every sentence must cite its sources.
- **Profile summaries:** extractive (the quotes, labelled with holder and occasion) vs generated (Qwen3-4B from Task 5, given the same facts).
- **Recommendation explanations** for a request matching the member's contexts (e.g. "Recommend a quick dinner for my children."): template ("You found lengthy preparation a problem on weekdays for one recipe ("...") [E1]. This recipe takes 20 minutes and has 6 steps [R].") vs generated (Qwen, same facts and record, told that the facts are about other recipes).
- **Context-blind baseline:** the same template explanation for the same recipe, citing a profile built without holders and contexts (as in Task 7's check). Its checks use the holder, occasion and condition each quoted opinion really has, so they measure the attribution errors and lost conditions that keeping holders and contexts avoids.
- **Automatic checks:** factual support (every sentence cites a source that exists; every number appears in the record or a cited quote), attribution (a cited opinion of someone else names them; anyone named is a cited holder or the person the request is for; a sentence about "you" does not rest only on someone else's opinions), and conditions (a cited occasion or condition is kept).
- **Findings (15 members):** extractive summaries and template explanations pass every check by construction but are longer (94 and 52 words, against 65 and 43 for the generated ones). Generated text cites a source in every sentence, but its incorrect-attribution rate is 0.42 for summaries and 0.16 for explanations (for example a sentence about "you" that rests only on the children's or the family's opinion). Reading the generated explanations also showed opinions about past recipes stated as facts about the new one ("your children will love the taste"); the prompt rule against this cut it from about 12 of 15 to about 1 by my reading ("likely to be well-received despite texture concerns noted in past reviews"), but the checks cannot detect it automatically. About 3 of 15 generated explanations still add recipe details that are not in the record ("uses common items available in most kitchens", "includes fresh cheese"). The context-blind template credits 4 of its 32 cited opinions (12%) to the wrong person, against none for the template that keeps holders and contexts; most facts a blind profile cites are still the reviewer's own, so the gap is modest for these members.
- **Output:** `task9_outputs.csv` (each summary or explanation with its facts, review ids and checks).

## Task 10 details

Run after the earlier cells in the same notebook. The annotators' guidelines are in [ANNOTATION.md](ANNOTATION.md). On Kaggle, attach the dataset with `annotation_checked.csv` and `topic_ratings_A.csv` (Task 10 finds it under `/kaggle/input` by itself), turn internet on and use the "GPU T4 x2" accelerator; the cell installs `bitsandbytes` for the relevance judge.

- **Annotation:** `annotation_sheet.csv` holds the 200-review pilot (the reviews of the Task 2 and Task 4 pilot sheets) and 1,000 further reviews, one row per clause (6,149 clauses). They were pre-annotated by LLM, following ANNOTATION.md: entities, and opinions with aspect, sentiment, holder, context, condition and negation, spans written as text. They were labelled in 24 batches of 50 reviews; where batches decided the same case differently, the case was settled once in ANNOTATION.md's Decisions and applied to all (104 clauses changed). When Task 1 was fixed to clean HTML and split at line breaks, the labels were carried over to the new clauses, and the 118 new clause pieces were labelled the same way. When `SUITABILITY` was added, the 175 clauses that mention an occasion were relabelled for it. After the author's check, the labels hold 5,607 entities and 2,297 opinions (TASTE 977, TEXTURE 411, PREP_EFFORT 386, TIME 191, INGREDIENT_AVAILABILITY 179, SUITABILITY 77, HEALTHINESS 76). 94% of opinions are the reviewer's own; 47 carry a condition or exception, and 112 an occasion.
- **Checking:** the author checks and corrects the pre-annotation in `annotation_checked.csv` (a copy with the full review for context) and marks each checked clause `y`. Task 10 uses only checked clauses as gold and scores the system on them: NER precision, recall and F1 (dictionary and model), aspect detection, sentiment, condition and negation macro-F1, and tuple-level F1. It also trains a domain-adapted NER model on the checked main-set clauses and compares it with the dictionary and Task 2's model (trained on dictionary labels) on the checked pilot clauses, which Task 2 held out of training. The brief's double annotation and agreement are not done: there is one annotator, who corrected the pre-annotation instead of labelling from scratch.
- **Findings on the author's checked labels (6,149 clauses):** NER micro P/R/F1 0.75/0.61/0.675 for the dictionary and 0.74/0.60/0.659 for Task 2's model, far below the 0.96 and 0.86 on the hand-written sentences, so a model trained on dictionary labels does not beat the dictionary on real reviews. A model trained on the checked main-set clauses instead reaches 0.78/0.74/0.760 on the checked pilot, against 0.671 for the dictionary and 0.646 for Task 2's model: annotated data, not more dictionary labels, is what improves NER. Aspect detection 0.89/0.65/0.75: what the system finds is mostly right, but it misses a third of the opinions. On matched opinions, sentiment macro-F1 is 0.60, condition 0.50 and negation 0.76; macro-F1 weighs the rare classes (neutral; condition and exception, 47 of 2,297) as much as the common ones. Tuple-level P/R/F1 (aspect, sentiment, holder, context, condition) is 0.73/0.54/0.62.
- **Contextual matching results (LLM-judged; measured locally with Qwen3-4B as the judge, to be replaced by the Kaggle run with Qwen3-14B):** every candidate already meets the request's checkable constraints, so the rubric gives 2 only when the recipe clearly suits the person or occasion named. Even so, the judge rated 219 of the 291 candidates fully relevant (2), 69 partly (1) and 3 not (0); a first, more lenient rubric gave 231, 46 and 14. With most candidates relevant, every method scores between 0.83 and 0.86 NDCG@10 (content only 0.855, combined with the contextual profile 0.849, CF 0.842, combined with the context-independent profile 0.841, popularity 0.828) and about 0.35 Recall@10. The contextual combination is ahead on one request ("Recommend a quick dinner for my children.", +0.18), level on six and slightly behind on three, which on 10 requests does not show that context helps. The judge sees only the request, a few quoted opinions and the recipe's title, time and tags, so most candidates look plausible to it; a human judgment, or a stronger judge, is needed; Qwen3-14B is the stronger judge, and whether it is stricter is not yet known.
- **Contextual matching:** 10 requests written before any candidate was seen ("Recommend a quick dinner for my children.", "A dessert for my guests."), each for a member whose history has opinions of that holder. They were written by the LLM that helped build the pipeline, not by the author, so they are not manually authored in the brief's sense. The top 10 recipes of five methods (combined with the contextual or the context-independent profile, content only, CF, popularity) are pooled (291 candidates, shuffled, methods hidden), and an LLM judges each one 0, 1 or 2 in a fixed JSON schema: Qwen3-14B (Apache-2.0) in 4-bit on a GPU, or Task 5's Qwen3-4B without one; neither takes part in the ranking. It judges from the request, the member's quoted earlier opinions and the recipe's record (2 only when the recipe clearly suits the person or occasion named and does not conflict with those opinions). NDCG@10 and Recall@10 are computed from these judgments. They are LLM judgments, not the human, independent assessment the brief describes. `relevance_sheet.csv` keeps the judgments and `relevance_key.csv` the rankings.
- **Topics:** the author (a second rater is optional) fills Task 3's `topics_for_human_rating_*.csv` (as `topic_ratings_A.csv`); the script reports mean interpretability per model, with agreement when there are two raters. Explanations are checked automatically in Task 9.
- **Status table** (`evaluation_status.csv`): every metric of the protocol, the value measured so far and what is still missing.

| Area | Metric | Measured so far | Still needed |
|---|---|---|---|
| Annotation | clauses checked by the author | all 6,149 clauses of 1,200 reviews | double annotation and agreement (one annotator) |
| NER | entity-level P/R/F1 | checked labels: micro F1 dictionary 0.675, Task 2 model 0.659; on the pilot, a model trained on the checked main set 0.760 vs dictionary 0.671 (41 hand-written sentences: 0.96, 0.86) | |
| Sentiment | macro-F1 | checked labels: 0.60 (45 hand-written reviews: ABSA model 0.87, lexicon 0.97) | |
| Conditions | macro-F1 | checked labels: condition 0.50, negation 0.76 (challenge set: 32 of 32 sentences) | |
| Relation extraction | tuple-level F1 | checked labels: 0.62 (P 0.73, R 0.54) (47 gold mentions, tuple accuracy: rules 0.94, LLM 0.68) | |
| Topics | coherence, diversity, human ratings | NPMI / diversity: LDA 0.077 / 0.67, NMF 0.060 / 0.94, BERTopic 0.077 / 0.36; the author's interpretability (1-3): NMF 2.67, BERTopic 2.60, LDA 2.33 | |
| Recommendation | NDCG@10, Recall@10 | chronological, popularity-matched: combined 0.18, CF 0.17, popularity 0.14 | |
| Contextual matching | NDCG@10, Recall@10 | LLM-judged, 10 requests: 0.83-0.86 NDCG@10 for every method; the judge rated 75% of candidates fully relevant, so the test does not separate the methods | human (or stronger) relevance judgments |
| Explanations | evidence support, incorrect-attribution rate | support 1.00 everywhere; incorrect attribution 0.42 (generated summaries), 0.16 (generated explanations), 0.12 (context-blind template), 0 (extractive, template) | |

**Does keeping holders and conditions help?** Profiles: a profile that ignores them would claim 9 stable general preferences that the members' own opinions do not support (Task 7). Explanations: a template explanation built from such a profile credits 12% of the opinions it cites to the wrong person, against none when they are kept (Task 9). Recommendation: not shown. The chronological test cannot see context, and the two profiles tie there (Task 8); on the 10 requests, the LLM judge rates most candidates fully relevant, even with a strict rubric, so the contextual combination's 0.01 lead in NDCG@10 means little.

The small sets written alongside the systems (Tasks 2, 4, 5 and 6) favour them: on 1,200 real reviews every score drops. The pre-annotation comes from the same model family that designed the system, so scores against it may be optimistic. The author's check removes the errors it finds, but checked labels still lean towards the pre-annotation they started from.
