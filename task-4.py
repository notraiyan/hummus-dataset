# Imports already made by task-1.py to task-3.py (same notebook): json, os, re, plt, nltk, np, pd, sns.
import torch
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics import cohen_kappa_score, f1_score
from transformers import pipeline

# Task 4: Aspect-Based Sentiment Analysis
# Run after load-dataset.py and task-1.py to task-3.py in the same notebook (this needs df_clauses,
# split_into_sentences, df_recipes, Task 2's LEXICONS and the earlier cells' imports).
# Kaggle: transformers is preinstalled; turn internet on so the ABSA model can be downloaded. A GPU is much faster.
# Sentiment is attached to an aspect in one clause, not to the whole review, so "delicious but difficult" gives
# positive TASTE and negative PREP_EFFORT. Rows keep clause offsets and ids so Task 5 (holder) and Task 6
# (negation, conditions) can attach to them, and Task 9 can cite the evidence span.

for required in ("df_clauses", "split_into_sentences", "df_recipes"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py in this session first.")

SEED = 42
N_REVIEWS = 5_000         # reviews whose clauses are analysed on the real data
ABSA_MODEL = "yangheng/deberta-v3-base-absa-v1.1"
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
nltk.download("vader_lexicon", quiet=True)

# SUITABILITY: how well the recipe fits an occasion the clause names ("great for weekend dinners").
ASPECTS = ["TASTE", "TEXTURE", "PREP_EFFORT", "TIME", "INGREDIENT_AVAILABILITY", "HEALTHINESS", "SUITABILITY"]
SENTIMENTS = ["positive", "neutral", "negative"]

# ==========================================
# 1. ASPECT CUES
# ==========================================
# cue -> prior polarity (+1, -1, or 0 when the cue names the aspect without judging it, e.g. "texture").
# Perceived healthiness comes only from the review text; nutrition scores are compared in section 6, never merged.
# Generic praise ("great", "loved it") names no aspect and produces no row, so it cannot be generalised to one.
ASPECT_CUES = {
    "TASTE": {
        "taste": 0, "tastes": 0, "tasted": 0, "flavor": 0, "flavour": 0, "flavors": 0, "flavours": 0, "seasoning": 0,
        "spicy": 0, "sweet": 0, "salty": 0, "flavorful": 1, "flavourful": 1, "delicious": 1, "tasty": 1, "yummy": 1,
        "scrumptious": 1, "savory": 1, "decadent": 1, "bland": -1, "flavorless": -1, "tasteless": -1, "boring": -1,
        "bitter": -1, "overpowering": -1, "too salty": -1, "too sweet": -1, "too spicy": -1, "too hot": -1,
        "too strong": -1, "lacked flavor": -1, "no flavor": -1, "not much flavor": -1,
    },
    "TEXTURE": {
        "texture": 0, "textures": 0, "consistency": 0, "chewy": 0, "soft": 0, "dense": 0, "moist": 1, "tender": 1,
        "crispy": 1, "crisp": 1, "crunchy": 1, "fluffy": 1, "creamy": 1, "juicy": 1, "flaky": 1, "smooth": 1,
        "gooey": 1, "mushy": -1, "soggy": -1, "dry": -1, "tough": -1, "rubbery": -1, "gummy": -1, "grainy": -1,
        "runny": -1, "watery": -1, "greasy": -1, "chalky": -1, "lumpy": -1, "too dense": -1, "too thick": -1,
        "too thin": -1,
    },
    "PREP_EFFORT": {
        "effort": 0, "prep": 0, "preparation": 0, "steps": 0, "easy": 1, "easier": 1, "simple": 1, "effortless": 1,
        "foolproof": 1, "fool proof": 1, "straightforward": 1, "no fuss": 1, "easy to make": 1, "easy to put together": 1,
        "difficult": -1, "hard to make": -1, "complicated": -1, "tricky": -1, "tedious": -1, "fussy": -1,
        "labor intensive": -1, "messy": -1, "a mess": -1, "a pain": -1, "challenging": -1, "too many steps": -1,
    },
    "TIME": {
        # "quickly" is left out: in reviews it is mostly an instruction ("work quickly"), not a judgement of time.
        "cooking time": 0, "prep time": 0, "baking time": 0, "quick": 1, "fast": 1, "in no time": 1,
        "time consuming": -1, "took forever": -1, "takes forever": -1, "too long": -1, "takes a while": -1,
        "took a while": -1, "all day": -1,
    },
    "INGREDIENT_AVAILABILITY": {
        "pantry": 0, "available": 0, "on hand": 1, "staples": 1, "common ingredients": 1, "simple ingredients": 1,
        "basic ingredients": 1, "easy to find": 1, "readily available": 1, "hard to find": -1, "couldn't find": -1,
        "could not find": -1, "can't find": -1, "cannot find": -1, "didn't have": -1, "did not have": -1,
        "don't have": -1, "ran out of": -1, "not available": -1, "unavailable": -1, "specialty store": -1,
    },
    "HEALTHINESS": {
        "calories": 0, "calorie": 0, "healthy": 1, "healthier": 1, "nutritious": 1, "wholesome": 1, "guilt free": 1,
        "lighter": 1, "low fat": 1, "low calorie": 1, "low cal": 1, "good for you": 1, "unhealthy": -1,
        "fattening": -1, "heavy": -1, "too rich": -1,
    },
}


def normalise(cue):
    return re.sub(r"[\s-]+", " ", cue.lower().replace("'", "").replace("’", ""))


CUE_INDEX = {normalise(cue): (aspect, prior) for aspect, cues in ASPECT_CUES.items() for cue, prior in cues.items()}
# Longest cue first so "easy to find" (availability) wins over "easy" (effort) and "too salty" over "salty".
_cue_patterns = [
    re.escape(cue).replace(r"\ ", r"[\s-]+").replace("'", "['’]?")
    for cue in sorted({c for cues in ASPECT_CUES.values() for c in cues}, key=len, reverse=True)
]
CUE_RE = re.compile(r"\b(?:" + "|".join(_cue_patterns) + r")\b", re.IGNORECASE)


# Cues that are often part of a product name: before an ingredient they describe what was bought, not how the dish
# turned out ("heavy cream", "low fat cheese", "dry white wine", "sweet potatoes", "creamy peanut butter"). Ingredient
# words come from the Task 2 lexicon in the same notebook. Other cues are kept before ingredients ("juicy chicken").
PRODUCT_CUES = {"heavy", "low fat", "low calorie", "low cal", "lighter", "dry", "sweet", "salty", "spicy", "smooth",
                "crunchy", "creamy", "soft"}
INGREDIENT_WORDS = set(LEXICONS["INGREDIENT"]) if "LEXICONS" in globals() else set()
INGREDIENT_WORDS |= {"cream", "cheese", "milk", "mayo", "mayonnaise", "yogurt", "yoghurt", "wine", "ingredients", "rub"}
NEXT_WORDS_RE = re.compile(r"[\s-]*([a-z']+)(?:\s+([a-z']+))?")


def is_product_or_idiom(clause, m, aspect):
    following = NEXT_WORDS_RE.match(clause[m.end():].lower())
    w1, w2 = (following.group(1), following.group(2) or "") if following else ("", "")
    if aspect == "INGREDIENT_AVAILABILITY" and w1 == "to":  # "don't have to", "didn't have to"
        return True
    return normalise(m.group(0)) in PRODUCT_CUES and (w1 in INGREDIENT_WORDS or f"{w1} {w2}" in INGREDIENT_WORDS)


# Occasion fit: a judgement word followed, within a few words, by an occasion from Task 2's lexicon ("great for
# weekend dinners", "perfect for a potluck", "a hit at our Christmas party"). Praise without an occasion ("great
# recipe") and an occasion without a judgement ("made it for Christmas") are not fits. Task 5 reads the occasion as the
# context, and Task 6's negation scope turns "not good for a weeknight" negative.
OCCASION_WORDS = sorted(LEXICONS["OCCASION"] if "LEXICONS" in globals() else
                        {"weeknight", "weeknights", "weekend", "weekends", "party", "potluck", "holiday", "holidays",
                         "christmas", "thanksgiving"}, key=len, reverse=True)
FIT_RE = re.compile(r"\b(?:great|perfect|good|ideal|nice|wonderful|excellent|fantastic|awesome|terrific|suitable|"
                    r"a (?:big |huge )?hit)(?:\s+[a-z']+){0,2}?\s+(?:for|at|on)\s+(?:[a-z'-]+\s+){0,3}?"
                    r"(?:" + "|".join(map(re.escape, OCCASION_WORDS)) + r")\b", re.IGNORECASE)


def find_mentions(clause):
    """One mention per aspect per clause: a polar cue is preferred over a neutral one, then the longer cue."""
    best = {}
    for m in CUE_RE.finditer(clause):
        aspect, prior = CUE_INDEX[normalise(m.group(0))]
        if is_product_or_idiom(clause, m, aspect):
            continue
        rank = (prior != 0, m.end() - m.start(), -m.start())
        if aspect not in best or rank > best[aspect][0]:
            best[aspect] = (rank, {"aspect": aspect, "cue": m.group(0), "start": m.start(), "end": m.end(), "prior": prior})
    fit = FIT_RE.search(clause)
    if fit:
        best["SUITABILITY"] = (None, {"aspect": "SUITABILITY", "cue": fit.group(0), "start": fit.start(),
                                      "end": fit.end(), "prior": 1})
    return [mention for _, mention in best.values()]


# ==========================================
# 2. SENTIMENT: LEXICON BASELINE AND PRETRAINED ABSA MODEL
# ==========================================
NEGATORS = {"not", "no", "never", "hardly", "without", "nothing", "isnt", "wasnt", "arent", "werent", "dont",
            "doesnt", "didnt", "cant", "cannot", "couldnt", "wouldnt", "wont", "aint"}
vader = SentimentIntensityAnalyzer()


def lexicon_sentiment(clause, mention):
    """The cue's prior, flipped by a negator in the three words before it ("not too difficult" -> positive).
    Cues without a prior ("texture") take the VADER score of the clause; "too" before them is negative."""
    before = [w.replace("'", "").replace("’", "") for w in re.findall(r"[\w'’]+", clause[:mention["start"]].lower())[-3:]]
    if mention["prior"]:
        polarity = -mention["prior"] if any(w in NEGATORS for w in before) else mention["prior"]
    elif before[-1:] == ["too"]:
        polarity = -1
    else:
        compound = vader.polarity_scores(clause)["compound"]
        polarity = 1 if compound >= 0.05 else -1 if compound <= -0.05 else 0
    return {1: "positive", 0: "neutral", -1: "negative"}[polarity]


device = 0 if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else -1)
absa = pipeline("text-classification", model=ABSA_MODEL, device=device)


def model_sentiment(pairs, batch_size=64):
    """pairs: [(clause, cue)]. The model scores sentiment toward the cue inside the clause."""
    if not pairs:
        return []
    out = absa([{"text": c, "text_pair": a} for c, a in pairs], batch_size=batch_size, truncation=True)
    return [(o["label"].lower(), o["score"]) for o in out]


def analyse(clause_texts):
    """Aspect mentions with both sentiments for a list of clauses. Returns one dict per mention."""
    mentions = [dict(m, clause_pos=i) for i, text in enumerate(clause_texts) for m in find_mentions(text)]
    predictions = model_sentiment([(clause_texts[m["clause_pos"]], m["cue"]) for m in mentions])
    for m, (label, score) in zip(mentions, predictions):
        m["lexicon_sentiment"] = lexicon_sentiment(clause_texts[m["clause_pos"]], m)
        m["model_sentiment"], m["model_score"] = label, score
    return mentions


# ==========================================
# 3. EVALUATION SET (small, hand-written pilot gold)
# ==========================================
# Whole reviews, split into clauses exactly as in Task 1. Gold is a set of (aspect, sentiment) per review; an aspect
# can appear with two sentiments ("too spicy" for one diner, "loved the heat" for another). Some items are
# deliberately beyond the lexicon ("took me all afternoon", "melted in your mouth"). Extend with Task 10 annotations.
P, N, O = "positive", "negative", "neutral"
GOLD = [
    ("This was delicious but difficult to make.", {("TASTE", P), ("PREP_EFFORT", N)}),
    ("Quick and easy weeknight dinner, and the kids loved the flavor.", {("TIME", P), ("PREP_EFFORT", P), ("TASTE", P)}),
    ("The texture was mushy and it was way too salty.", {("TEXTURE", N), ("TASTE", N)}),
    ("Not too difficult, but it took forever.", {("PREP_EFFORT", P), ("TIME", N)}),
    ("I had everything on hand, which is always a plus.", {("INGREDIENT_AVAILABILITY", P)}),
    ("I couldn't find tamarind paste anywhere, so I used lime juice.", {("INGREDIENT_AVAILABILITY", N)}),
    ("A nice healthy dinner that doesn't taste like diet food.", {("HEALTHINESS", P), ("TASTE", P)}),
    ("Way too heavy and greasy for us.", {("HEALTHINESS", N), ("TEXTURE", N)}),
    ("Great recipe, thanks for posting!", set()),
    ("The chicken was moist and tender.", {("TEXTURE", P)}),
    ("Bland. Needed a lot more seasoning.", {("TASTE", N)}),
    ("Very time-consuming, but worth it for the flavor.", {("TIME", N), ("TASTE", P)}),
    ("Ready in 20 minutes, which is perfect for busy nights.", {("TIME", P)}),
    ("Too many steps and too many dishes to wash.", {("PREP_EFFORT", N)}),
    ("Crispy on the outside and fluffy inside.", {("TEXTURE", P)}),
    ("My husband said it was too spicy, but I loved the heat.", {("TASTE", N), ("TASTE", P)}),
    ("Uses simple ingredients that I always keep in my pantry.", {("INGREDIENT_AVAILABILITY", P)}),
    ("I would not call this healthy with all that butter.", {("HEALTHINESS", N)}),
    ("It was a bit dry.", {("TEXTURE", N)}),
    ("The flavor was ok, nothing special.", {("TASTE", O)}),
    ("Easy to put together the night before.", {("PREP_EFFORT", P)}),
    ("Lighter than most versions and still very satisfying.", {("HEALTHINESS", P)}),
    ("Rich, decadent and totally worth the calories.", {("TASTE", P), ("HEALTHINESS", N)}),
    ("The sauce was watery and lacked flavor.", {("TEXTURE", N), ("TASTE", N)}),
    ("It took me all afternoon.", {("TIME", N)}),
    ("I had to go to three stores to find sumac.", {("INGREDIENT_AVAILABILITY", N)}),
    ("Melted in your mouth.", {("TEXTURE", P)}),
    ("Perfect for a potluck.", {("SUITABILITY", P)}),
    ("I halved the recipe and it worked fine.", set()),
    ("So simple, even my teenager can make it.", {("PREP_EFFORT", P)}),
    ("Way too sweet for my taste.", {("TASTE", N)}),
    ("The crust was soggy but the filling tasted amazing.", {("TEXTURE", N), ("TASTE", P)}),
    ("Low fat and you would never know it.", {("HEALTHINESS", P)}),
    ("Not something I'd call quick.", {("TIME", N)}),
    ("The prep was a pain, but the result was delicious.", {("PREP_EFFORT", N), ("TASTE", P)}),
    ("Chewy, gooey and just sweet enough.", {("TEXTURE", P), ("TASTE", P)}),
    ("We'll make it again.", set()),
    ("It's a guilt-free treat my whole family enjoys.", {("HEALTHINESS", P)}),
    ("The meat came out tough.", {("TEXTURE", N)}),
    ("Pretty fast once the vegetables are chopped, though the chopping takes a while.", {("TIME", P), ("TIME", N)}),
    ("I love spicy food, but this was too hot for my children.", {("TASTE", P), ("TASTE", N)}),
    ("Great for weekend dinners, but too time-consuming for weekdays.", {("SUITABILITY", P), ("TIME", N)}),
    ("It was a big hit at our Christmas party.", {("SUITABILITY", P)}),
    ("Not good for a weeknight, too many steps.", {("SUITABILITY", N), ("PREP_EFFORT", N)}),
    ("I made it for Christmas.", set()),
]


def predict_reviews(texts):
    """Per review: {(aspect, sentiment)} for both methods, from its Task 1 clauses."""
    clauses, owner = [], []
    for i, text in enumerate(texts):
        for clause in split_into_sentences(text):
            clauses.append(clause)
            owner.append(i)
    mentions = analyse(clauses)
    preds = {"lexicon": [set() for _ in texts], "ABSA model": [set() for _ in texts]}
    for m in mentions:
        review = owner[m["clause_pos"]]
        preds["lexicon"][review].add((m["aspect"], m["lexicon_sentiment"]))
        preds["ABSA model"][review].add((m["aspect"], m["model_sentiment"]))
    return preds


def prf(gold_sets, pred_sets):
    tp = sum(len(g & p) for g, p in zip(gold_sets, pred_sets))
    fp = sum(len(p - g) for g, p in zip(gold_sets, pred_sets))
    fn = sum(len(g - p) for g, p in zip(gold_sets, pred_sets))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return prec, rec, (2 * prec * rec / (prec + rec) if prec + rec else 0.0)


gold_texts = [t for t, _ in GOLD]
gold_sets = [g for _, g in GOLD]
gold_preds = predict_reviews(gold_texts)

rows = []
for method, preds in gold_preds.items():
    aspect_only = prf([{a for a, _ in g} for g in gold_sets], [{a for a, _ in p} for p in preds])
    pairs = prf(gold_sets, preds)
    # Sentiment on aspects both gold and prediction have with a single sentiment (macro-F1 over the three classes).
    y_true, y_pred = [], []
    for g, p in zip(gold_sets, preds):
        g_by, p_by = {}, {}
        for a, s in g:
            g_by.setdefault(a, set()).add(s)
        for a, s in p:
            p_by.setdefault(a, set()).add(s)
        for a in g_by.keys() & p_by.keys():
            if len(g_by[a]) == 1 and len(p_by[a]) == 1:
                y_true.append(next(iter(g_by[a])))
                y_pred.append(next(iter(p_by[a])))
    rows.append({
        "method": method, "aspect P": aspect_only[0], "aspect R": aspect_only[1], "aspect F1": aspect_only[2],
        "aspect+sentiment P": pairs[0], "aspect+sentiment R": pairs[1], "aspect+sentiment F1": pairs[2],
        "sentiment macro-F1": f1_score(y_true, y_pred, labels=SENTIMENTS, average="macro", zero_division=0),
        "sentiment n": len(y_true),
    })
gold_results = pd.DataFrame(rows).set_index("method")
print(f"\n--- Gold evaluation on {len(GOLD)} hand-written reviews ---")
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(gold_results.round(3))

per_aspect = pd.DataFrame({
    method: {a: prf([{x for x in g if x[0] == a} for g in gold_sets], [{x for x in p if x[0] == a} for p in preds])[2]
             for a in ASPECTS}
    for method, preds in gold_preds.items()
})
print("\n--- Aspect+sentiment F1 per aspect (gold) ---")
print(per_aspect.round(3))

print("\n--- Gold errors ---")
for text, g, lex, mod in zip(gold_texts, gold_sets, gold_preds["lexicon"], gold_preds["ABSA model"]):
    if g != lex or g != mod:
        print(f"{text}\n   gold   {sorted(g)}\n   lexicon {sorted(lex)}\n   model  {sorted(mod)}")

# ==========================================
# 4. REAL REVIEWS
# ==========================================
review_rows = df_clauses["review_row"].drop_duplicates().sample(min(N_REVIEWS, df_clauses["review_row"].nunique()),
                                                                 random_state=SEED)
df_apply = df_clauses[df_clauses["review_row"].isin(review_rows)].reset_index(drop=True)
apply_texts = df_apply["sentences"].astype(str).tolist()
mentions = analyse(apply_texts)

id_cols = [c for c in ("review_row", "review_id", "member_id", "recipe_id", "rating", "clause_idx") if c in df_apply.columns]
df_aspects = pd.DataFrame([
    {**df_apply.loc[m["clause_pos"], id_cols].to_dict(), "clause": apply_texts[m["clause_pos"]], "aspect": m["aspect"],
     "cue": m["cue"], "start": m["start"], "end": m["end"], "lexicon_sentiment": m["lexicon_sentiment"],
     "model_sentiment": m["model_sentiment"], "model_score": round(m["model_score"], 4)}
    for m in mentions
], columns=id_cols + ["clause", "aspect", "cue", "start", "end", "lexicon_sentiment", "model_sentiment", "model_score"])

n_reviews = df_apply["review_row"].nunique()
print(f"\n--- {len(df_aspects)} aspect mentions in {len(df_apply)} clauses from {n_reviews} reviews ---")
dist = pd.crosstab(df_aspects["aspect"], df_aspects["model_sentiment"]).reindex(index=ASPECTS, columns=SENTIMENTS).fillna(0).astype(int)
dist["reviews with aspect (%)"] = (df_aspects.groupby("aspect")["review_row"].nunique().reindex(ASPECTS).fillna(0) / n_reviews * 100).round(1)
print(dist)

dist[SENTIMENTS].plot(kind="barh", stacked=True, figsize=(10, 4.5), color=["#4c9a2a", "#b0b0b0", "#c0392b"],
                      title="Aspect sentiment in reviews (ABSA model)")
plt.xlabel("Mentions")
plt.tight_layout()
plt.show()

# Several sentiments in one review: the reason sentiment is kept per aspect instead of per review.
polar = df_aspects[df_aspects["model_sentiment"] != "neutral"]
per_review = polar.groupby("review_row")["model_sentiment"].agg(set)
mixed = per_review[per_review.map(len) == 2]
print(f"\nReviews with at least one polar aspect: {len(per_review)} ({len(per_review) / n_reviews:.1%} of reviews)")
print(f"... with both positive and negative aspects: {len(mixed)} ({len(mixed) / max(len(per_review), 1):.1%} of those)")
print("\n--- Examples of mixed reviews ---")
for review_row in mixed.index[:5]:
    print(" | ".join(df_apply.loc[df_apply["review_row"] == review_row, "sentences"]))
    for r in polar[polar["review_row"] == review_row].itertuples():
        print(f"     {r.aspect:<24} {r.model_sentiment:<9} evidence: {r.clause[r.start:r.end]!r} in {r.clause!r}")

# Agreement between the two methods on the same mentions.
print("\n--- Lexicon vs ABSA model agreement per aspect ---")
agree = df_aspects.groupby("aspect").apply(lambda g: pd.Series({
    "mentions": len(g), "agreement": (g["lexicon_sentiment"] == g["model_sentiment"]).mean(),
    "kappa": cohen_kappa_score(g["lexicon_sentiment"], g["model_sentiment"]) if g["lexicon_sentiment"].nunique() > 1 else np.nan,
}), include_groups=False).reindex(ASPECTS)
print(agree.round(3))

# Sanity check against star ratings (explicit 1-5 only): positive aspects should come from higher-rated reviews.
if "rating" in df_aspects.columns:
    rated = df_aspects[df_aspects["rating"].isin([1, 2, 3, 4, 5])]
    rating_table = rated.pivot_table(index="aspect", columns="model_sentiment", values="rating", aggfunc="mean")
    print("\n--- Mean star rating by aspect sentiment (ABSA model) ---")
    print(rating_table.reindex(index=ASPECTS, columns=SENTIMENTS).round(2))

# ==========================================
# 5. PERCEIVED HEALTHINESS VS NUTRITION (kept separate)
# ==========================================
# HUMMUS scores are 0-1, higher = healthier. Perceived healthiness is what the reviewer says; the scores are measured
# on the recipe. They are compared here and stored in separate columns, never combined.
health_cols = [c for c in ("who_score", "fsa_score", "nutri_score") if c in df_recipes.columns]
health = df_aspects[df_aspects["aspect"] == "HEALTHINESS"]
if health_cols and len(health):
    health = health.merge(df_recipes[["recipe_id"] + health_cols].drop_duplicates("recipe_id"), on="recipe_id", how="left")
    print("\n--- Recipe nutrition scores by perceived healthiness (separate signals) ---")
    print(health.groupby("model_sentiment")[health_cols].agg(["mean", "count"]).reindex(SENTIMENTS).round(3))
    polarity = health["model_sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
    for col in health_cols:
        print(f"Spearman(perceived healthiness, {col}) = {polarity.corr(health[col], method='spearman'):.3f}")

# ==========================================
# 6. OUTPUTS
# ==========================================
df_aspects.to_csv(os.path.join(OUT_DIR, "task4_aspect_sentiments.csv"), index=False)

# Same 200 pilot reviews as the Task 2 annotation sheet, so one annotation pass covers entities and aspects.
pilot_rows = sorted(df_clauses["review_row"].drop_duplicates().sample(200, random_state=SEED))
pilot = df_clauses[df_clauses["review_row"].isin(pilot_rows)].reset_index(drop=True)
pilot_mentions = analyse(pilot["sentences"].astype(str).tolist())
pilot["predicted_aspects"] = [
    json.dumps([[m["start"], m["end"], m["aspect"], m["model_sentiment"]] for m in pilot_mentions if m["clause_pos"] == i])
    for i in range(len(pilot))
]
pilot["corrected_aspects"] = ""  # fill in by hand: [[start, end, ASPECT, sentiment], ...]
pilot.to_csv(os.path.join(OUT_DIR, "task4_pilot_annotation.csv"), index=False)
print(f"\nSaved {len(df_aspects)} aspect mentions and a {len(pilot)}-clause pilot sheet in {OUT_DIR}.")
