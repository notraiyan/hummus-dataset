# Imports already made by task-1.py to task-7.py (same notebook): os, re, np, pd, TfidfVectorizer, SentenceTransformer.
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# Task 8: Context-Aware Recipe Retrieval and Recommendation
# Run after task-7.py in the same notebook (uses df_reviews and DATE_COL from Task 1, df_recipes, HOLDER_OF,
# FIRST_PERSON and rule_context from Task 5, and extract from Task 7).
# A request such as "Recommend a quick dinner for my children." is parsed into a holder (children), an occasion, and
# checkable constraints on structured recipe attributes ("quick": duration <= 30 minutes; "dinner": tagged main-dish).
# Recipes that pass the constraints are ranked by the sum of three standardised scores:
#   content: TF-IDF similarity between the request and the recipe's title and tags
#   CF:      collaborative filtering (truncated SVD of which members liked which recipes)
#   profile: similarity to recipes the relevant contextual profile liked, minus those it disliked (Task 7 facts)
# Evaluation is chronological: everything is built from reviews before a cut-off date, and each evaluated member's
# first liked recipe after it is ranked among recipes they never reviewed, sampled uniformly and in proportion to
# popularity (NDCG@10, Recall@10). Nothing from the held-out review is used, so the request is context-free; contextual
# matching is evaluated in Task 10. Baselines: popularity, CF, review embeddings, and the same profile built without
# holders and contexts.
# A GPU makes the sentiment model and the embeddings fast; on CPU, lower N_EVAL_MEMBERS.

for required in ("df_reviews", "DATE_COL", "df_recipes", "HOLDER_OF", "FIRST_PERSON", "rule_context", "extract"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-7.py in this session first.")

SEED = 42
TRAIN_SHARE = 0.9          # reviews before the 90th-percentile date are history; later ones are the test period
MIN_RECIPE_REVIEWS = 5     # catalog: recipes with at least this many reviews before the cut-off
N_EVAL_MEMBERS = 300
MIN_TRAIN_REVIEWS = 5      # evaluated members need this many reviews before the cut-off
MAX_TRAIN_REVIEWS = 30     # the most recent of them feed the member's profile, as in Task 7
N_NEGATIVES = 49           # the held-out recipe is ranked among this many recipes the member never reviewed
K = 10
QUICK_MINUTES = 30
REVIEW_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
rng = np.random.default_rng(SEED)

# ==========================================
# 1. RECIPES: STRUCTURED ATTRIBUTES AND CONTENT
# ==========================================
recipes = df_recipes[["recipe_id", "title", "tags", "duration", "direction_size"]].drop_duplicates("recipe_id")
recipes = recipes.set_index("recipe_id")
# Food.com's tag-category headers, carried by every tagged recipe; they say nothing about the recipe itself.
META_TAGS = {"time-to-make", "course", "main-ingredient", "preparation", "occasion", "cuisine", "dietary", "equipment",
             "number-of-servings", "taste-mood", "technique"}
recipes["tag_list"] = recipes["tags"].map(
    lambda t: [x for x in t.strip("[]").replace("'", "").split(", ") if x not in META_TAGS] if isinstance(t, str) else [])
tfidf = TfidfVectorizer(stop_words="english", min_df=5, sublinear_tf=True)
recipe_tfidf = tfidf.fit_transform(recipes["title"].fillna("") + " " + recipes["tag_list"].str.join(" "))
row_of = pd.Series(np.arange(len(recipes)), index=recipes.index)

# ==========================================
# 2. CHRONOLOGICAL SPLIT, POPULARITY AND COLLABORATIVE FILTERING
# ==========================================
cutoff = df_reviews[DATE_COL].quantile(TRAIN_SHARE)
train = df_reviews[df_reviews[DATE_COL] < cutoff]
test = df_reviews[df_reviews[DATE_COL] >= cutoff]
review_counts = train["recipe_id"].value_counts()
catalog = review_counts[(review_counts >= MIN_RECIPE_REVIEWS) & review_counts.index.isin(recipes.index)].index
liked = train[(train["rating"] >= 4) & train["recipe_id"].isin(catalog)]  # explicit 4-5 stars (2.5 means no rating)
popularity = liked["recipe_id"].value_counts().reindex(catalog, fill_value=0)

member_codes, item_codes = pd.Index(liked["member_id"].unique()), pd.Index(catalog)
cf_matrix = csr_matrix((np.ones(len(liked)), (member_codes.get_indexer(liked["member_id"]),
                                              item_codes.get_indexer(liked["recipe_id"]))),
                       shape=(len(member_codes), len(item_codes)))
svd = TruncatedSVD(n_components=64, random_state=SEED)
member_factors, item_factors = svd.fit_transform(cf_matrix), svd.components_.T
print(f"Cut-off {cutoff.date()}: {len(train)} history reviews, {len(test)} test reviews; catalog of {len(catalog)} "
      f"recipes; CF on {len(liked)} likes by {len(member_codes)} members")


def cf_scores(member, recipe_ids):
    m, items = member_codes.get_indexer([member])[0], item_codes.get_indexer(recipe_ids)
    if m < 0:
        return np.zeros(len(items))
    return np.where(items >= 0, item_factors[np.maximum(items, 0)] @ member_factors[m], 0.0)


# ==========================================
# 3. PROFILES FROM THE EXTRACTED FACTS
# ==========================================
def profile_vector(tuples):
    """Recipes the profile liked add, disliked ones subtract (weighted by extraction confidence), in TF-IDF space."""
    tuples = tuples[tuples["recipe_id"].isin(row_of.index)]
    weights = (tuples["sentiment"].map({"positive": 1.0, "negative": -1.0}).fillna(0.0) * tuples["confidence"]).to_numpy()
    if not weights.any():
        return None
    return normalize(csr_matrix(weights) @ recipe_tfidf[row_of[tuples["recipe_id"]].to_numpy()])


def contextual_tuples(tuples, holder, scope):
    """The profile relevant to a request (that holder and occasion), or the reviewer's general profile when it is empty.
    Opinions with an unknown holder are never used."""
    own = tuples[(tuples["holder"] == holder) & (tuples["scope"] == scope)]
    return own if len(own) else tuples[(tuples["holder"] == "reviewer") & (tuples["scope"] == "general")]


# ==========================================
# 4. REQUESTS AND SCORING
# ==========================================
CONTEXT_TERMS = {"children": "kid friendly", "guests": "party crowd", "weekday": "weeknight", "gathering": "party crowd",
                 "holiday": "holiday event christmas thanksgiving", "special occasion": "dinner party"}
MEAL_TAGS = {"dinner": "main-dish", "supper": "main-dish", "lunch": "lunch", "breakfast": "breakfast",
             "dessert": "desserts"}


def parse_request(text):
    """Holder, occasion, checkable constraints and content terms of a request."""
    lower = text.lower()
    words = re.findall(r"[a-z']+", lower)
    holder = next((HOLDER_OF[w] for w in words if w in HOLDER_OF and w not in FIRST_PERSON
                   and HOLDER_OF[w] != "unknown"), "reviewer")
    scope = rule_context(text)
    minutes = re.search(r"(?:under|in|less than)\s+(\d+)\s*min", lower)
    max_minutes = int(minutes.group(1)) if minutes else QUICK_MINUTES if re.search(r"\b(?:quick|fast|hurry)\b", lower) else None
    return {"holder": holder, "scope": "general" if scope == "unknown" else scope, "max_minutes": max_minutes,
            "meal_tag": next((MEAL_TAGS[w] for w in words if w in MEAL_TAGS), None),
            "terms": " ".join([text, CONTEXT_TERMS.get(holder, ""), CONTEXT_TERMS.get(scope, "")])}


def satisfies(recipe_ids, request):
    """Checkable constraints on structured attributes: duration in minutes and the meal tag."""
    r = recipes.loc[recipe_ids]
    ok = np.ones(len(r), dtype=bool)
    if request["max_minutes"]:
        ok &= r["duration"].fillna(np.inf).to_numpy() <= request["max_minutes"]
    if request["meal_tag"]:
        ok &= r["tag_list"].map(lambda tags: request["meal_tag"] in tags).to_numpy()
    return ok


def zscore(x):
    return (x - x.mean()) / x.std() if x.std() > 0 else np.zeros_like(x)


def score_candidates(member, recipe_ids, request, tuples, contextual=True):
    """Content, CF and profile scores for candidate recipes, and their standardised sum."""
    rows = row_of[recipe_ids].to_numpy()
    content = (recipe_tfidf[rows] @ tfidf.transform([request["terms"]]).T).toarray().ravel()
    vec = profile_vector(contextual_tuples(tuples, request["holder"], request["scope"]) if contextual else tuples)
    profile = (recipe_tfidf[rows] @ vec.T).toarray().ravel() if vec is not None else np.zeros(len(rows))
    cf = cf_scores(member, recipe_ids)
    return {"content": content, "CF": cf, "profile": profile, "total": zscore(content) + zscore(cf) + zscore(profile)}


# ==========================================
# 5. EVALUATION (chronological)
# ==========================================
history_counts = train["member_id"].value_counts()
first_liked = (test[(test["rating"] >= 4) & test["recipe_id"].isin(catalog)].sort_values(DATE_COL)
               .drop_duplicates("member_id"))
first_liked = first_liked[first_liked["member_id"].isin(history_counts[history_counts >= MIN_TRAIN_REVIEWS].index)]
seen_before = set(zip(train["member_id"], train["recipe_id"]))
first_liked = first_liked[[(m, r) not in seen_before for m, r in zip(first_liked["member_id"], first_liked["recipe_id"])]]
# Members are drawn at random: nothing about the held-out review (its text or context) is used before ranking.
targets = first_liked.sample(min(N_EVAL_MEMBERS, len(first_liked)), random_state=SEED).reset_index(drop=True)
history = (train[train["member_id"].isin(targets["member_id"])].sort_values(DATE_COL, ascending=False)
           .groupby("member_id").head(MAX_TRAIN_REVIEWS).reset_index(drop=True))
facts = extract(history)  # Tasks 4-6 on the history only

reviewed = df_reviews[df_reviews["member_id"].isin(targets["member_id"])].groupby("member_id")["recipe_id"].agg(set)
# Uniform negatives favour popularity (the held-out recipes are mostly popular ones); popularity-matched negatives
# test whether a method picks the right recipe among similarly popular ones.
SCHEMES = {"uniform": None, "popularity-matched": (popularity / popularity.sum()).to_numpy()}
candidates = {scheme: {} for scheme in SCHEMES}
for t in targets.itertuples():
    for scheme, p in SCHEMES.items():
        pool = rng.choice(catalog.to_numpy(), size=N_NEGATIVES * 3, replace=False, p=p)
        negatives = [r for r in pool if r not in reviewed[t.member_id] and r != t.recipe_id][:N_NEGATIVES]
        candidates[scheme][t.member_id] = [t.recipe_id] + negatives

# Review embeddings: a member is the mean of their history reviews; a recipe is up to three history reviews by others.
encoder = SentenceTransformer(REVIEW_EMBEDDING_MODEL)
history_emb = encoder.encode(history["text"].tolist(), batch_size=128, normalize_embeddings=True)
member_emb = {m: normalize(history_emb[idx].mean(axis=0, keepdims=True))[0]
              for m, idx in history.groupby("member_id").indices.items()}
needed = {r for by_member in candidates.values() for c in by_member.values() for r in c}
recipe_texts = (train[train["recipe_id"].isin(needed)].sort_values(DATE_COL).groupby("recipe_id")["text"]
                .apply(lambda texts: " ".join(texts.head(3))))
recipe_emb = dict(zip(recipe_texts.index, encoder.encode(recipe_texts.tolist(), batch_size=128,
                                                         normalize_embeddings=True)))

# Food.com does not record the request behind a review, so every held-out case gets the same context-free request;
# contextual matching is evaluated separately in Task 10 with authored requests and judged relevance.
GENERAL_REQUEST = {"holder": "reviewer", "scope": "general", "max_minutes": None, "meal_tag": None, "terms": ""}
rows = []
for t in targets.itertuples():
    member, request = t.member_id, GENERAL_REQUEST
    tuples = facts[facts["member_id"] == member]
    for scheme in SCHEMES:
        cands = candidates[scheme][member]
        aware = score_candidates(member, cands, request, tuples, contextual=True)
        blind = score_candidates(member, cands, request, tuples, contextual=False)
        scores = {
            "popularity": popularity[cands].to_numpy(dtype=float),
            "CF": aware["CF"],
            "review embeddings": np.array([recipe_emb[r] @ member_emb[member] if r in recipe_emb else 0.0 for r in cands]),
            "profile, context-independent": blind["profile"],
            "profile, contextual": aware["profile"],
            "combined, context-independent profile": blind["total"],
            "combined, contextual profile": aware["total"],
        }
        for method, s in scores.items():
            rank = 1 + (s[1:] > s[0]).sum() + 0.5 * (s[1:] == s[0]).sum()  # the held-out recipe is first; ties split
            rows.append({"member_id": member, "scheme": scheme, "method": method, "rank": rank,
                         "NDCG@10": 1 / np.log2(rank + 1) if rank <= K else 0.0, "Recall@10": float(rank <= K)})
evaluation = pd.DataFrame(rows)
print(f"\n--- Held-out liked recipe ranked among {N_NEGATIVES + 1} candidates, {len(targets)} members "
      f"(random Recall@10 = {K / (N_NEGATIVES + 1):.2f}) ---")
with pd.option_context("display.width", 200):
    print(evaluation.pivot_table(index="method", columns="scheme", values=["NDCG@10", "Recall@10"], aggfunc="mean")
          .reindex(evaluation["method"].unique()).round(3))

# ==========================================
# 6. A REQUEST
# ==========================================
REQUEST = "Recommend a quick dinner for my children."
with_kids = facts.loc[facts["holder"] == "children", "member_id"]
member = with_kids.value_counts().index[0] if len(with_kids) else targets["member_id"].iloc[0]
tuples = facts[facts["member_id"] == member]
request = parse_request(REQUEST)
pool = catalog[satisfies(catalog, request)]
pool = pool[~pool.isin(reviewed.get(member, set()))]
s = score_candidates(member, pool, request, tuples)
used = contextual_tuples(tuples, request["holder"], request["scope"])
print(f"\n--- {REQUEST!r} for member {member} ---")
print({k: v for k, v in request.items() if k != "terms"}, f"-> {len(pool)} recipes pass the constraints")
print(f"Profile used: {used['holder'].iloc[0]}/{used['scope'].iloc[0]} with {len(used)} opinions" if len(used)
      else "Profile used: none")
for o in used.head(5).itertuples():
    print(f"   ({o.sentiment} {o.aspect}) {recipes.loc[o.recipe_id, 'title'] if o.recipe_id in recipes.index else '?'}: "
          f"{o.clause[:70]!r}")
print("Top recommendations:")
for i in np.argsort(-s["total"])[:5]:
    r = recipes.loc[pool[i]]
    print(f"   {r['title'][:45]:<45} {r['duration']:>5.0f} min  content={s['content'][i]:.2f} CF={s['CF'][i]:.2f} "
          f"profile={s['profile'][i]:.2f}  kid-friendly={'kid-friendly' in r['tag_list']}")

evaluation.to_csv(os.path.join(OUT_DIR, "task8_evaluation.csv"), index=False)
print(f"\nSaved per-member ranks in {OUT_DIR}.")
