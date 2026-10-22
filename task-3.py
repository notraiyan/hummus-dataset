!pip install -q bertopic gensim

# =====================================================================
# Task 3 - Topic modelling of cooking experiences: LDA vs NMF vs BERTopic
# Needs df_reviews and df_recipes loaded with kagglehub (see loading cell).
# Kaggle: Internet ON (pip + embedding model), GPU recommended for BERTopic.
# =====================================================================
import re, ast, html
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans

N_DOCS, K, SEED = 50_000, 15, 42      # same sample and same K for all three models
OUT_DIR = "/kaggle/working"
RUN = "experience_vocab"              # tag for output files, so the baseline run is not overwritten

# ---------------------------------------------------------------------
# 1. Corpus: one random sample of reviews, shared by all models
# ---------------------------------------------------------------------
def clean(text):
    t = html.unescape(str(text)).lower()
    t = re.sub(r"<[^>]+>|https?://\S+", " ", t)
    t = re.sub(r"(?<=\w)-(?=\w)", " ", t).replace("'", "")       # time-consuming -> time consuming, didn't -> didnt
    t = re.sub(r"\b(?:made|reviewed|tagged)\s+(?:this\s+|it\s+)?for\s+[^.!?]{0,40}?\b(?:tag|zwt|pac)\b[^.!?]*", " ", t)  # Food.com game tags
    return re.sub(r"\s+", " ", t).strip()

pool = df_reviews[df_reviews["text"].notna()].sample(min(2 * N_DOCS, df_reviews["text"].notna().sum()), random_state=SEED)
pool = pool.assign(clean=pool["text"].map(clean))
corpus = pool[pool["clean"].str.split().str.len() >= 8].head(N_DOCS).reset_index(drop=True)   # very short reviews carry no theme
docs = corpus["clean"].tolist()
print(f"{len(docs):,} reviews for topic modelling")

# ---------------------------------------------------------------------
# 2. Shared vocabulary + evaluation helpers
# ---------------------------------------------------------------------
# Praise and recipe boilerplate appear in every review and drown the themes
DOMAIN_STOP = {"recipe", "recipes", "make", "made", "making", "great", "good", "delicious", "yummy", "tasty",
               "wonderful", "excellent", "perfect", "nice", "love", "loved", "really", "just", "like", "liked",
               "will", "definitely", "thanks", "thank", "posting", "sharing", "stars", "star", "zwt", "tag",
               "dont", "didnt", "doesnt", "wasnt", "isnt", "wont", "cant", "wouldnt", "im", "ive", "id", "its"}
# Experience-focused vocabulary: drop ingredient, dish and unit words so topics describe
# the cooking experience rather than which dish was cooked
DOMAIN_STOP |= {"used", "use", "added", "add", "did", "does", "got", "came",
                "tsp", "tbsp", "teaspoon", "teaspoons", "tablespoon", "tablespoons", "cup", "cups",
                "oz", "ounce", "ounces", "lb", "lbs", "pound", "pounds", "inch", "can", "package"}
PROTECT = {  # context words that also occur in ingredient names - never remove these
    "quick", "fast", "minutes", "hour", "hours", "time", "weeknight", "weekend", "busy", "easy", "simple",
    "family", "kids", "kid", "children", "husband", "wife", "son", "daughter", "toddler", "picky", "mom", "grandma", "old",
    "prep", "ahead", "freeze", "freezer", "leftovers", "batch", "double", "half", "lunch", "lunches",
    "breakfast", "brunch", "dinner", "supper", "snack", "dessert", "appetizer",
    "party", "christmas", "thanksgiving", "easter", "holiday", "birthday", "guests", "potluck", "picnic", "company", "crowd",
    "special", "occasion", "summer", "winter", "spring", "fall", "new",
    "difficult", "hard", "complicated", "steps", "consuming", "work", "effort", "tricky",
    "crock", "pot", "slow", "cooker", "instant", "pressure", "grill", "oven", "microwave",
    "healthy", "light", "low", "fat", "free", "spicy", "hot", "mild", "sweet", "salty", "best", "favorite"}
DISHES = {"cake", "cakes", "cookie", "cookies", "pie", "pies", "bread", "muffin", "muffins", "soup", "salad", "casserole",
          "dip", "pasta", "pizza", "brownie", "brownies", "lasagna", "cheesecake", "pudding", "bars", "stew", "chili",
          "sandwich", "sandwiches", "burgers", "meatloaf", "meatballs", "pancakes", "waffles", "smoothie", "frosting",
          "icing", "crust", "dough", "loaf", "rolls", "biscuits", "quiche", "curry", "enchiladas", "tacos", "fudge"}
ing_tokens = Counter(w for s in df_recipes["ingredient_food_kg_names"].dropna().astype(str)
                     for w in set(re.findall(r"[a-z]{3,}", s.lower())) - {"nan"})
FOOD_STOP = ({w for w, _ in ing_tokens.most_common(1500)} | DISHES) - PROTECT
print(f"{len(FOOD_STOP):,} ingredient/dish words removed")

VEC_ARGS = dict(stop_words=sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP | FOOD_STOP), ngram_range=(1, 2),
                min_df=20, max_df=0.5, token_pattern=r"(?u)\b[a-z][a-z]+\b")

count_vec = CountVectorizer(**VEC_ARGS)
X_counts = count_vec.fit_transform(docs)                  # LDA input (raw counts)
tfidf_vec = TfidfVectorizer(**VEC_ARGS, sublinear_tf=True)
X_tfidf = tfidf_vec.fit_transform(docs)                   # NMF input (tf-idf)
print("vocabulary:", X_counts.shape[1])

# Coherence is computed on the same tokens the models see (unigrams + bigrams)
analyzer = count_vec.build_analyzer()
tokenized = [analyzer(d) for d in docs]
dictionary = Dictionary(tokenized)

THEMES = {  # seed words used only to check whether the expected themes were found
    "quick meals":        {"quick", "fast", "minutes", "weeknight", "busy", "easy", "simple", "time"},
    "family cooking":     {"family", "kids", "children", "husband", "wife", "son", "daughter", "picky"},
    "meal preparation":   {"prep", "ahead", "freeze", "freezer", "leftovers", "batch", "lunch", "lunches"},
    "celebrations":       {"party", "christmas", "thanksgiving", "holiday", "birthday", "guests", "potluck", "company"},
    "cooking difficulty": {"difficult", "hard", "complicated", "steps", "consuming", "work", "effort", "tricky"},
}

def top_words(components, vocab, n=25):
    return {k: [vocab[i] for i in row.argsort()[::-1][:n]] for k, row in enumerate(components)}

def coherence(topics, measure, topn=10):
    tops = [[w for w in words if w in dictionary.token2id][:topn] for words in topics.values()]
    return CoherenceModel(topics=tops, texts=tokenized, dictionary=dictionary, coherence=measure, topn=topn).get_coherence()

def diversity(topics, n=25):                                # share of unique words among all topics' top-n (Dieng et al., 2020)
    return len({w for words in topics.values() for w in words[:n]}) / (len(topics) * n)

def theme_match(topics, n=25):
    rows = []
    for theme, seeds in THEMES.items():
        best = max(topics, key=lambda k: len(set(topics[k][:n]) & seeds))
        hits = sorted(set(topics[best][:n]) & seeds)
        rows.append({"theme": theme, "best_topic": best if hits else None, "matched_words": ", ".join(hits)})
    return pd.DataFrame(rows)

def show(topics, prevalence, n=8):
    for k, words in topics.items():
        print(f"{k:>3} ({prevalence.get(k, 0):5.1%}): {', '.join(words[:n])}")

def exemplars(doc_topic, n=3):                              # most representative original reviews per topic
    return {k: corpus.loc[np.argsort(doc_topic[:, k])[::-1][:n], "text"].map(html.unescape).str.replace(r"\s+", " ", regex=True).str.slice(0, 300).tolist()
            for k in range(doc_topic.shape[1])}

RESULTS = {}

# ---------------------------------------------------------------------
# 3. LDA
# ---------------------------------------------------------------------
lda = LatentDirichletAllocation(n_components=K, learning_method="online", batch_size=2048,
                                max_iter=10, random_state=SEED, n_jobs=-1)
W_lda = lda.fit_transform(X_counts)
RESULTS["LDA"] = {"topics": top_words(lda.components_, count_vec.get_feature_names_out()),
                  "prevalence": pd.Series(W_lda.argmax(1)).value_counts(normalize=True).to_dict(),
                  "examples": exemplars(W_lda)}
print("\n--- LDA ---"); show(RESULTS["LDA"]["topics"], RESULTS["LDA"]["prevalence"])

# ---------------------------------------------------------------------
# 4. NMF
# ---------------------------------------------------------------------
nmf = NMF(n_components=K, init="nndsvda", beta_loss="kullback-leibler", solver="mu",
          max_iter=400, random_state=SEED)
W_nmf = nmf.fit_transform(X_tfidf)
RESULTS["NMF"] = {"topics": top_words(nmf.components_, tfidf_vec.get_feature_names_out()),
                  "prevalence": pd.Series(W_nmf.argmax(1)).value_counts(normalize=True).to_dict(),
                  "examples": exemplars(W_nmf)}
print("\n--- NMF ---"); show(RESULTS["NMF"]["topics"], RESULTS["NMF"]["prevalence"])

# ---------------------------------------------------------------------
# 5. BERTopic
# ---------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(corpus["text"].tolist(), batch_size=256, show_progress_bar=True)  # embed the original text

bertopic = BERTopic(
    embedding_model=embedder,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
    hdbscan_model=KMeans(n_clusters=K, n_init=10, random_state=SEED),   # exactly K topics, every review assigned (like LDA/NMF)
    vectorizer_model=CountVectorizer(**{k: v for k, v in VEC_ARGS.items() if k not in ("min_df", "max_df")}),  # same tokens; df limits don't apply to topic-level docs
    ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
    top_n_words=25,
    verbose=True,
)
bt_labels, _ = bertopic.fit_transform(docs, embeddings)
bt_ids = [t for t in bertopic.get_topic_info()["Topic"] if t != -1]
RESULTS["BERTopic"] = {"topics": {t: [w for w, _ in bertopic.get_topic(t)] for t in bt_ids},
                       "prevalence": pd.Series(bt_labels).value_counts(normalize=True).to_dict(),
                       "examples": {t: [d[:300] for d in bertopic.get_representative_docs(t)] for t in bt_ids}}
print("\n--- BERTopic ---")
show(RESULTS["BERTopic"]["topics"], RESULTS["BERTopic"]["prevalence"])

# ---------------------------------------------------------------------
# 6. Compare: coherence, diversity, expected themes
# ---------------------------------------------------------------------
scores = pd.DataFrame([{"model": name,
                        "topics": len(r["topics"]),
                        "coherence_c_v": coherence(r["topics"], "c_v"),
                        "coherence_npmi": coherence(r["topics"], "c_npmi"),
                        "diversity@25": diversity(r["topics"])} for name, r in RESULTS.items()]).round(3)
display(scores)
scores.to_csv(f"{OUT_DIR}/topic_scores_{RUN}.csv", index=False)

themes = pd.concat([theme_match(r["topics"]).assign(model=name) for name, r in RESULTS.items()])
display(themes.pivot(index="theme", columns="model", values="matched_words"))

# ---------------------------------------------------------------------
# 7. Inspect representative reviews (quick human check)
# ---------------------------------------------------------------------
def inspect(name, n_topics=None, n_docs=2, width=200):
    r = RESULTS[name]
    for k in list(r["topics"])[:n_topics]:
        print(f"\n[{name}] topic {k} ({r['prevalence'].get(k, 0):.1%}): {', '.join(r['topics'][k][:8])}")
        for ex in r["examples"][k][:n_docs]:
            print("   -", ex[:width])

for name in RESULTS:
    inspect(name)

# ---------------------------------------------------------------------
# 8. Export for human rating (two raters: label + interpretability 1-3)
# ---------------------------------------------------------------------
rows = [{"model": name, "topic": k, "share": round(r["prevalence"].get(k, 0), 4),
         "top_words": ", ".join(words[:10]), "example_reviews": " || ".join(r["examples"][k]),
         "label": "", "interpretable_1to3": ""}
        for name, r in RESULTS.items() for k, words in r["topics"].items()]
pd.DataFrame(rows).to_csv(f"{OUT_DIR}/topics_for_human_rating_{RUN}.csv", index=False)
print(f"\n{len(rows)} topics exported to {OUT_DIR}/topics_for_human_rating_{RUN}.csv")

# ---------------------------------------------------------------------
# 9. Contrast: topics of recipe directions (+ content features for Task 8)
# ---------------------------------------------------------------------
def directions_text(s):
    try:
        steps = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return ""
    return " ".join(x for x in steps if isinstance(x, str)).lower() if isinstance(steps, list) else ""

rec = df_recipes[["recipe_id", "directions"]].dropna().drop_duplicates("recipe_id")
rec = rec.sample(min(N_DOCS, len(rec)), random_state=SEED).reset_index(drop=True)
rec_vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=20, max_df=0.5,
                          token_pattern=r"(?u)\b[a-z][a-z]+\b", sublinear_tf=True)
X_rec = rec_vec.fit_transform(rec["directions"].map(directions_text))
rec_nmf = NMF(n_components=K, init="nndsvda", max_iter=400, random_state=SEED)
W_rec = rec_nmf.fit_transform(X_rec)
rec_topics = top_words(rec_nmf.components_, rec_vec.get_feature_names_out())

print("\n--- Recipe-direction topics (NMF) ---")
show(rec_topics, pd.Series(W_rec.argmax(1)).value_counts(normalize=True).to_dict())
display(theme_match(rec_topics))           # experience themes should be largely absent here

(pd.DataFrame(W_rec, columns=[f"rtopic_{k}" for k in range(K)])
   .assign(recipe_id=rec["recipe_id"].to_numpy())
   .to_parquet(f"{OUT_DIR}/recipe_direction_topics.parquet", index=False))
print(f"Per-recipe topic mix saved to {OUT_DIR}/recipe_direction_topics.parquet")