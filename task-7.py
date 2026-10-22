# Imports already made by task-1.py to task-6.py (same notebook): json, os, np, pd.

# Task 7: Contextual User Profile Construction
# Run after task-6.py in the same notebook (uses df_reviews and DATE_COL from Task 1, rules_label from Task 5 and
# preferences_for from Task 6).
# Profiles per member, built from the preference tuples of Tasks 4-6:
#   general profile:  the reviewer's own opinions, with no occasion or condition
#   context profiles: opinions held by someone else (children, partner, family, guests), or tied to an occasion
#                     (weekday, weekend, holiday, gathering, special occasion) or to a condition ("if ...",
#                     "except ..."). They are never merged into the general profile.
# Opinions whose holder is unknown are left out. A preference is "stable" only when it holds across at least two
# recipes; otherwise it stays "tentative", since a single comment ("too salty") is usually about that one recipe.
# Every profile entry keeps its evidence spans and an extraction confidence.
# Task 4 sampled random reviews, so most members appear only once; this task runs the same extraction on the reviews
# of members with a longer history.

for required in ("df_reviews", "DATE_COL", "rules_label", "preferences_for"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-6.py in this session first.")

SEED = 42
N_MEMBERS = 200                  # members profiled
MIN_MEMBER_REVIEWS = 10          # members with at least this many reviews with text
MAX_REVIEWS_PER_MEMBER = 30      # their most recent reviews
MIN_RECIPES_STABLE = 2           # a preference is stable when it holds across at least this many recipes
NEGATION_RULE_CONFIDENCE = 0.9   # sentiments set by Task 6's negation rule (about 19 of 20 sampled reversals were right)
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ==========================================
# 1. PREFERENCE TUPLES FOR MEMBERS WITH A HISTORY
# ==========================================
review_counts = df_reviews["member_id"].value_counts()
eligible = review_counts[review_counts >= MIN_MEMBER_REVIEWS]
members = eligible.sample(min(N_MEMBERS, len(eligible)), random_state=SEED).index
member_reviews = (df_reviews[df_reviews["member_id"].isin(members)]
                  .sort_values(DATE_COL, ascending=False).groupby("member_id").head(MAX_REVIEWS_PER_MEMBER)
                  .reset_index(drop=True))


def extract(reviews):
    """Preference tuples (Tasks 4-6) for a DataFrame of reviews with text, member_id, recipe_id and review_id (and the
    date, kept for chronological splits)."""
    prefs = preferences_for(reviews["text"].tolist())
    prefs["holder"], prefs["context"] = rules_label(prefs)
    prefs = prefs.join(reviews[[c for c in ("member_id", "recipe_id", "review_id", DATE_COL) if c in reviews]], on="review")
    # Extraction confidence: the negation rule's precision where Task 6's negation rule set the sentiment, else Task 4's
    # model probability (low where Task 6 overrode an unsure model).
    prefs["confidence"] = np.where((prefs["sentiment"] != prefs["model_sentiment"]) & prefs["negated"],
                                   NEGATION_RULE_CONFIDENCE, prefs["model_score"])
    prefs["scope"] = np.where(prefs["condition_type"] != "", "conditional", prefs["context"])
    return prefs


prefs = extract(member_reviews)
print(f"{len(prefs)} preference tuples from {len(member_reviews)} reviews of {len(members)} members")

# ==========================================
# 2. PROFILES
# ==========================================
SENTIMENT_OF = {"likes": "positive", "dislikes": "negative"}


def build_profiles(prefs):
    """One entry per member, holder, scope and aspect; tuples with an unknown holder are left out. Rebuild it from any
    subset of tuples, e.g. only reviews before a cut-off date for a chronological evaluation (Task 8)."""
    entries = []
    known = prefs[prefs["holder"] != "unknown"]
    for (member, holder, scope, aspect), g in known.groupby(["member_id", "holder", "scope", "aspect"]):
        pos, neg = int((g["sentiment"] == "positive").sum()), int((g["sentiment"] == "negative").sum())
        preference = "likes" if pos > neg else "dislikes" if neg > pos else "mixed" if pos else "neutral"
        support = g[g["sentiment"] == SENTIMENT_OF[preference]] if preference in SENTIMENT_OF else g
        recipes = support["recipe_id"].nunique()
        consistency = max(pos, neg) / (pos + neg) if pos + neg else 0.0
        stable = preference in SENTIMENT_OF and recipes >= MIN_RECIPES_STABLE and consistency >= 2 / 3
        evidence = [{"review_id": None if pd.isna(r.review_id) else int(r.review_id), "clause": r.clause,
                     "start": int(r.start), "end": int(r.end), "sentiment": r.sentiment, "condition": r.condition,
                     "confidence": round(float(r.confidence), 3)}
                    for r in support.nlargest(3, "confidence").itertuples()]
        entries.append({
            "member_id": member, "profile": "general" if (holder, scope) == ("reviewer", "general") else "context",
            "holder": holder, "scope": scope, "aspect": aspect, "preference": preference,
            "status": "stable" if stable else "tentative", "positive": pos, "negative": neg, "mentions": len(g),
            "recipes": recipes, "consistency": round(consistency, 2), "confidence": round(support["confidence"].mean(), 3),
            "cues": ", ".join(support["cue"].str.lower().value_counts().index[:5]), "evidence": json.dumps(evidence),
        })
    return pd.DataFrame(entries)


def show(entries):
    for e in entries.itertuples():
        first = json.loads(e.evidence)[0]
        print(f"   {e.holder:<9} {e.scope:<17} {e.aspect:<24} {e.preference:<8} {e.status:<9} recipes={e.recipes} "
              f"conf={e.confidence:.2f}  e.g. {first['clause'][:70]!r}")


profiles = build_profiles(prefs)
general = profiles[profiles["profile"] == "general"]
context = profiles[profiles["profile"] == "context"]
print(f"\n--- {len(profiles)} profile entries for {profiles['member_id'].nunique()} members ---")
print(pd.crosstab([profiles["profile"], profiles["holder"]], profiles["status"], margins=True))
print("\nContext profiles by scope:", context["scope"].value_counts().to_dict())

# ==========================================
# 3. CHECKS
# ==========================================
# What a profile that ignores holders and contexts would claim about the reviewer.
naive = build_profiles(prefs.assign(holder="reviewer", scope="general"))
compared = naive.merge(general, on=["member_id", "aspect"], how="left", suffixes=("_naive", ""))
claimed = compared[(compared["status_naive"] == "stable") &
                   ((compared["status"] != "stable") | (compared["preference"] != compared["preference_naive"]))]
kept_out = prefs[(prefs["holder"] != "reviewer") | (prefs["scope"] != "general")]
print(f"\nTuples kept out of the general profiles: {len(kept_out)} of {len(prefs)} "
      f"(holders: {kept_out.loc[kept_out['holder'] != 'reviewer', 'holder'].value_counts().to_dict()}; "
      f"scopes: {kept_out.loc[kept_out['scope'] != 'general', 'scope'].value_counts().to_dict()})")
print(f"A holder- and context-blind profile would claim {len(claimed)} stable general preferences "
      f"that the members' own general opinions do not support, e.g.:")
for c in claimed.head(3).itertuples():
    others = prefs[(prefs["member_id"] == c.member_id) & (prefs["aspect"] == c.aspect) &
                   ((prefs["holder"] != "reviewer") | (prefs["scope"] != "general"))]
    r = others.iloc[0]
    print(f"   member {c.member_id}: blind profile '{c.preference_naive} {c.aspect}' (stable), own general profile "
          f"'{c.preference if isinstance(c.preference, str) else 'none'}'; from ({r.holder}, {r.scope}): {r.clause[:70]!r}")

stable_share = (general["status"] == "stable").mean()
print(f"\nGeneral entries that are stable (hold across >= {MIN_RECIPES_STABLE} recipes): {stable_share:.1%}; "
      f"the rest come from a single recipe and stay tentative.")

richest = general.assign(stable=general["status"] == "stable").groupby("member_id")["stable"].sum().idxmax()
print(f"\n--- Profile of member {richest} ---")
print(" general:")
show(general[general["member_id"] == richest].sort_values(["status", "recipes"], ascending=[True, False]))
print(" context:")
show(context[context["member_id"] == richest])

# The project's motivating example as a one-review member.
EXAMPLE = ("I love spicy food, but this was too hot for my children. "
           "Great for weekend dinners, but too time-consuming for weekdays.")
example_profile = build_profiles(extract(pd.DataFrame({"text": [EXAMPLE], "member_id": ["example"], "recipe_id": [0],
                                                       "review_id": [0]})))
print("\n--- Motivating example ---")
show(example_profile)

# ==========================================
# 4. OUTPUT
# ==========================================
profiles.to_csv(os.path.join(OUT_DIR, "task7_profiles.csv"), index=False)
tuple_cols = ["member_id", "recipe_id", "review_id", DATE_COL, "holder", "aspect", "sentiment", "scope", "condition",
              "negated", "confidence", "clause", "cue", "start", "end"]
prefs[tuple_cols].to_csv(os.path.join(OUT_DIR, "task7_profile_tuples.csv"), index=False)
print(f"\nSaved {len(profiles)} profile entries and {len(prefs)} tuples with evidence in {OUT_DIR}.")
