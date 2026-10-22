# Imports already made by task-1.py to task-5.py (same notebook): os, re, np, pd.

# Task 6: Negation and Conditional Preference Detection
# Run after task-5.py in the same notebook (uses df_tuples, mentions_for and nlp_dep from Task 5, CUE_INDEX and
# normalise from Task 4, and df_clauses and split_into_sentences from Task 1).
# Adds to each preference tuple:
#   negated, negator: the opinion word is inside a negation ("not too spicy", "I don't like spicy food",
#                     "I don't think it was bland"), and the word that negates it
#   contrast:         its clause starts with a contrast word ("..., but difficult"); Task 1 splits clauses there
#   condition:        the preference holds only under a condition ("quick if you use canned beans")
#   exception:        an exception reverses it in one situation ("I enjoy spicy food, except at breakfast"); it becomes
#                     an extra tuple with the opposite sentiment and that situation as its condition
# Sentiment: Task 4's model sentiment, except that a cue with its own polarity takes the opposite polarity when a
# negation reverses it ("not too spicy", "never bland"), and keeps its own polarity when the model is unsure. A cue
# without one ("spicy", "texture") keeps the model's reading of the verb ("I don't like spicy food").
# Task 4's model alone misses negated polar cues ("not too spicy"), see the challenge set. The negation flag also lets a
# profile tell "not too spicy" (fine at this level) from "likes spicy food".

for required in ("df_tuples", "mentions_for", "nlp_dep", "CUE_INDEX", "normalise", "df_clauses", "split_into_sentences"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-5.py in this session first.")

SEED = 42
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ==========================================
# 1. NEGATION SCOPE
# ==========================================
NEGATION_WORDS = {"not", "n't", "never", "no", "hardly", "nothing", "neither", "nor"}
# Not "believe": "can't believe how easy" is an exclamation, not a negated belief.
THINKING_VERBS = {"think", "thought", "feel", "felt", "find", "found", "expect", "expected"}
COMPARATIVE_CUES = {"easier", "healthier", "lighter"}  # negated comparatives are praise: "couldn't have been easier"
MODAL_RE = re.compile(r"\b(?:could|can|would)\b|\b(?:couldn|can|wouldn)['’]?t\b", re.IGNORECASE)
UNSURE_SCORE = 0.5  # below this probability the model's reading of a cue with its own polarity is not trusted
# Verbs that pass a negation on to the opinion word ("didn't get soggy", "doesn't have a greasy taste").
LINKING_VERBS = {"seem", "seems", "seemed", "get", "gets", "got", "turn", "turns", "turned", "come", "comes", "came",
                 "stay", "stays", "stayed", "taste", "tastes", "tasted", "have", "has", "had", "having", "make", "makes",
                 "made", "find", "found"}


def negation_of(span):
    """(negator, reverses) for the opinion word (a spaCy span); ("", False) when it is not negated. In scope: the word
    itself and its degree modifier ("not too spicy"), the words up to its verb ("wasn't bland", "don't like spicy
    food"), and a verb of thinking above that clause ("I don't think it was bland", but not the surprise in "I never
    thought I could get it this crispy"). Out of scope: coordinated words and clauses ("not too sweet, and fluffy"; "I
    didn't add salt and it was bland"), subordinate clauses, a negator after the word ("flavorful, not too spicy"),
    compounds ("no-fuss"), and a negation inside the cue ("no flavor"), which Task 4's polarity already includes.
    The negation reverses the word's polarity when it is on the word, a noun it describes ("no soggy cookies"), a
    copula or linking verb, or a verb of thinking; on another verb ("I don't like things too sweet") it does not."""
    token = span.root
    candidates = [(token, True), *[(c, True) for c in token.children if c.dep_ == "advmod"]]
    while token.pos_ not in ("VERB", "AUX") and token.dep_ not in ("ROOT", "conj"):
        token = token.head
        candidates.append((token, token.pos_ != "VERB" or token.lower_ in LINKING_VERBS))
    if (token.dep_ in ("ccomp", "xcomp") and token.head.lower_ in THINKING_VERBS
            and not any(c.lower_ in ("would", "could", "'d") for c in token.children)):
        candidates.append((token.head, True))
    doc = span.doc
    for t, reverses in candidates:
        for c in t.children:
            compound = c.i + 1 < len(doc) and doc[c.i + 1].text == "-"
            if c.i < span.start and not compound and (
                    c.dep_ == "neg" or (c.lower_ in NEGATION_WORDS and c.dep_ in ("advmod", "det"))):
                return c.text, reverses
    return "", False


# ==========================================
# 2. CONTRASTS, CONDITIONS AND EXCEPTIONS
# ==========================================
CONTRAST_RE = re.compile(r"^(?:but|although|however|yet|though|even though)\b", re.IGNORECASE)
EXCEPTION_RE = re.compile(r"\b(?:except(?: for| when)?|unless|other than|apart from)\s+(.*)", re.IGNORECASE)
# "but not" names an exception only before a situation ("but not at breakfast"), not a degree ("but not overly so").
BUT_NOT_RE = re.compile(r"^but not\s+((?:at|for|on|in|with|when|if|during)\b.*)", re.IGNORECASE)
STATEMENT_RE = re.compile(r"^(?:i|we|you|he|she|they|that)\b", re.IGNORECASE)  # "except I left out the onions"
OPPOSITE = {"positive": "negative", "negative": "positive", "neutral": "neutral"}


def phrase(text):
    """A condition up to the end of its phrase, without a leading 'if'/'when': 'if you add cream.' -> 'you add cream'."""
    return re.sub(r"^(?:if|when)\s+", "", re.split(r"[(),;.!?]", text.strip())[0].strip().lower())


QUESTION_VERBS = {"ask", "asked", "asks", "know", "knew", "wonder", "wondered", "see", "sure", "decide", "decided",
                  "check", "tell", "told"}  # "asked if it had pumpkin" is an indirect question, not a condition


def condition_of(span):
    """An 'if' clause attached to the opinion word, the words above it in its clause, or an infinitive under them
    ('quick to make if you use canned beans'). Indirect questions and 'even if' are not conditions."""
    token = span.root
    while True:
        for head in [token, *[c for c in token.children if c.dep_ == "xcomp"]]:
            for clause in head.children:
                marks = {c.lower_ for c in clause.children if c.dep_ == "mark"}
                if (clause.dep_ in ("advcl", "ccomp") and "if" in marks and head.lower_ not in QUESTION_VERBS
                        and not any(c.lower_ == "even" for c in clause.children)):
                    return phrase(span.doc[clause.left_edge.i:clause.right_edge.i + 1].text)
        if token.dep_ in ("ROOT", "conj"):
            return ""
        token = token.head


def exception_text(m):
    """The situation an exception names; '' for no match or when 'except' starts a statement ('except I left out the
    onions' reports a change, not an exception)."""
    if m is None or (m.group(0).lower().startswith(("except", "other", "apart")) and STATEMENT_RE.match(m.group(1))):
        return ""
    return phrase(m.group(1))


def exception_of(clause, cue_end, next_clause):
    """An exception after the opinion word ('except at breakfast'), or one that the next clause starts when that clause
    has no opinion of its own ('..., but not at breakfast'). next_clause is None when there is no such clause."""
    found = exception_text(EXCEPTION_RE.search(clause, cue_end))
    if not found and next_clause:
        found = exception_text(BUT_NOT_RE.match(next_clause) or EXCEPTION_RE.match(next_clause))
    return found


def rule_sentiment(prior, negator, reverses, comparative_praise, model, model_score):
    """Task 4's model sentiment, except: a negated comparative after "could"/"can" is praise ("it could not have been
    easier"); a cue with its own polarity takes the opposite one when the negation reverses it; and it keeps its own
    polarity when it is not negated and the model is unsure ("Difficult" alone)."""
    if comparative_praise:
        return "positive"
    if prior and reverses:
        return "negative" if prior > 0 else "positive"
    if prior and not negator and model_score < UNSURE_SCORE:
        return "positive" if prior > 0 else "negative"
    return model


def add_flags(rows, next_clauses):
    """rows: tuples with clause, start, end and sentiment columns. Adds the flags and one extra row per exception."""
    docs = dict(zip(rows["clause"].unique(), nlp_dep.pipe(rows["clause"].unique(), batch_size=256)))
    spans = [docs[c].char_span(s, e, alignment_mode="expand") for c, s, e in rows[["clause", "start", "end"]].itertuples(index=False)]
    rows = rows.copy()
    negations = [negation_of(span) for span in spans]
    rows["negator"] = [negator for negator, _ in negations]
    rows["negated"] = rows["negator"] != ""
    # An occasion fit ("great for weeknights") is positive, so "not great for weeknights" is negative.
    priors = [1 if aspect == "SUITABILITY" else 0 if normalise(cue) in COMPARATIVE_CUES else CUE_INDEX[normalise(cue)][1]
              for cue, aspect in zip(rows["cue"], rows["aspect"])]
    praise = [normalise(cue) in COMPARATIVE_CUES and negator != "" and bool(MODAL_RE.search(clause[:start]))
              for cue, (negator, _), clause, start in zip(rows["cue"], negations, rows["clause"], rows["start"])]
    rows["model_sentiment"] = rows["sentiment"]
    rows["sentiment"] = [rule_sentiment(p, negator, reverses, comparative_praise, model, score)
                         for p, (negator, reverses), comparative_praise, model, score in
                         zip(priors, negations, praise, rows["model_sentiment"], rows["model_score"])]
    rows["contrast"] = [bool(CONTRAST_RE.match(c)) for c in rows["clause"]]
    rows["condition"] = [condition_of(span) for span in spans]
    rows["condition_type"] = np.where(rows["condition"] != "", "condition", "")
    exceptions = [exception_of(c, e, n) for c, e, n in zip(rows["clause"], rows["end"], next_clauses)]
    extra = rows[[x != "" for x in exceptions]].copy()
    extra["condition"] = [x for x in exceptions if x]
    extra["condition_type"] = "exception"
    for col in ("sentiment", "model_sentiment", "lexicon_sentiment"):
        extra[col] = extra[col].map(OPPOSITE)
    return pd.concat([rows, extra]).sort_index(kind="stable")


# ==========================================
# 3. CHALLENGE SET (minimal pairs, manually checked)
# ==========================================
# Each group changes one thing at a time. Expected preferences were written and checked by hand, one tuple per
# preference: (aspect, sentiment, condition, negated); condition "" means the preference holds in general.
P, N = "positive", "negative"
CHALLENGE = [
    ("negation", "I like spicy food.", [("TASTE", P, "", False)]),
    ("negation", "I dislike spicy food.", [("TASTE", N, "", False)]),
    ("negation", "I don't like spicy food.", [("TASTE", N, "", True)]),
    ("negation", "It was too spicy.", [("TASTE", N, "", False)]),
    ("negation", "It was not too spicy.", [("TASTE", P, "", True)]),
    ("negation", "I don't like things too spicy.", [("TASTE", N, "", True)]),
    ("negation", "It was easy to make.", [("PREP_EFFORT", P, "", False)]),
    ("negation", "It was not easy to make.", [("PREP_EFFORT", N, "", True)]),
    ("negation", "It was not difficult to make.", [("PREP_EFFORT", P, "", True)]),
    ("negation", "The sauce was bland.", [("TASTE", N, "", False)]),
    ("negation", "The sauce was never bland.", [("TASTE", P, "", True)]),
    ("negation", "I don't think it was bland.", [("TASTE", P, "", True)]),
    ("negation", "It could not have been easier.", [("PREP_EFFORT", P, "", True)]),
    ("negation", "It was great for a weeknight.", [("SUITABILITY", P, "", False)]),
    ("negation", "It was not great for a weeknight.", [("SUITABILITY", N, "", True)]),
    ("negation scope", "I didn't add salt and it was bland.", [("TASTE", N, "", False)]),
    ("negation scope", "It was bland because I didn't add salt.", [("TASTE", N, "", False)]),
    ("negation scope", "The kids didn't eat much because it was too spicy.", [("TASTE", N, "", False)]),
    # Patterns found in real reviews: negation does not spread over a list, and surprise is not a negated belief.
    ("negation scope", "Not too sweet, and fluffy.", [("TASTE", P, "", True), ("TEXTURE", P, "", False)]),
    ("negation scope", "I never thought I could get the bacon this crispy.", [("TEXTURE", P, "", False)]),
    ("exception", "I enjoy spicy food, except at breakfast.", [("TASTE", P, "", False), ("TASTE", N, "at breakfast", False)]),
    ("exception", "I enjoy spicy food, especially at breakfast.", [("TASTE", P, "", False)]),
    ("exception", "I enjoy spicy food, but not at breakfast.", [("TASTE", P, "", False), ("TASTE", N, "at breakfast", False)]),
    ("exception", "Quick to make, unless you make the sauce from scratch.",
     [("TIME", P, "", False), ("TIME", N, "you make the sauce from scratch", False)]),
    ("condition", "Quick to make if you use canned beans.", [("TIME", P, "you use canned beans", False)]),
    ("condition", "Healthy if you skip the cream.", [("HEALTHINESS", P, "you skip the cream", False)]),
    ("condition", "Healthy, but not if you add the cream.",
     [("HEALTHINESS", P, "", False), ("HEALTHINESS", N, "you add the cream", False)]),
    ("contrast", "Delicious but difficult.", [("TASTE", P, "", False), ("PREP_EFFORT", N, "", False)]),
    ("contrast", "Difficult but delicious.", [("PREP_EFFORT", N, "", False), ("TASTE", P, "", False)]),
    ("contrast", "Delicious and easy.", [("TASTE", P, "", False), ("PREP_EFFORT", P, "", False)]),
    ("contrast", "Delicious, although not easy.", [("TASTE", P, "", False), ("PREP_EFFORT", N, "", True)]),
    ("contrast", "Not delicious, but easy.", [("TASTE", N, "", True), ("PREP_EFFORT", P, "", False)]),
]


def preferences_for(texts):
    """Task 4 mentions with the Task 6 flags for whole texts, split into clauses as in Task 1 (also used by Task 7)."""
    clauses = [(i, c) for i, text in enumerate(texts) for c in split_into_sentences(text)]
    rows = mentions_for(texts).rename(columns={"model_sentiment": "sentiment"})
    with_mention = set(rows["clause_pos"])
    next_clauses = [clauses[p + 1][1] if p + 1 < len(clauses) and clauses[p + 1][0] == clauses[p][0]
                    and p + 1 not in with_mention else None for p in rows["clause_pos"]]
    return add_flags(rows, next_clauses)


# Sentiment sources compared: Task 6 (negation scope) and Task 4's two methods, each with Task 6's flags.
SENTIMENT_SOURCES = {"Task 6": "sentiment", "ABSA model": "model_sentiment", "lexicon": "lexicon_sentiment"}
challenge = preferences_for([text for _, text, _ in CHALLENGE])
results = []
for i, (group, text, expected) in enumerate(CHALLENGE):
    rows = challenge[challenge["review"] == i]
    row = {"group": group, "sentence": text, "expected": sorted(expected),
           "negation flags": sorted(zip(rows["aspect"], rows["negated"])) == sorted((a, n) for a, _, _, n in expected)}
    for method, col in SENTIMENT_SOURCES.items():
        row[method] = sorted(zip(rows["aspect"], rows[col], rows["condition"], rows["negated"]))
        row[f"{method} correct"] = row[method] == row["expected"]
    results.append(row)
results = pd.DataFrame(results)

print(f"--- Challenge set: {len(results)} sentences in minimal pairs (correct = every extracted preference matches) ---")
summary = results.groupby("group", sort=False)[["negation flags"] + [f"{m} correct" for m in SENTIMENT_SOURCES]].sum()
summary.insert(0, "sentences", results.groupby("group", sort=False).size())
summary.loc["all"] = summary.sum()
print(summary)
for method in SENTIMENT_SOURCES:
    wrong = results[~results[f"{method} correct"]]
    print(f"\n{method} errors: {len(wrong)}")
    for _, r in wrong.iterrows():
        print(f"   {r['sentence']}  expected {r['expected']}  extracted {r[method]}")

# ==========================================
# 4. REAL REVIEWS
# ==========================================
# The next clause of each tuple's review, unless it has an aspect mention of its own.
reviews = set(df_tuples["review_row"])
clause_text = df_clauses[df_clauses["review_row"].isin(reviews)].set_index(["review_row", "clause_idx"])["sentences"]
with_mention = set(zip(df_tuples["review_row"], df_tuples["clause_idx"]))
next_clauses = [None if (r, k + 1) in with_mention else clause_text.get((r, k + 1))
                for r, k in zip(df_tuples["review_row"], df_tuples["clause_idx"])]
df_preferences = add_flags(df_tuples, next_clauses)

n = len(df_tuples)
print(f"\n--- Task 6 flags on {n} real preference tuples ---")
flags = {
    "negated": df_preferences["negated"] & (df_preferences["condition_type"] != "exception"),
    "in a contrast clause": df_preferences["contrast"] & (df_preferences["condition_type"] != "exception"),
    "with a condition": df_preferences["condition_type"] == "condition",
    "with an exception (one extra tuple each)": df_preferences["condition_type"] == "exception",
}
for name, mask in flags.items():
    print(f"{name}: {int(mask.sum())} ({mask.sum() / n:.1%})")
print("\nSentiment of negated and other tuples (share of each row):")
base = df_preferences[df_preferences["condition_type"] != "exception"]
print(pd.crosstab(base["negated"], base["sentiment"], normalize="index").round(2))
changed = base[base["sentiment"] != base["model_sentiment"]]
flags["sentiment changed from Task 4's model"] = df_preferences.index.isin(changed.index) & \
    (df_preferences["condition_type"] != "exception")
print(f"Sentiment changed from Task 4's model by the rules above: {len(changed)} tuples "
      f"({int(changed['negated'].sum())} negated)")
for name, mask in flags.items():
    shown = df_preferences[mask].sample(min(4, int(mask.sum())), random_state=SEED)
    print(f"\n{name}, e.g.:")
    for r in shown.itertuples():
        extra = f", condition={r.condition!r}" if r.condition else ""
        model = f" (Task 4 model: {r.model_sentiment})" if r.model_sentiment != r.sentiment else ""
        print(f"   ({r.holder}, {r.aspect}, {r.sentiment}, {r.context}{extra}){model} negator={r.negator!r} "
              f"{r.clause[:110]!r}")

# ==========================================
# 5. OUTPUT
# ==========================================
df_preferences.to_csv(os.path.join(OUT_DIR, "task6_preference_tuples.csv"), index=False)
print(f"\nSaved {len(df_preferences)} preference tuples ({len(df_preferences) - n} from exceptions) in {OUT_DIR}.")
