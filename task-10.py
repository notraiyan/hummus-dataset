# Imports already made by task-1.py to task-9.py (same notebook): json, os, random, re, np, pd, spacy, Example,
# compounding, minibatch, cohen_kappa_score, f1_score.

# Task 10: Annotation and Evaluation Protocol
# Run after task-9.py in the same notebook. Guidelines: ANNOTATION.md.
#   1. Annotation sheet: the 200-review pilot of Tasks 2 and 4 and 1,000 further reviews, pre-annotated by LLM and
#      checked and corrected by the project author (annotation_checked.csv); only rows marked checked = y are gold.
#   2. The system scored on the author's checked labels (NER precision/recall/F1; sentiment, condition and negation
#      macro-F1; tuple-level F1), and an NER model trained on the checked main-set clauses, tested on the pilot.
#   3. Contextual matching: manually authored requests, candidates pooled from each method, and blind relevance
#      judgments, scored with NDCG@10 and Recall@10 once judged. Task 8's chronological split cannot test context:
#      Food.com does not record the request behind a review, and the held-out review is not known in advance.
#   4. Topic ratings (Task 3's sheet) and explanation checks (Task 9's outputs), by one rater or two.
#   5. A status table: every metric of the protocol, what has been measured so far, and what is still missing.

for required in ("df_clauses", "df_reviews", "pilot_rows", "dictionary_ner", "model_ner", "nlp_ner", "preferences_for",
                 "rules_label", "split_into_sentences", "facts", "recipes", "catalog", "popularity", "reviewed",
                 "parse_request", "satisfies", "score_candidates", "member_facts", "extractive_summary", "df_out"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-9.py in this session first.")

SEED = 42
N_MAIN = 1_000                       # further reviews after the 200-review pilot
POOL_DEPTH = 10                      # candidates per method and request in the relevance pool
N_NER_EPOCHS = 10                    # training the NER model on the checked labels
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
ANNOTATION_DIR = OUT_DIR             # where annotation_checked.csv and the *_filled.csv sheets are read from


def filled(name):
    path = os.path.join(ANNOTATION_DIR, name)
    return pd.read_csv(path) if os.path.exists(path) else None


# ==========================================
# 1. ANNOTATION SHEET
# ==========================================
pilot = set(pilot_rows)  # the 200 reviews of the Task 2 and Task 4 pilot sheets
others = pd.Series(df_clauses["review_row"].unique())
others = others[~others.isin(pilot)].sample(N_MAIN, random_state=SEED).tolist()
sheet = df_clauses[df_clauses["review_row"].isin(pilot | set(others))][["review_row", "review_id", "clause_idx", "sentences"]]
sheet = sheet.rename(columns={"sentences": "clause"})
sheet.insert(0, "stage", np.where(sheet["review_row"].isin(pilot), "pilot", "main"))
sheet["review"] = df_reviews.loc[sheet["review_row"], "text"].to_numpy()
sheet[["entities", "opinions", "notes"]] = ""
sheet = sheet.sort_values(["stage", "review_row", "clause_idx"], ascending=[False, True, True])
sheet.to_csv(os.path.join(OUT_DIR, "annotation_sheet.csv"), index=False)
print(f"annotation_sheet.csv: {sheet['review_row'].nunique()} reviews, {len(sheet)} clauses")

# ==========================================
# 2. SYSTEM EVALUATION ON THE AUTHOR'S CHECKED LABELS
# ==========================================
def read_checked(name):
    """The rows marked checked = y: entities as {(span, LABEL)} and opinions as {(span, ASPECT, sentiment, holder,
    context, condition, negated)} per clause, spans lower-cased; indexed by (review_row, clause_idx)."""
    sheet = filled(name)
    if sheet is None:
        return None
    sheet = sheet[sheet["checked"].astype(str).str.strip().str.lower().str.startswith("y")].copy()
    parse = lambda v: json.loads(v) if isinstance(v, str) and v.strip() else []
    sheet["entities"] = sheet["entities"].map(lambda v: {(s.lower(), label) for s, label in parse(v)})
    sheet["opinions"] = sheet["opinions"].map(
        lambda v: {(o[0].lower(), *o[1:6], str(o[6]).lower() in ("true", "1", "yes")) for o in parse(v)})
    return sheet.set_index(["review_row", "clause_idx"])


def set_f1(gold, pred):
    """Micro precision, recall and F1 of predicted sets against gold sets (aligned lists); NaN when both are empty."""
    tp, n_pred, n_gold = sum(len(g & p) for g, p in zip(gold, pred)), sum(map(len, pred)), sum(map(len, gold))
    if n_pred == n_gold == 0:
        return np.nan, np.nan, np.nan
    prec, rec = (tp / n_pred if n_pred else 0.0), (tp / n_gold if n_gold else 0.0)
    return round(prec, 3), round(rec, 3), round(2 * prec * rec / (prec + rec) if prec + rec else 0.0, 3)


def system_opinions(gold):
    """System opinions (Tasks 4-6, holders and contexts from Task 5) for the gold clauses, as sets like the gold. An
    exception (Task 6 adds it as an extra tuple) is marked on the opinion it qualifies, as the annotators do."""
    review_rows = gold.index.get_level_values("review_row").unique()
    texts = df_reviews.loc[review_rows, "text"].tolist()
    prefs = preferences_for(texts)
    prefs["holder"], prefs["context"] = rules_label(prefs)
    first_clause = np.cumsum([0] + [len(split_into_sentences(t)) for t in texts])[:-1]
    prefs["key"] = [(review_rows[r], c - first_clause[r]) for r, c in zip(prefs["review"], prefs["clause_pos"])]
    excepted = set(zip(prefs.loc[prefs["condition_type"] == "exception", "clause_pos"],
                       prefs.loc[prefs["condition_type"] == "exception", "start"]))
    pred = {k: set() for k in gold.index}
    for r in prefs[prefs["condition_type"] != "exception"].itertuples():
        condition = "exception" if (r.clause_pos, r.start) in excepted else r.condition_type
        if r.key in pred:
            pred[r.key].add((r.cue.lower(), r.aspect, r.sentiment, r.holder, r.context, condition, bool(r.negated)))
    return [pred[k] for k in gold.index]


def evaluate_system(gold):
    texts = gold["clause"].astype(str).tolist()
    out = {}
    for name, preds in (("dictionary", dictionary_ner(texts)), ("domain model", model_ner(nlp_ner, texts))):
        out[f"NER {name} P/R/F1"] = set_f1(gold["entities"].tolist(),
                                           [{(t[s:e].lower(), label) for s, e, label in p} for t, p in zip(texts, preds)])
    predicted = system_opinions(gold)
    pairs = []  # same aspect, spans overlapping
    for g, p in zip(gold["opinions"], predicted):
        free = set(p)
        for go in g:
            match = next((po for po in free if po[1] == go[1] and (po[0] in go[0] or go[0] in po[0])), None)
            if match:
                pairs.append((go, match))
                free.discard(match)
    n_gold, n_pred = sum(map(len, gold["opinions"])), sum(map(len, predicted))
    prec, rec = len(pairs) / n_pred if n_pred else 0.0, len(pairs) / n_gold if n_gold else 0.0
    out["aspect detection P/R/F1"] = (round(prec, 3), round(rec, 3), round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0)
    for i, field in ((2, "sentiment"), (5, "condition"), (6, "negation")):
        out[f"{field} macro-F1"] = round(f1_score([str(g[i]) for g, _ in pairs], [str(p[i]) for _, p in pairs],
                                                  average="macro"), 3) if pairs else np.nan
    span_free = lambda sets: [{o[1:6] for o in s} for s in sets]  # (aspect, sentiment, holder, context, condition)
    out["tuple-level P/R/F1"] = set_f1(span_free(gold["opinions"]), span_free(predicted))
    return out


author_labels = read_checked("annotation_checked.csv")
annotation_results = None
if author_labels is not None and len(author_labels):
    n_reviews = author_labels.index.get_level_values("review_row").nunique()
    annotation_results = evaluate_system(author_labels)
    print(f"\n--- System on the author's checked labels ({len(author_labels)} clauses from {n_reviews} reviews) ---")
    for metric, value in annotation_results.items():
        print(f"   {metric}: {value}")
else:
    print("\nannotation_checked.csv (the author's checked labels, rows marked checked = y) is not in ANNOTATION_DIR "
          "yet: the system scores on the annotations are skipped.")


def entity_offsets(text, entities):
    """Character offsets of checked entity spans in their clause: the first free occurrence, longest span first."""
    taken = []
    for span, label in sorted(entities, key=lambda e: -len(e[0])):
        for m in re.finditer(re.escape(span), text.lower()):
            if all(m.end() <= s or m.start() >= e for s, e, _ in taken):
                taken.append((m.start(), m.end(), label))
                break
    return sorted(taken)


def ner_on_checked_labels(labels):
    """A domain-adapted NER model trained on the checked main-set clauses, and P/R/F1 on the checked pilot clauses (held
    out of Task 2's training) for it, the dictionary and Task 2's model trained on dictionary labels."""
    train_rows, test_rows = labels[labels["stage"] == "main"], labels[labels["stage"] == "pilot"]
    if not len(train_rows) or not len(test_rows):
        return None
    spacy.util.fix_random_seed(SEED)
    random.seed(SEED)
    nlp_checked = spacy.blank("en")
    ner_checked = nlp_checked.add_pipe("ner")
    for label in LABELS:  # Task 2's entity labels
        ner_checked.add_label(label)
    examples = []
    for text, entities in zip(train_rows["clause"].astype(str), train_rows["entities"]):
        doc = nlp_checked.make_doc(text)
        spans = [(s, e, l) for s, e, l in entity_offsets(text, entities) if doc.char_span(s, e) is not None]
        examples.append(Example.from_dict(doc, {"entities": spans}))
    optimizer = nlp_checked.initialize(lambda: examples)
    for _ in range(N_NER_EPOCHS):
        random.shuffle(examples)
        for batch in minibatch(examples, size=compounding(4.0, 32.0, 1.001)):
            nlp_checked.update(batch, sgd=optimizer, drop=0.2)
    texts = test_rows["clause"].astype(str).tolist()
    as_sets = lambda preds: [{(t[s:e].lower(), label) for s, e, label in p} for t, p in zip(texts, preds)]
    return {name: set_f1(test_rows["entities"].tolist(), as_sets(preds)) for name, preds in (
        ("dictionary", dictionary_ner(texts)), ("Task 2 model (dictionary labels)", model_ner(nlp_ner, texts)),
        ("model trained on checked labels", model_ner(nlp_checked, texts)))}


checked_ner_results = ner_on_checked_labels(author_labels) if annotation_results else None
if checked_ner_results:
    n_train, n_test = (author_labels["stage"] == "main").sum(), (author_labels["stage"] == "pilot").sum()
    print(f"\n--- NER on the checked pilot ({n_test} clauses), model trained on {n_train} checked main-set clauses ---")
    for name, value in checked_ner_results.items():
        print(f"   {name} P/R/F1: {value}")
elif annotation_results:
    print("\nThe NER model on checked labels needs checked clauses from both the pilot and the main set: skipped.")

# ==========================================
# 3. CONTEXTUAL MATCHING: AUTHORED REQUESTS AND JUDGED RELEVANCE
# ==========================================
# Written by hand before any candidate was seen. Each request goes to a member whose history (before Task 8's cut-off)
# has opinions of that holder, so the contextual profile has something to use.
AUTHORED_REQUESTS = [
    ("children", "Recommend a quick dinner for my children."),
    ("children", "A lunch my kids will like that takes less than 20 minutes."),
    ("family", "An easy weeknight dinner for the family."),
    ("family", "A family breakfast for the weekend."),
    ("partner", "A quick dinner for my husband."),
    ("guests", "Something to bring to a potluck with friends."),
    ("guests", "A dessert for my guests."),
    ("reviewer", "A quick weeknight dinner."),
    ("reviewer", "A healthy breakfast in under 15 minutes."),
    ("reviewer", "A dessert for Christmas."),
]
used_members = set()


def member_for(holder):
    """The unused member with the most opinions of that holder (any member when none is left)."""
    for counts in (facts.loc[facts["holder"] == holder, "member_id"].value_counts(), facts["member_id"].value_counts()):
        unused = [m for m in counts.index if m not in used_members]
        if unused:
            return unused[0]


def ranked_by_method(member, text):
    """Top POOL_DEPTH recipes per method, all within the request's checkable constraints."""
    request = parse_request(text)
    pool = catalog[satisfies(catalog, request)]
    pool = pool[~pool.isin(list(reviewed.get(member, set())))]
    tuples = facts[facts["member_id"] == member]
    aware = score_candidates(member, pool, request, tuples, contextual=True)
    blind = score_candidates(member, pool, request, tuples, contextual=False)
    scores = {"combined, contextual profile": aware["total"], "combined, context-independent profile": blind["total"],
              "content only": aware["content"], "CF": aware["CF"], "popularity": popularity[pool].to_numpy(dtype=float)}
    return {m: pool[np.argsort(-s, kind="stable")[:POOL_DEPTH]].tolist() for m, s in scores.items()}


key_rows, sheet_rows = [], []
for i, (holder, text) in enumerate(AUTHORED_REQUESTS):
    member = member_for(holder)
    used_members.add(member)
    rankings = ranked_by_method(member, text)
    key_rows += [{"request_id": i, "method": m, "rank": k + 1, "recipe_id": r}
                 for m, ranked in rankings.items() for k, r in enumerate(ranked)]
    history_note = extractive_summary(member_facts(member))  # Task 9: earlier opinions, quoted, with their holders
    for r in dict.fromkeys(r for ranked in rankings.values() for r in ranked):  # pooled, first-seen order
        rec = recipes.loc[r]
        sheet_rows.append({"request_id": i, "member_id": member, "request": text, "member_history": history_note,
                           "recipe_id": r, "title": rec["title"], "minutes": rec["duration"],
                           "steps": rec["direction_size"], "tags": ", ".join(rec["tag_list"][:12]),
                           "relevance_A": "", "relevance_B": ""})
relevance_key = pd.DataFrame(key_rows)
relevance_sheet = pd.DataFrame(sheet_rows).sample(frac=1, random_state=SEED)  # shuffled, so the method is hidden
relevance_sheet.to_csv(os.path.join(OUT_DIR, "relevance_sheet.csv"), index=False)
relevance_key.to_csv(os.path.join(OUT_DIR, "relevance_key.csv"), index=False)
print(f"\nrelevance_sheet.csv: {len(AUTHORED_REQUESTS)} authored requests, {len(relevance_sheet)} candidates to judge "
      f"(top {POOL_DEPTH} of {relevance_key['method'].nunique()} methods, pooled)")

matching_results = None
judged = filled("relevance_sheet_filled.csv")
if judged is not None:
    raters = [c for c in ("relevance_A", "relevance_B") if judged[c].notna().any()]
    both = judged.dropna(subset=["relevance_A", "relevance_B"])
    if len(raters) == 2 and len(both):
        print(f"Relevance agreement (quadratic-weighted kappa, {len(both)} items): "
              f"{cohen_kappa_score(both['relevance_A'].astype(int), both['relevance_B'].astype(int), weights='quadratic'):.3f}")
    relevance = judged.set_index(["request_id", "recipe_id"])[raters].mean(axis=1).fillna(0.0)
    rows = []
    for (request_id, method), ranked in relevance_key.sort_values("rank").groupby(["request_id", "method"])["recipe_id"]:
        gains = np.array([relevance.get((request_id, r), 0.0) for r in ranked])
        ideal = np.sort(relevance.loc[request_id].to_numpy())[::-1][:POOL_DEPTH]
        discount = 1 / np.log2(np.arange(2, POOL_DEPTH + 2))
        dcg, idcg = ((2 ** gains - 1) * discount[:len(gains)]).sum(), ((2 ** ideal - 1) * discount[:len(ideal)]).sum()
        n_relevant = (relevance.loc[request_id] >= 1).sum()
        rows.append({"method": method, "NDCG@10": dcg / idcg if idcg else 0.0,
                     "Recall@10": (gains >= 1).sum() / n_relevant if n_relevant else 0.0})
    matching_results = pd.DataFrame(rows).groupby("method")[["NDCG@10", "Recall@10"]].mean().round(3)
    print(f"\n--- Contextual matching on {len(AUTHORED_REQUESTS)} authored requests (judged relevance) ---")
    print(matching_results)
else:
    print("Relevance judgments are not in yet: NDCG@10 and Recall@10 for the authored requests will appear here once "
          "relevance_sheet_filled.csv is in ANNOTATION_DIR.")

# ==========================================
# 4. TOPIC RATINGS AND EXPLANATION CHECKS
# ==========================================
topic_sheets = [s for s in (filled("topic_ratings_A.csv"), filled("topic_ratings_B.csv")) if s is not None]
topic_results = None
if topic_sheets:
    ratings = pd.concat([s[["model", "topic", "interpretable_1to3"]].assign(rater=i) for i, s in enumerate(topic_sheets)])
    ratings = ratings.dropna()
    topic_results = ratings.groupby("model")["interpretable_1to3"].mean().round(2)
    note = f"{len(topic_sheets)} rater(s)"
    if len(topic_sheets) == 2:
        paired = ratings.pivot_table(index=["model", "topic"], columns="rater", values="interpretable_1to3").dropna()
        note += f", weighted kappa {cohen_kappa_score(paired[0].astype(int), paired[1].astype(int), weights='quadratic'):.3f}"
    print(f"\nTopic interpretability (1-3, {note}):", topic_results.to_dict())

checks = df_out[["member_id", "output", "request", "recipe", "text", "facts"]].copy()
for field in ("supported", "attribution", "conditions"):
    checks[f"{field}_A"], checks[f"{field}_B"] = "", ""
checks.to_csv(os.path.join(OUT_DIR, "explanation_checks.csv"), index=False)
explanation_human = None
returned = filled("explanation_checks_filled.csv")
if returned is not None:
    raters = [r for r in ("A", "B") if returned[f"supported_{r}"].notna().any()]
    yes = lambda col: returned[col].astype(str).str.lower().str.startswith("y")
    explanation_human = pd.DataFrame({
        field: returned.assign(v=np.logical_and.reduce([yes(f"{field}_{r}") for r in raters])).groupby("output")["v"].mean()
        for field in ("supported", "attribution", "conditions")})
    print(f"\nExplanation checks by {len(raters)} rater(s) (share judged yes by every rater):")
    print(explanation_human.round(3))

# ==========================================
# 5. STATUS OF THE PROTOCOL
# ==========================================
def value(name, compute):
    return compute() if name in globals() else "not run in this session"


# Task 3 writes its scores to a file (its `scores` name is reused by later tasks).
topic_files = sorted(f for f in os.listdir(OUT_DIR) if f.startswith("topic_scores_") and f.endswith(".csv"))
topic_scores = pd.read_csv(os.path.join(OUT_DIR, topic_files[-1])) if topic_files else None


def source(pilot_source):
    return f"the author's checked labels ({len(author_labels)} clauses)" if annotation_results else pilot_source


status = [
    ("Annotation", "clauses checked and corrected by the author",
     f"{len(author_labels)} clauses from {n_reviews} reviews" if annotation_results else "pending: annotation_checked.csv",
     "LLM's pre-annotation of 1,200 reviews, checked by the author"),
    ("NER", "entity-level P/R/F1",
     (f"dictionary {annotation_results['NER dictionary P/R/F1']}, Task 2 model "
      f"{annotation_results['NER domain model P/R/F1']}" + (
          f"; pilot, model trained on checked labels {checked_ner_results['model trained on checked labels']} vs "
          f"dictionary {checked_ner_results['dictionary']}" if checked_ner_results else ""))
     if annotation_results else value("comparison", lambda: (
         f"pilot gold, micro F1: dictionary {comparison.loc['MICRO', ('dictionary', 'f1')]:.2f}, "
         f"domain model {comparison.loc['MICRO', ('domain model', 'f1')]:.2f}")), source("Task 2 (41 hand-written sentences)")),
    ("Sentiment", "macro-F1",
     annotation_results["sentiment macro-F1"] if annotation_results else value("gold_results", lambda: (
         f"pilot gold: ABSA model {gold_results.loc['ABSA model', 'sentiment macro-F1']:.2f}, "
         f"lexicon {gold_results.loc['lexicon', 'sentiment macro-F1']:.2f}")), source("Task 4 (42 hand-written reviews)")),
    ("Conditions", "macro-F1 (condition, negation)",
     f"{annotation_results['condition macro-F1']}, {annotation_results['negation macro-F1']}" if annotation_results
     else value("results", lambda: f"challenge set: {int(results['Task 6 correct'].sum())}/{len(results)} sentences, "
                                    f"negation flags {int(results['negation flags'].sum())}/{len(results)}"),
     source("Task 6 (challenge set of minimal pairs)")),
    ("Relation extraction", "tuple-level F1",
     str(annotation_results["tuple-level P/R/F1"]) if annotation_results else value("gold_mentions", lambda: (
         "pilot gold, tuple accuracy: " + ", ".join(f"{name} {score(h, c, gold_mentions['gold'])['tuple accuracy']:.2f}"
                                                   for name, (h, c) in methods.items()))), source("Task 5 (45 gold mentions)")),
    ("Topic modelling", "coherence, diversity, human ratings",
     ("; ".join(f"{m}: NPMI {n}, diversity {d}" for m, n, d in
                zip(topic_scores["model"], topic_scores["coherence_npmi"], topic_scores["diversity@25"]))
      if topic_scores is not None else "Task 3 scores file not found")
     + ("" if topic_results is not None else "; human ratings pending"), "Task 3"),
    ("Recommendation", "NDCG@10, Recall@10 (chronological, history only)",
     value("evaluation", lambda: "; ".join(
         f"{m}: {evaluation[(evaluation['scheme'] == 'popularity-matched') & (evaluation['method'] == m)]['NDCG@10'].mean():.3f}"
         for m in ("combined, contextual profile", "CF", "popularity"))) + " (NDCG@10, popularity-matched negatives)",
     "Task 8 (300 members)"),
    ("Contextual matching", "NDCG@10, Recall@10 (authored requests)",
     "see table above" if matching_results is not None else "pending: relevance_sheet.csv exported", "Task 10"),
    ("Explanations", "evidence support, incorrect-attribution rate",
     "; ".join(f"{o}: support {(g['cited'].sum() / g['sentences'].sum()):.2f}, incorrect attribution "
               f"{1 - np.mean([v for l in g['attribution'] for v in l]):.2f}" for o, g in df_out.groupby("output", sort=False)
               if any(len(l) for l in g["attribution"])), "Task 9 (15 members, automatic checks)"),
]
df_status = pd.DataFrame(status, columns=["area", "metric", "measured so far", "data"])
print("\n--- Evaluation protocol status ---")
with pd.option_context("display.width", 250, "display.max_colwidth", 140):
    print(df_status.to_string(index=False))
df_status.to_csv(os.path.join(OUT_DIR, "evaluation_status.csv"), index=False)
