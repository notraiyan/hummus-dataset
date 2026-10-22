# Imports already made by task-1.py to task-4.py (same notebook): json, os, re, np, pd, spacy, torch, cohen_kappa_score,
# f1_score.
from transformers import AutoModelForCausalLM, AutoTokenizer

# Task 5: Preference-Holder and Relation Extraction
# Run after task-4.py in the same notebook (uses df_aspects and analyse from Task 4, df_clauses and
# split_into_sentences from Task 1, OCCASIONS and plural_variants from Task 2).
# Each Task 4 aspect mention becomes a tuple (holder, aspect, sentiment, context, evidence span). Aspect, sentiment and
# evidence come from Task 4; this task adds:
#   holder:  reviewer, children, partner, family, guests, or unknown when it cannot be told who holds the opinion
#   context: the occasion the opinion is tied to (weekday, weekend, holiday, gathering, special occasion),
#            general when none is named, unknown when the clause names two different ones
# Two extractors are compared: dependency-parsing rules (spaCy) and a free, open LLM (Qwen3-4B-Instruct, run locally
# with transformers, no account or key) that must answer in a fixed JSON schema. The model is downloaded once (8 GB,
# internet on); a GPU (Kaggle: Settings > Accelerator) makes it fast. Without it the rules alone produce the tuples.

for required in ("df_aspects", "analyse", "df_clauses", "split_into_sentences", "OCCASIONS", "plural_variants"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-4.py in this session first.")

SEED = 42
N_LLM_REVIEWS = 40        # real reviews also labelled by the LLM, to measure agreement with the rules
LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"  # Apache-2.0; "Qwen/Qwen2.5-1.5B-Instruct" is smaller and faster on CPU
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ==========================================
# 1. LABELS
# ==========================================
HOLDERS = ["reviewer", "children", "partner", "family", "guests", "unknown"]
HOLDER_WORDS = {
    "reviewer": "i me my myself",
    # No "baby", "boy" or "girl": in reviews they are mostly food ("baby carrots") or "oh boy".
    "children": "kid kids kiddo kiddos child children toddler toddlers son sons daughter daughters boys girls teen "
                "teens teenager teenagers grandkids grandchildren ds dd",
    "partner": "husband wife hubby partner boyfriend girlfriend fiance fiancee spouse dh dw bf",
    "family": "we us our ourselves family parents mom mother dad father sister brother grandma grandpa laws",  # in-laws
    "guests": "guest guests company friend friends crowd coworkers neighbors visitors",
    # Pronouns whose referent is outside the clause. "they"/"it" are left out: in reviews they are mostly the food.
    "unknown": "he she him her everyone everybody",
}
HOLDER_OF = {word: holder for holder, words in HOLDER_WORDS.items() for word in words.split()}
FIRST_PERSON = {"i", "me", "my", "myself", "we", "us", "our", "ourselves"}
COOK_ASPECTS = {"PREP_EFFORT", "TIME", "INGREDIENT_AVAILABILITY"}  # judged by whoever cooked: the reviewer

# Task 2's occasions, grouped into the contexts a profile is kept under.
CONTEXT_GROUPS = {
    "weekday": ["weeknight", "weekday", "school night", "work night", "lunchbox"],
    "weekend": ["weekend", "sunday dinner"],
    "holiday": ["holiday", "christmas", "thanksgiving", "easter", "new year's eve", "valentine's day", "halloween",
                "hanukkah"],
    "gathering": ["party", "dinner party", "cocktail party", "potluck", "bbq", "barbecue", "cookout", "picnic",
                  "game day", "super bowl", "tailgate", "baby shower", "wedding"],
    "special occasion": ["special occasion", "birthday", "anniversary", "date night"],
}
CONTEXTS = list(CONTEXT_GROUPS) + ["general", "unknown"]
CONTEXT_OF = {occasion: context for context, occasions in CONTEXT_GROUPS.items() for occasion in occasions}
OCCASION_FORMS = {form: base for base in OCCASIONS for form in plural_variants(base)}
OCCASION_RE = re.compile(r"\b(?:" + "|".join(map(re.escape, sorted(OCCASION_FORMS, key=len, reverse=True))) + r")\b",
                         re.IGNORECASE)

# ==========================================
# 2. DEPENDENCY-PARSING RULES
# ==========================================
try:
    nlp_dep = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
except OSError:  # not installed: download it (needs internet)
    spacy.cli.download("en_core_web_sm")
    nlp_dep = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

CLIMB_DEPS = {"ccomp", "xcomp", "pcomp", "advcl", "prep", "pobj"}  # "my husband said [it was too salty]"


def person(token):
    """Holder class of a word naming people, else None. 'for my taste' is the reviewer."""
    if token.lower_ in ("taste", "liking"):
        token = next((c for c in token.children if c.dep_ == "poss"), token)
    return HOLDER_OF.get(token.lower_)


def experiencer(token):
    """Person the opinion word is judged for: 'too hot for my children', 'spicy enough for my husband', 'dry for my
    taste', 'easy enough for my teen to make'. A plain 'great for company' says who the dish suits, so it is ignored."""
    graded = any(c.lower_ in ("too", "enough") for c in token.children)
    for f in [*token.children, *token.head.children]:  # 'for' on the opinion word or on its verb
        if f.lower_ == "for" and f.dep_ == "prep":
            for target in f.children:
                if target.dep_ == "pobj" and (graded or target.lower_ in ("taste", "liking")) and person(target):
                    return person(target)
    for clause in token.children:  # 'for my teen to make'
        if clause.dep_ in ("advcl", "xcomp") and any(c.lower_ == "for" and c.dep_ == "mark" for c in clause.children):
            subject = next((c for c in clause.children if c.dep_ == "nsubj"), None)
            if subject is not None and person(subject):
                return person(subject)
    return None


def subject_holder(subject):
    """Holder named by a subject; two people ('my husband and I', 'my husband and daughter') are the family."""
    holders = {person(t) for t in [subject, *subject.conjuncts]} - {None}
    return holders.pop() if len(holders) == 1 else "family" if holders else None


def rule_holder(doc, start, end, aspect):
    """An experiencer phrase on the opinion word, else the subject of the first verb above it. A non-person subject
    ("it was too salty") ends the walk unless that clause is the complement of a verb with a person subject
    ("my husband said ...", "the kids thought ..."). Coordinated verbs are not crossed."""
    token = doc.char_span(start, end, alignment_mode="expand").root
    holder = experiencer(token)
    if holder or aspect in COOK_ASPECTS:  # effort, time and availability are judged by whoever cooked: the reviewer
        return holder or "reviewer"
    while True:
        subject = next((c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")), None)
        if subject is not None:
            holder = subject_holder(subject)
            if holder:
                return holder
            if token.dep_ not in CLIMB_DEPS:
                break
        if token.dep_ == "ROOT" or (token.dep_ == "conj" and token.pos_ in ("VERB", "AUX")):
            break
        token = token.head
    # No holder in the syntax: for taste, texture and healthiness, a person named elsewhere in the clause
    # ("made it for my kids and it was too spicy") makes the holder unclear. People named with an occasion fit are its
    # audience ("great for a party with friends"), so the fit is the reviewer's.
    others = [t for t in doc if t.lower_ in HOLDER_OF and t.lower_ not in FIRST_PERSON
              and t.dep_ not in ("compound", "poss")]
    return "unknown" if others and aspect != "SUITABILITY" else "reviewer"


def rule_context(clause):
    """Context of the occasion named in the clause. Adjacent occasion words are one phrase, named by its last word
    ("Christmas party" -> gathering)."""
    spans = []
    for m in OCCASION_RE.finditer(clause):
        base = OCCASION_FORMS[m.group(0).lower()]
        context = CONTEXT_OF.get(base, base)
        if spans and not clause[spans[-1][1]:m.start()].strip():
            spans[-1] = (spans[-1][0], m.end(), context)
        else:
            spans.append((m.start(), m.end(), context))
    contexts = {c for _, _, c in spans}
    return "general" if not contexts else contexts.pop() if len(contexts) == 1 else "unknown"


def rules_label(mentions):
    """mentions: DataFrame with clause, start, end, aspect -> lists of holders and contexts."""
    docs = dict(zip(mentions["clause"].unique(), nlp_dep.pipe(mentions["clause"].unique(), batch_size=256)))
    holders = [rule_holder(docs[c], s, e, a) for c, s, e, a in mentions[["clause", "start", "end", "aspect"]].itertuples(index=False)]
    return holders, [rule_context(c) for c in mentions["clause"]]


# ==========================================
# 3. LLM WITH A FIXED OUTPUT SCHEMA (free, open model run locally)
# ==========================================
# One prompt per mention. The answer must be a JSON object using only the labels above; anything else is a schema
# violation and becomes "unknown".
ANSWER_FORMAT = '{"holder": "<' + "|".join(HOLDERS) + '>", "context": "<' + "|".join(CONTEXTS) + '>"}'
LLM_INSTRUCTIONS = """You label opinions in recipe reviews from Food.com. You get a review, one clause from it, and a \
word in that clause that carries an opinion about an aspect of the recipe. Decide:
- holder: whose opinion it is. "reviewer" is the person who wrote the review, also when nobody else is named. \
"children" are the reviewer's kids or grandkids, "partner" is a husband, wife, boyfriend or girlfriend, "family" is the \
household as a group ("we", "the whole family") or other relatives, and "guests" are friends, company or people at a \
party. Use "unknown" when the review does not make clear whose opinion it is.
- context: the occasion the opinion is tied to. "weekday" (weeknights, school nights, packed lunches), "weekend", \
"holiday" (Christmas, Thanksgiving, Easter and other holidays), "gathering" (parties, potlucks, barbecues, game day), \
"special occasion" (birthdays, anniversaries, date night), "general" when no occasion is named for that opinion, or \
"unknown" when it is unclear which named occasion applies.
Use only what the review says. Answer with one JSON object and nothing else: """ + ANSWER_FORMAT

llm_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
try:
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, padding_side="left")
    llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL, dtype=torch.float32 if llm_device == "cpu" else torch.float16)
    llm = llm.to(llm_device).eval()
except OSError as e:  # the model could not be downloaded (internet off)
    llm = None
    print(f"LLM comparison skipped: could not load {LLM_MODEL} ({type(e).__name__}). Turn internet on to download it.")


def parse_answer(text):
    """(holder, context, valid) from the model's answer; labels outside the schema become 'unknown'."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    try:
        answer = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        answer = {}
    holder, context = answer.get("holder"), answer.get("context")
    valid = holder in HOLDERS and context in CONTEXTS
    return (holder if holder in HOLDERS else "unknown"), (context if context in CONTEXTS else "unknown"), valid


def llm_label(mentions, review_texts, batch_size=8):
    """mentions: DataFrame with review, aspect, cue, clause; review_texts: {review: text}. Greedy decoding."""
    prompts = [llm_tokenizer.apply_chat_template(
        [{"role": "system", "content": LLM_INSTRUCTIONS},
         {"role": "user", "content": f'Review: {review_texts[review]}\nClause: {clause}\nOpinion word: "{cue}" ({aspect})'}],
        tokenize=False, add_generation_prompt=True)
        for review, aspect, cue, clause in mentions[["review", "aspect", "cue", "clause"]].itertuples(index=False)]
    answers = []
    for i in range(0, len(prompts), batch_size):
        batch = llm_tokenizer(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(llm_device)
        with torch.no_grad():
            output = llm.generate(**batch, max_new_tokens=40, do_sample=False)
        answers += llm_tokenizer.batch_decode(output[:, batch["input_ids"].shape[1]:], skip_special_tokens=True)
    labels = [parse_answer(a) for a in answers]
    print(f"LLM answers outside the schema: {sum(not valid for _, _, valid in labels)} of {len(labels)}")
    return [h for h, _, _ in labels], [c for _, c, _ in labels]


# ==========================================
# 4. EVALUATION SET (small, hand-written pilot gold)
# ==========================================
# Whole reviews, split into clauses and aspects exactly as in Tasks 1 and 4. Gold: {cue: (holder, context)}. Some items
# need information the rules cannot use: "they" referring to the kids, "everyone" at a party. Extend with Task 10.
GOLD = [
    ("I love spicy food, but this was too hot for my children.",
     {"spicy": ("reviewer", "general"), "too hot": ("children", "general")}),
    ("My husband said it was too salty.", {"too salty": ("partner", "general")}),
    ("The kids loved the flavor.", {"flavor": ("children", "general")}),
    ("Too time-consuming for weeknights, but perfect for the weekend.",
     {"time-consuming": ("reviewer", "weekday"), "perfect for the weekend": ("reviewer", "weekend")}),
    ("Quick and easy for a busy weeknight.", {"quick": ("reviewer", "weekday"), "easy": ("reviewer", "weekday")}),
    ("My guests raved about how moist it was.", {"moist": ("guests", "general")}),
    ("It was a bit dry for my taste.", {"dry": ("reviewer", "general"), "taste": ("reviewer", "general")}),
    ("We served it at our Christmas party and the texture was perfect.", {"texture": ("reviewer", "gathering")}),
    ("My son thought it was bland.", {"bland": ("children", "general")}),
    ("My wife found the sauce too sweet.", {"too sweet": ("partner", "general")}),
    ("Made this for my kids and it was too spicy.", {"too spicy": ("unknown", "general")}),
    ("Easy to make, and my husband loved the flavor.",
     {"easy to make": ("reviewer", "general"), "flavor": ("partner", "general")}),
    ("The whole family thought it was delicious.", {"delicious": ("family", "general")}),
    ("Everyone at the potluck said it was delicious.", {"delicious": ("guests", "gathering")}),
    ("Too heavy for a weeknight dinner.", {"heavy": ("reviewer", "weekday")}),
    ("Our guests found it a little dry.", {"dry": ("guests", "general")}),
    ("Perfect texture.", {"texture": ("reviewer", "general")}),
    ("My toddler wouldn't eat it because it was too spicy.", {"too spicy": ("children", "general")}),
    ("I thought it was great but my husband found it bland.", {"bland": ("partner", "general")}),
    ("We had it for Thanksgiving and it was a little bland.", {"bland": ("reviewer", "holiday")}),
    ("We loved how crispy it was.", {"crispy": ("family", "general")}),
    ("Great for a quick weeknight dinner.",
     {"quick": ("reviewer", "weekday"), "great for a quick weeknight": ("reviewer", "weekday")}),
    ("Took forever, so it's only for special occasions.", {"took forever": ("reviewer", "special occasion")}),
    ("My mother-in-law said the crust was soggy.", {"soggy": ("family", "general")}),
    ("Our friends couldn't stop talking about how tender the meat was.", {"tender": ("guests", "general")}),
    ("It was easy enough for my teenager to make.", {"easy": ("children", "general")}),
    ("I couldn't find tahini, so I used peanut butter.", {"couldn't find": ("reviewer", "general")}),
    ("Healthy and filling.", {"healthy": ("reviewer", "general")}),
    ("Too salty for us.", {"too salty": ("family", "general")}),
    ("The texture was too mushy for my boyfriend.", {"mushy": ("partner", "general")}),
    ("Made it for a birthday party and the frosting was too sweet.", {"too sweet": ("reviewer", "gathering")}),
    ("The guests loved it, though it was a bit dry.", {"dry": ("reviewer", "general")}),
    ("I'd make it for company because it is easy.", {"easy": ("reviewer", "general")}),
    ("He said it was too dry.", {"dry": ("unknown", "general")}),
    ("My family and I thought it was delicious.", {"delicious": ("family", "general")}),
    ("Our dinner guests at Easter said the lamb was tender.", {"tender": ("guests", "holiday")}),
    ("My kids are picky, but they loved the flavor.", {"flavor": ("children", "general")}),
    ("For a school night this was quick, and the kids thought it was yummy.",
     {"quick": ("reviewer", "weekday"), "yummy": ("children", "weekday")}),
    ("Spicy enough for my husband.", {"spicy": ("partner", "general")}),
    ("We made it for a Super Bowl party and everyone said it was too salty.", {"too salty": ("guests", "gathering")}),
]


def mentions_for(texts):
    """Task 4 aspect mentions (with sentiment) for whole reviews, from their Task 1 clauses."""
    clauses, owner = [], []
    for i, text in enumerate(texts):
        for clause in split_into_sentences(text):
            clauses.append(clause)
            owner.append(i)
    return pd.DataFrame([dict(m, review=owner[m["clause_pos"]], clause=clauses[m["clause_pos"]]) for m in analyse(clauses)])


gold_texts = [text for text, _ in GOLD]
gold_mentions = mentions_for(gold_texts)
gold_mentions["gold"] = [GOLD[r][1].get(cue.lower()) for r, cue in zip(gold_mentions["review"], gold_mentions["cue"])]
unlabelled = gold_mentions["gold"].isna()
if unlabelled.any():
    print("Mentions without a gold label (not scored):", gold_mentions.loc[unlabelled, ["review", "cue"]].values.tolist())
gold_mentions = gold_mentions[~unlabelled].reset_index(drop=True)
missed = sum(len(g) for _, g in GOLD) - len(gold_mentions)
if missed:
    print(f"{missed} gold cues were not found as Task 4 mentions (not scored).")

methods = {"dependency rules": rules_label(gold_mentions)}
if llm is not None:
    methods["LLM (fixed schema)"] = llm_label(gold_mentions, dict(enumerate(gold_texts)))


def score(holders, contexts, gold):
    gold_h, gold_c = [g[0] for g in gold], [g[1] for g in gold]
    return {
        "holder accuracy": np.mean([h == g for h, g in zip(holders, gold_h)]),
        "holder macro-F1": f1_score(gold_h, holders, labels=sorted(set(gold_h)), average="macro", zero_division=0),
        "context accuracy": np.mean([c == g for c, g in zip(contexts, gold_c)]),
        "tuple accuracy": np.mean([(h, c) == g for h, c, g in zip(holders, contexts, gold)]),
        "unknown holders": int(sum(h == "unknown" for h in holders)),
    }


print(f"\n--- Gold evaluation: {len(gold_mentions)} aspect mentions in {len(GOLD)} hand-written reviews "
      f"(gold unknown holders: {sum(g[0] == 'unknown' for g in gold_mentions['gold'])}) ---")
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(pd.DataFrame({name: score(h, c, gold_mentions["gold"]) for name, (h, c) in methods.items()}).T.round(3))

print("\n--- Gold errors ---")
for i, row in gold_mentions.iterrows():
    predicted = {name: (h[i], c[i]) for name, (h, c) in methods.items()}
    if any(p != row["gold"] for p in predicted.values()):
        print(f"{gold_texts[row['review']]}  [{row['cue']}]\n   gold {row['gold']}  " +
              "  ".join(f"{name} {p}" for name, p in predicted.items()))

# ==========================================
# 5. PREFERENCE TUPLES FOR THE REAL REVIEWS
# ==========================================
df_tuples = df_aspects.rename(columns={"model_sentiment": "sentiment"}).copy()
df_tuples["holder"], df_tuples["context"] = rules_label(df_tuples)
df_tuples["evidence"] = df_tuples["clause"]  # the clause the sentiment was read from; cue offsets are start/end

n_reviews = df_tuples["review_row"].nunique()
print(f"\n--- Rule-based tuples: {len(df_tuples)} from {n_reviews} reviews ---")
print(pd.crosstab(df_tuples["holder"], df_tuples["sentiment"], margins=True).reindex(HOLDERS + ["All"]).fillna(0).astype(int))
print("\nContexts:", df_tuples["context"].value_counts().to_dict())
others = df_tuples[~df_tuples["holder"].isin(["reviewer", "unknown"])]
print(f"Opinions held by someone other than the reviewer: {len(others)} ({len(others) / len(df_tuples):.1%}); "
      f"a profile without holders would credit all of them to the reviewer.")
print("\n--- Examples of tuples not held by the reviewer ---")
for r in others.sample(min(8, len(others)), random_state=SEED).itertuples():
    print(f"({r.holder}, {r.aspect}, {r.sentiment}, {r.context}, {r.evidence!r})")

# LLM on a sample of real reviews: agreement with the rules on the same mentions.
if llm is not None:
    sample_rows = df_tuples["review_row"].drop_duplicates().sample(min(N_LLM_REVIEWS, n_reviews), random_state=SEED)
    sample = df_tuples[df_tuples["review_row"].isin(sample_rows)].rename(columns={"review_row": "review"})
    review_texts = df_clauses[df_clauses["review_row"].isin(sample_rows)].groupby("review_row")["sentences"].agg(" ".join)
    sample["llm_holder"], sample["llm_context"] = llm_label(sample, review_texts.to_dict())
    df_tuples.loc[sample.index, "llm_holder"] = sample["llm_holder"]
    df_tuples.loc[sample.index, "llm_context"] = sample["llm_context"]
    print(f"\n--- Rules vs LLM on {len(sample)} mentions from {len(sample_rows)} real reviews ---")
    print(f"holder agreement {(sample['holder'] == sample['llm_holder']).mean():.3f} "
          f"(kappa {cohen_kappa_score(sample['holder'], sample['llm_holder']):.3f}), "
          f"context agreement {(sample['context'] == sample['llm_context']).mean():.3f}")
    print(pd.crosstab(sample["holder"], sample["llm_holder"], rownames=["rules"], colnames=["LLM"]))

# The project's motivating example, end to end.
EXAMPLE = ("I love spicy food, but this was too hot for my children. "
           "Great for weekend dinners, but too time-consuming for weekdays.")
example = mentions_for([EXAMPLE])
example["holder"], example["context"] = rules_label(example)
print("\n--- Motivating example (dependency rules) ---")
for r in example.itertuples():
    print(f"({r.holder}, {r.aspect}, {r.model_sentiment}, {r.context}, {r.clause!r})")

# ==========================================
# 6. OUTPUT
# ==========================================
cols = [c for c in ("review_row", "review_id", "member_id", "recipe_id", "clause_idx", "holder", "aspect", "sentiment",
                    "context", "evidence", "cue", "start", "end", "llm_holder", "llm_context") if c in df_tuples.columns]
df_tuples[cols].to_csv(os.path.join(OUT_DIR, "task5_preference_tuples.csv"), index=False)
print(f"\nSaved {len(df_tuples)} preference tuples in {OUT_DIR}.")
