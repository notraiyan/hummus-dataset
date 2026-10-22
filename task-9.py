# Imports already made by task-1.py to task-8.py (same notebook): json, os, re, np, pd, torch, sent_tokenize.

# Task 9: Evidence-Grounded Summarization and Explanations
# Run after task-8.py in the same notebook (uses facts, recipes, catalog, reviewed, parse_request, satisfies and
# score_candidates from Task 8, build_profiles from Task 7, and the local LLM from Task 5).
# Every sentence cites its source: [E#] for an opinion from the member's earlier reviews (a quote, with its holder and
# occasion or condition) or [R] for the recipe's structured record (time, steps, tags). The facts come from reviews
# before Task 8's cut-off, so every personalised claim traces back to historical evidence.
#   profile summaries:           extractive (labelled quotes) vs generated (the Task 5 LLM, same facts)
#   recommendation explanations: template (sentences built from the facts) vs generated (the LLM, same facts), and a
#                                context-blind template that cites a profile built without holders and contexts, to
#                                measure what keeping them gains
# Automatic checks: each sentence cites a source that exists (factual support); a cited opinion of someone else names
# that person, and the reviewer's own opinion names no one else (attribution); a cited occasion or condition is kept
# (conditions); every number matches the record or the quote.

for required in ("facts", "recipes", "catalog", "reviewed", "parse_request", "satisfies", "score_candidates",
                 "build_profiles", "llm"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py to task-8.py in this session first.")

N_MEMBERS = 15     # members summarised and given a recommendation explanation
MAX_FACTS = 4
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

# ==========================================
# 1. CITABLE FACTS
# ==========================================
profiles9 = build_profiles(facts)
# The same facts with holders and contexts ignored, as in Task 7's check: what a context-blind system would cite. Its
# explanations are checked against the holder, occasion and condition each quoted opinion really has.
blind_profiles9 = build_profiles(facts.assign(holder="reviewer", scope="general"))
TRUE_LABELS = {(None if pd.isna(r.review_id) else int(r.review_id), r.clause, int(r.start), r.sentiment):
               (r.holder, r.scope, r.condition) for r in facts.itertuples()}
HOLDER_PHRASE = {"reviewer": "You", "children": "Your children", "partner": "Your partner", "family": "Your household",
                 "guests": "Your guests"}
SCOPE_PHRASE = {"weekday": " on weekdays", "weekend": " at weekends", "holiday": " for holidays",
                "gathering": " for gatherings", "special occasion": " for special occasions"}
ASPECT_PHRASE = {  # (likes, dislikes)
    "TASTE": ("enjoyed the taste", "disliked the taste"),
    "TEXTURE": ("enjoyed the texture", "disliked the texture"),
    "PREP_EFFORT": ("valued easy recipes", "found demanding preparation a problem"),
    "TIME": ("valued quick recipes", "found lengthy preparation a problem"),
    "INGREDIENT_AVAILABILITY": ("liked using ingredients on hand", "had trouble getting ingredients"),
    "HEALTHINESS": ("valued healthy recipes", "found heavy recipes a problem"),
    "SUITABILITY": ("found recipes a good fit", "found recipes a poor fit"),
}


def member_facts(member, holder=None, scope=None, profiles=None):
    """Up to MAX_FACTS profile entries (likes or dislikes) as facts E1, E2, ...: the requested holder and occasion first
    (or, with none requested, the entries outside the general profile), then stable general entries."""
    profiles = profiles9 if profiles is None else profiles
    e = profiles[(profiles["member_id"] == member) & profiles["preference"].isin(["likes", "dislikes"])].copy()
    if holder:
        e["first"] = (e["holder"] == holder) & (e["scope"] == scope)
    else:
        e["first"] = e["profile"] == "context"
    e = e.sort_values(["first", "status", "recipes"], ascending=[False, True, False]).head(MAX_FACTS)  # "stable" first
    out = []
    for i, r in enumerate(e.itertuples(), 1):
        evidence = json.loads(r.evidence)[0]
        quote = re.sub(r"\s*[.!?]+\s*", ", ", evidence["clause"]).strip(", ")  # one quoted clause, no sentence breaks
        out.append({"id": f"E{i}", "holder": r.holder, "scope": r.scope, "aspect": r.aspect, "preference": r.preference,
                    "status": r.status, "recipes": int(r.recipes), "quote": quote, "condition": evidence["condition"],
                    "review_id": evidence["review_id"],
                    "key": [evidence["review_id"], evidence["clause"], evidence["start"], evidence["sentiment"]]})
    return out


def true_labels(fs):
    """Facts with the holder, occasion and condition their quoted opinion really has."""
    return [{**f, **dict(zip(("holder", "scope", "condition"), TRUE_LABELS[tuple(f["key"])]))} for f in fs]


def occasion(f):
    return f" when {f['condition']}" if f["scope"] == "conditional" and f["condition"] else SCOPE_PHRASE.get(f["scope"], "")


def fact_line(f):
    """One fact as given to the LLM."""
    return (f"[{f['id']}] holder: {f['holder']}; occasion or condition: {occasion(f).strip() or 'none'}; "
            f"aspect: {f['aspect']}; opinion: {f['preference']} ({'in ' + str(f['recipes']) + ' recipes' if f['status'] == 'stable' else 'once'}); "
            f"quote: \"{f['quote']}\"")


# ==========================================
# 2. SUMMARIES AND EXPLANATIONS
# ==========================================
def extractive_summary(fs):
    """Quoted evidence, each labelled with its holder and occasion."""
    return " ".join(f"[{f['holder']}{', ' + occasion(f).strip() if occasion(f) else ''}] \"{f['quote']}\" [{f['id']}]."
                    for f in fs)


def claim(f):
    how_often = f"across {f['recipes']} recipes" if f["status"] == "stable" else "for one recipe"
    phrase = ASPECT_PHRASE[f["aspect"]][f["preference"] == "dislikes"]
    quote = f["quote"] if len(f["quote"]) <= 80 else f["quote"][:80].rsplit(" ", 1)[0] + "..."
    return f"{HOLDER_PHRASE[f['holder']]} {phrase}{occasion(f)} {how_often} (\"{quote}\") [{f['id']}]."


def minutes(recipe):
    return f"{recipe['duration']:.0f} minutes" if pd.notna(recipe["duration"]) else "no listed time"


def recipe_record(recipe):
    tags = [t for t in recipe["tag_list"] if t in ("kid-friendly", "main-dish", "30-minutes-or-less", "easy", "healthy", "weeknight")]
    return f"\"{recipe['title']}\": {minutes(recipe)}, {recipe['direction_size']:.0f} steps, tags: {', '.join(tags) or 'none'}"


def template_explanation(recipe, fs):
    """Claims built from the facts, then the checkable recipe record."""
    return " ".join([claim(f) for f in fs[:2]] +
                    [f"This recipe takes {minutes(recipe)} and has {recipe['direction_size']:.0f} steps [R]."])


def generate(prompt):
    """Greedy generation with the Task 5 LLM."""
    text = llm_tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    batch = llm_tokenizer([text], return_tensors="pt").to(llm_device)
    with torch.no_grad():
        output = llm.generate(**batch, max_new_tokens=160, do_sample=False)
    return llm_tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip()


RULES = ("Address the member as \"you\". For each claim, say whose preference it is (you, your children, your partner, "
         "your household or your guests) and keep its occasion or condition. Use only the facts given; do not add "
         "details or numbers that are not in them. End every sentence with the ids of the facts it uses in square "
         "brackets, like [E1]")


def generated_summary(fs):
    return generate("Facts about a Food.com member, from their earlier reviews:\n" + "\n".join(map(fact_line, fs)) +
                    f"\n\nSummarise this member's food preferences in 2 or 3 short sentences. {RULES}.")


def generated_explanation(request_text, recipe, fs):
    return generate(f"Request: \"{request_text}\"\nRecipe record [R]: {recipe_record(recipe)}\n"
                    "Facts about the member, from their earlier reviews of other recipes:\n" + "\n".join(map(fact_line, fs)) +
                    f"\n\nExplain in 2 short sentences why this recipe suits the request. {RULES}, or [R] for the recipe "
                    "record. The facts are about other recipes: do not say how this recipe tastes or that anyone will "
                    "like it; only connect the member's past opinions to what the recipe record shows.")


# ==========================================
# 3. CHECKS
# ==========================================
HOLDER_RE = {"children": r"\b(?:child|children|kids?|kiddos?|sons?|daughters?|toddlers?|teens?|grandkids)\b",
             "partner": r"\b(?:partner|husband|wife|hubby|boyfriend|girlfriend|spouse)\b",
             "family": r"\b(?:family|household)\b", "guests": r"\b(?:guests?|friends|company)\b"}
SCOPE_RE = {"weekday": r"week ?days?|week ?nights?|school nights?|work nights?", "weekend": r"weekends?",
            "holiday": r"holidays?|christmas|thanksgiving|easter|halloween|valentine",
            "gathering": r"gatherings?|part(?:y|ies)|potlucks?|barbecues?|bbq|picnics?|cookouts?|game day",
            "special occasion": r"special occasions?|birthdays?|anniversar(?:y|ies)|date night"}


def check(text, fs, recipe=None, request_holder=None):
    """Per-sentence checks of one output against its facts (and recipe record). A person named in a sentence must be
    the holder of a fact it cites, or the person the request is for; a cited opinion of someone else must name them;
    and a sentence about "you" must not rest only on someone else's opinions."""
    by_id = {f["id"]: f for f in fs}
    allowed = {n for f in fs for n in re.findall(r"\d+", f["quote"]) + [str(f["recipes"])]}
    if recipe is not None:
        allowed |= set(re.findall(r"\d+", f"{minutes(recipe)} {recipe['direction_size']:.0f} {recipe['title']}"))
    result = {"sentences": 0, "cited": 0, "attribution": [], "conditions": [], "unsupported numbers": 0,
              "words": len(text.split())}
    for s in sent_tokenize(text):
        result["sentences"] += 1
        ids = [i for group in re.findall(r"\[([^\]]+)\]", s) for i in re.findall(r"E\d+|\bR\b", group)]
        valid = [i for i in ids if i in by_id or (i == "R" and recipe is not None)]
        result["cited"] += bool(valid) and len(valid) == len(ids)
        plain = re.sub(r"[\"“][^\"”]*[\"”]", "", s).lower()  # the sentence without its quotes
        cited = [by_id[i] for i in valid if i != "R"]
        supported = {f["holder"] for f in cited} | {request_holder}
        result["attribution"] += [False for h in HOLDER_RE if h not in supported and re.search(HOLDER_RE[h], plain)]
        if cited and all(f["holder"] != "reviewer" for f in cited) and re.match(r"you\b", plain.strip()):
            result["attribution"].append(False)  # "You dislike ..." citing only the children's opinion
        for f in cited:  # an opinion of an unknown person cannot be credited to anyone named or to "you"
            result["attribution"].append(f["holder"] == "reviewer" or
                                         (f["holder"] in HOLDER_RE and bool(re.search(HOLDER_RE[f["holder"]], plain))))
            if f["scope"] == "conditional":
                result["conditions"].append(any(w in plain for w in re.findall(r"[a-z]{4,}", f["condition"])))
            elif f["scope"] in SCOPE_RE:
                result["conditions"].append(bool(re.search(SCOPE_RE[f["scope"]], plain)))
        result["unsupported numbers"] += sum(n not in allowed for n in re.findall(r"\b\d+\b", plain))
    return result


# ==========================================
# 4. RUN AND EVALUATE
# ==========================================
REQUEST_FOR = {"children": "Recommend a quick dinner for my children.", "partner": "Recommend a quick dinner for my partner.",
               "family": "Recommend a quick dinner for my family.", "guests": "Recommend a dish for guests.",
               "weekday": "Recommend a quick weeknight dinner.", "weekend": "Recommend a weekend dinner.",
               "holiday": "Recommend a dish for Christmas.", "gathering": "Recommend a dish for a party.",
               "special occasion": "Recommend a dish for a birthday."}


def request_for(fs):
    """A request matching the member's first fact outside the general profile."""
    for f in fs:
        key = f["holder"] if f["holder"] != "reviewer" else f["scope"]
        if key in REQUEST_FOR:
            return REQUEST_FOR[key]
    return "Recommend a quick dinner."


# Members with the most entries outside their general profile, so that attribution and conditions are tested.
counts = profiles9[profiles9["preference"].isin(["likes", "dislikes"])].groupby("member_id")["profile"].agg(
    lambda p: (p == "context").sum() * 100 + len(p))
members9 = counts.sort_values(ascending=False).index[:N_MEMBERS]
if llm is None:
    print("The Task 5 LLM is not loaded: generated summaries and explanations are skipped.")

outputs = []
for member in members9:
    fs = member_facts(member)
    texts = {"extractive summary": extractive_summary(fs)}
    request_text = request_for(fs)
    request = parse_request(request_text)
    pool = catalog[satisfies(catalog, request)]
    pool = pool[~pool.isin(list(reviewed.get(member, set())))]
    scores = score_candidates(member, pool, request, facts[facts["member_id"] == member])
    recipe = recipes.loc[pool[int(np.argmax(scores["total"]))]]
    efs = member_facts(member, request["holder"], request["scope"])
    texts["template explanation"] = template_explanation(recipe, efs)
    bfs = member_facts(member, request["holder"], request["scope"], profiles=blind_profiles9)
    texts["template explanation, context-blind"] = template_explanation(recipe, bfs)
    if llm is not None:
        texts["generated summary"] = generated_summary(fs)
        texts["generated explanation"] = generated_explanation(request_text, recipe, efs)
    for kind, text in texts.items():
        is_explanation = "explanation" in kind
        used = true_labels(bfs) if kind.endswith("context-blind") else efs if is_explanation else fs
        outputs.append({"member_id": member, "output": kind, "request": request_text if is_explanation else "",
                        "recipe": recipe["title"] if is_explanation else "", "text": text, "facts": json.dumps(used),
                        **check(text, used, recipe if is_explanation else None,
                                request["holder"] if is_explanation else None)})
df_out = pd.DataFrame(outputs)

summary = df_out.groupby("output", sort=False).agg(
    outputs=("text", "size"), words=("words", "mean"),
    cited=("cited", "sum"), sentences=("sentences", "sum"),
    attribution=("attribution", lambda x: np.mean([v for l in x for v in l]) if any(len(l) for l in x) else np.nan),
    attribution_n=("attribution", lambda x: sum(len(l) for l in x)),
    conditions=("conditions", lambda x: np.mean([v for l in x for v in l]) if any(len(l) for l in x) else np.nan),
    conditions_n=("conditions", lambda x: sum(len(l) for l in x)),
    unsupported_numbers=("unsupported numbers", "sum"))
summary.insert(2, "sentences citing a source", (summary.pop("cited") / summary.pop("sentences")).round(3))
print(f"--- Checks on {len(members9)} members (attribution and conditions over cited opinions; n = cited opinions) ---")
with pd.option_context("display.width", 200, "display.max_columns", None):
    print(summary.round(3))

example = df_out[df_out["member_id"] == members9[0]]
print(f"\n--- Member {members9[0]} ---")
for r in example.itertuples():
    print(f"{r.output}{' for ' + repr(r.request) + ' -> ' + r.recipe if r.request else ''}:\n   {r.text}\n")

df_out.drop(columns=["attribution", "conditions"]).to_csv(os.path.join(OUT_DIR, "task9_outputs.csv"), index=False)
print(f"Saved {len(df_out)} summaries and explanations with their checks in {OUT_DIR}.")
