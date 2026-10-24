# Imports already made by task-1.py (same notebook): re, plt, nltk, pd, sns.
import ast
import json
import os
import random
from collections import Counter

import spacy
from spacy.matcher import PhraseMatcher
from spacy.training import Example
from spacy.util import compounding, minibatch

# Task 2: Named Entity Recognition for Food and Context
# Run load-dataset.py and task-1.py first (this needs df_recipes and df_clauses).
# Kaggle: spaCy is preinstalled; en_core_web_sm (pretrained, fine-tuned here) is downloaded if missing, so keep internet on.

for required in ("df_recipes", "df_clauses"):
    if required not in globals():
        raise NameError(f"{required} is missing: run load-dataset.py and task-1.py in this session first.")

SEED = 42
N_LEXICON_RECIPES = 100_000  # recipes sampled to build the ingredient and dish lexicons
N_TRAIN_CLAUSES = 20_000     # dictionary-labelled clauses used to train the NER model
N_EPOCHS = 8
N_APPLY_CLAUSES = 20_000     # clauses used for the frequency analysis
OUT_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."

random.seed(SEED)
spacy.util.fix_random_seed(SEED)

LABELS = ["INGREDIENT", "DISH", "COOKING_EQUIPMENT", "DURATION", "OCCASION", "DINER_GROUP"]
# Between overlapping matches of equal length, the label listed first wins.
PRIORITY = {l: i for i, l in enumerate(["DISH", "COOKING_EQUIPMENT", "OCCASION", "DINER_GROUP", "DURATION", "INGREDIENT"])}


def terms(csv):
    return {t.strip() for t in csv.replace("\n", " ").split(",") if t.strip()}


# ==========================================
# 1. LEXICONS
# ==========================================
SEED_INGREDIENTS = terms("""chicken, beef, pork, lamb, turkey, bacon, sausage, ham, shrimp, salmon, tuna, egg, eggs, butter,
milk, cream, cheese, parmesan, cheddar, mozzarella, yogurt, sour cream, flour, sugar, brown sugar, honey, salt, pepper, garlic,
onion, onions, shallot, ginger, tomato, tomatoes, potato, potatoes, carrot, carrots, celery, mushroom, mushrooms, spinach,
broccoli, cauliflower, zucchini, cabbage, cucumber, lettuce, corn, peas, beans, black beans, chickpeas, lentils, rice, pasta,
noodles, breadcrumbs, oats, olive oil, vegetable oil, vinegar, soy sauce, lemon, lime, orange, apple, banana, strawberries,
blueberries, cinnamon, vanilla, cumin, paprika, oregano, basil, parsley, cilantro, thyme, rosemary, chili powder, cayenne,
jalapeno, coconut milk, peanut butter, chocolate, cocoa, walnuts, almonds, pecans, cornstarch, baking powder, baking soda,
yeast, tofu, avocado, mayonnaise, mustard, ketchup, salsa, broth, stock, wine, egg whites, butternut squash""")

DISH_NOUNS = terms("""soup, stew, casserole, lasagna, pizza, curry, chili, cake, cookies, pie, pasta, salad, sandwich, burger,
burgers, tacos, enchiladas, quiche, pancakes, waffles, muffins, brownies, bread, meatloaf, meatballs, risotto, stir fry,
pot roast, mac and cheese, cheesecake, omelet, burritos, fajitas, gumbo, jambalaya, lo mein, dumplings, sushi, pudding,
cupcakes, scones, biscuits, fudge, smoothie, lemonade, chowder, roast, kebabs, fritters, frittata, pancake, cookie""")

EQUIPMENT = terms("""oven, slow cooker, crock pot, crockpot, instant pot, pressure cooker, skillet, frying pan, fry pan, dutch oven,
grill, microwave, blender, food processor, stand mixer, hand mixer, mixer, air fryer, rice cooker, baking sheet, cookie sheet,
saucepan, sauce pan, stockpot, stock pot, wok, casserole dish, baking dish, baking pan, cake pan, muffin tin, loaf pan,
pie plate, whisk, colander, cast iron, griddle, broiler, toaster oven, deep fryer, smoker, roasting pan, steamer, bundt pan,
springform pan, mandoline, rolling pin, thermometer, immersion blender""")

OCCASIONS = terms("""weeknight, weekday, weekend, school night, work night, special occasion, holiday, christmas, thanksgiving,
easter, birthday, anniversary, dinner party, party, potluck, bbq, barbecue, cookout, picnic, game day, super bowl, new year's eve,
valentine's day, halloween, hanukkah, lunchbox, tailgate, wedding, baby shower, cocktail party, sunday dinner, date night""")

DINER_GROUPS = terms("""children, child, kids, kid, kiddos, toddler, toddlers, baby, teenagers, teens, family, husband, wife, hubby,
partner, boyfriend, girlfriend, fiance, guests, guest, company, friends, grandkids, grandchildren, parents, mother, father,
mother-in-law, father-in-law, in-laws, picky eaters, picky eater, crowd, daughter, son, dh, dd, ds, boys, girls, roommates,
coworkers, neighbors""")


def plural_variants(term):
    if term.endswith("y") and not term.endswith(("ay", "ey", "oy", "uy")):
        return {term, term[:-1] + "ies"}
    if term.endswith(("s", "x", "ch", "sh")):
        return {term, term + "es"}
    return {term, term + "s"}


def parse_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError, TypeError):
        return []


WORD_OK = re.compile(r"^[a-z][a-z \-']{2,39}$")
recipes_sample = df_recipes.sample(min(N_LEXICON_RECIPES, len(df_recipes)), random_state=SEED)

# Ingredients: frequent FoodKG-normalised ingredient names from the recipe table.
ingredient_lexicon = set(SEED_INGREDIENTS)
if "ingredient_food_kg_names" in recipes_sample.columns:
    counts = Counter()
    for value in recipes_sample["ingredient_food_kg_names"].dropna():
        counts.update({str(n).lower().strip() for n in parse_list(value)})
    ingredient_lexicon |= {n for n, c in counts.items() if c >= 30 and WORD_OK.match(n) and len(n.split()) <= 3}

# Dishes: dish nouns plus the up-to-two-word phrase before them in recipe titles ("chicken soup").
TITLE_MODIFIERS = {"the", "a", "best", "easy", "quick", "simple", "super", "my", "our", "favorite", "perfect", "classic",
                   "ultimate", "great", "good", "homemade", "basic", "old", "fashioned", "style", "delicious", "awesome",
                   "amazing", "new", "fast", "healthy", "low", "fat", "free", "and", "with", "i", "mom's", "grandma's"}
dish_lexicon = set(DISH_NOUNS)
if "title" in recipes_sample.columns:
    single_nouns = {n for n in DISH_NOUNS if " " not in n}
    phrase_counts = Counter()
    for title in recipes_sample["title"].dropna().str.lower().str.replace(r"[^a-z' \-]", " ", regex=True):
        tokens = title.split()
        if len(tokens) >= 2 and tokens[-1] in single_nouns and tokens[-2] not in TITLE_MODIFIERS and len(tokens[-2]) > 2:
            phrase_counts[" ".join(tokens[-2:])] += 1
    dish_lexicon |= {p for p, c in phrase_counts.items() if c >= 5}

LEXICONS = {
    "INGREDIENT": {v for t in ingredient_lexicon for v in plural_variants(t)},
    "DISH": {v for t in dish_lexicon for v in plural_variants(t)},
    "COOKING_EQUIPMENT": {v for t in EQUIPMENT for v in plural_variants(t)},
    "OCCASION": {v for t in OCCASIONS for v in plural_variants(t)},
    "DINER_GROUP": {v for t in DINER_GROUPS for v in plural_variants(t)},
}
print({label: len(entries) for label, entries in LEXICONS.items()})

# ==========================================
# 2. DICTIONARY-BASED NER
# ==========================================
nlp_tok = spacy.blank("en")  # tokenizer only; the dictionary extractor needs no model
matcher = PhraseMatcher(nlp_tok.vocab, attr="LOWER")
for label, entries in LEXICONS.items():
    matcher.add(label, list(nlp_tok.tokenizer.pipe(sorted(entries))))

_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?|an?|one|two|three|four|five|six|seven|eight|nine|ten|twelve|couple of|few)"
_UNIT = r"(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?)"
DURATION_RE = re.compile(
    rf"\b(?:half an hour|overnight|{_NUM}(?:\s*(?:-|to|or)\s*{_NUM})?(?:\s+and\s+a\s+half)?\s*-?\s*{_UNIT})\b", re.IGNORECASE
)


def resolve_overlaps(candidates):
    """Greedy: longest span first, then label priority, then leftmost."""
    chosen = []
    for start, end, label in sorted(candidates, key=lambda c: (-(c[1] - c[0]), PRIORITY[c[2]], c[0])):
        if all(end <= s or start >= e for s, e, _ in chosen):
            chosen.append((start, end, label))
    return sorted(chosen)


def dictionary_ner(texts):
    results = []
    for doc in nlp_tok.pipe(texts, batch_size=1000):
        candidates = []
        for match_id, start, end in matcher(doc):
            candidates.append((doc[start].idx, doc[end - 1].idx + len(doc[end - 1]), nlp_tok.vocab.strings[match_id]))
        for m in DURATION_RE.finditer(doc.text):
            span = doc.char_span(m.start(), m.end(), alignment_mode="contract")
            if span is not None:
                candidates.append((span.start_char, span.end_char, "DURATION"))
        results.append(resolve_overlaps(candidates))
    return results


# ==========================================
# 3. EVALUATION SET (small, hand-written pilot gold)
# ==========================================
# Markup: [[span|LABEL]]. This pilot set is independent of the lexicons but small; extend it with the Task 10 annotations.
GOLD_MARKUP = [
    "Made this on a [[weeknight|OCCASION]] and it took only [[30 minutes|DURATION]].",
    "My [[kids|DINER_GROUP]] loved the [[chicken|INGREDIENT]] but it was too spicy for my [[husband|DINER_GROUP]].",
    "Perfect [[lasagna|DISH]] for [[Christmas|OCCASION]] [[guests|DINER_GROUP]]!",
    "Baked it in my [[Dutch oven|COOKING_EQUIPMENT]] for [[2 hours|DURATION]] and the [[beef|INGREDIENT]] was tender.",
    "I used [[coconut milk|INGREDIENT]] instead of [[cream|INGREDIENT]] and a pinch of [[sumac|INGREDIENT]].",
    "Great [[chili|DISH]] for a [[potluck|OCCASION]], the whole [[family|DINER_GROUP]] asked for seconds.",
    "Too time-consuming for [[weekdays|OCCASION]], but fine for the [[weekend|OCCASION]].",
    "Cooked the [[onions|INGREDIENT]] and [[garlic|INGREDIENT]] in a large [[skillet|COOKING_EQUIPMENT]] for [[5 minutes|DURATION]].",
    "My [[toddler|DINER_GROUP]] ate two helpings of this [[mac and cheese|DISH]].",
    "I made [[harissa|INGREDIENT]] [[chicken|INGREDIENT]] in the [[air fryer|COOKING_EQUIPMENT]].",
    "Served this [[pavlova|DISH]] at our [[anniversary|OCCASION]] dinner.",
    "Let the [[pork|INGREDIENT]] marinate [[overnight|DURATION]].",
    "My [[picky eaters|DINER_GROUP]] actually cleaned their plates.",
    "Used my [[stand mixer|COOKING_EQUIPMENT]] to whip the [[egg whites|INGREDIENT]].",
    "This [[tagine|DISH]] was a hit with my [[in-laws|DINER_GROUP]].",
    "Ready in [[20 minutes|DURATION]], which makes it ideal for a busy [[weeknight|OCCASION]].",
    "I baked these [[cookies|DISH]] for a [[birthday|OCCASION]].",
    "Cooked it in the [[crock pot|COOKING_EQUIPMENT]] on low for [[8 hours|DURATION]].",
    "My [[husband|DINER_GROUP]] and [[kids|DINER_GROUP]] devoured the [[pot roast|DISH]].",
    "Not enough [[salt|INGREDIENT]] for my taste, so I added extra [[cumin|INGREDIENT]].",
    "Easy [[stir fry|DISH]] for [[school nights|OCCASION]] when you only have [[15 minutes|DURATION]].",
    "I used a [[springform pan|COOKING_EQUIPMENT]] and the [[cheesecake|DISH]] came out perfect.",
    "Took [[1 1/2 hours|DURATION]] total, much longer than expected.",
    "Served it to [[guests|DINER_GROUP]] on [[Thanksgiving|OCCASION]] and everyone wanted the recipe.",
    "Added [[tamarind paste|INGREDIENT]] for a sour kick, delicious with [[rice|INGREDIENT]].",
    "Cooked [[30-45 minutes|DURATION]] on the [[grill|COOKING_EQUIPMENT]].",
    "I made [[chicken soup|DISH]] when my [[mother-in-law|DINER_GROUP]] was sick.",
    "This is going into my [[holiday|OCCASION]] rotation.",
    "No [[flour|INGREDIENT]] needed, just [[almond meal|INGREDIENT]] and [[eggs|INGREDIENT]].",
    "Four stars, I would make it again.",
    "Thanks for sharing!",
    "Next time I will use less [[sugar|INGREDIENT]].",
    "The [[mandoline|COOKING_EQUIPMENT]] made slicing the [[potatoes|INGREDIENT]] quick.",
    "Our [[teenagers|DINER_GROUP]] requested this [[pizza|DISH]] again for [[game day|OCCASION]].",
    "Simmer the [[tomatoes|INGREDIENT]] for [[10 minutes|DURATION]] in a [[saucepan|COOKING_EQUIPMENT]].",
    "Perfect for [[Easter|OCCASION]].",
    "I halved it for just the two of us.",
    "My [[grandkids|DINER_GROUP]] love these [[pancakes|DISH]].",
    "Cut the [[butternut squash|INGREDIENT]] into cubes and roast in the [[oven|COOKING_EQUIPMENT]] for [[25 minutes|DURATION]].",
    "The [[fenugreek|INGREDIENT]] gave it a bitter edge my [[wife|DINER_GROUP]] disliked.",
    "A [[panforte|DISH]] like this is perfect for a [[Christmas|OCCASION]] [[party|OCCASION]].",
]
MARKUP_RE = re.compile(r"\[\[(.+?)\|([A-Z_]+)\]\]")


def parse_markup(markup):
    text, entities, last = "", [], 0
    for m in MARKUP_RE.finditer(markup):
        text += markup[last:m.start()]
        entities.append((len(text), len(text) + len(m.group(1)), m.group(2)))
        text += m.group(1)
        last = m.end()
    return text + markup[last:], entities


gold = [parse_markup(m) for m in GOLD_MARKUP]
gold_texts = [t for t, _ in gold]
gold_sets = [set(e) for _, e in gold]


def entity_prf(gold_sets, pred_sets):
    """Entity-level exact-match precision, recall and F1 per label and micro-averaged."""
    rows = []
    for label in LABELS + ["MICRO"]:
        tp = fp = fn = 0
        for g, p in zip(gold_sets, pred_sets):
            g = {e for e in g if label in (e[2], "MICRO")}
            p = {e for e in p if label in (e[2], "MICRO")}
            tp, fp, fn = tp + len(g & p), fp + len(p - g), fn + len(g - p)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append({"label": label, "support": tp + fn, "precision": prec, "recall": rec, "f1": f1})
    return pd.DataFrame(rows).set_index("label")


dict_gold_preds = [set(p) for p in dictionary_ner(gold_texts)]

# ==========================================
# 4. DOMAIN-ADAPTED NER MODEL (weak supervision from the dictionary)
# ==========================================
# The pretrained en_core_web_sm NER, fine-tuned on dictionary-labelled review clauses with the six project labels, so it
# can generalise to unseen terms from context and word shape. A blank model trained on the same labels shows what the
# pretraining adds.
# The 200 pilot reviews (section 7) are held out so their corrected labels can later serve as unseen gold.
pilot_rows = sorted(df_clauses["review_row"].drop_duplicates().sample(200, random_state=SEED))
clause_texts = df_clauses.loc[~df_clauses["review_row"].isin(pilot_rows), "sentences"].astype(str)
clause_texts = clause_texts[clause_texts.str.split().str.len().between(3, 60)]
train_sample = clause_texts.sample(min(N_TRAIN_CLAUSES, len(clause_texts)), random_state=SEED)
train_texts = train_sample.tolist()
train_ents = dictionary_ner(train_texts)

def load_pretrained(**kwargs):
    """The pretrained general-purpose spaCy pipeline (downloaded when missing)."""
    try:
        return spacy.load("en_core_web_sm", **kwargs)
    except OSError:
        spacy.cli.download("en_core_web_sm")
        return spacy.load("en_core_web_sm", **kwargs)


def train_ner(texts, entity_lists, pretrained, epochs=N_EPOCHS):
    """A NER model with the six project labels trained on (start, end, label) spans. pretrained=True fine-tunes the
    pretrained en_core_web_sm NER (domain adaptation: it keeps what it knows of English and learns the new labels);
    False trains a blank model from scratch. Spans that do not fall on token boundaries are dropped."""
    random.seed(SEED)
    spacy.util.fix_random_seed(SEED)
    nlp = load_pretrained(exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"]) if pretrained else spacy.blank("en")
    ner = nlp.get_pipe("ner") if pretrained else nlp.add_pipe("ner")
    for label in LABELS:
        ner.add_label(label)
    examples = []
    for text, ents in zip(texts, entity_lists):
        doc = nlp.make_doc(text)
        examples.append(Example.from_dict(doc, {"entities": [(s, e, l) for s, e, l in ents if doc.char_span(s, e)]}))
    optimizer = nlp.resume_training() if pretrained else nlp.initialize(lambda: examples)
    for epoch in range(epochs):
        random.shuffle(examples)
        losses = {}
        for batch in minibatch(examples, size=compounding(4.0, 32.0, 1.001)):
            nlp.update(batch, sgd=optimizer, drop=0.2, losses=losses)
        print(f"  epoch {epoch + 1}/{epochs}  ner loss: {losses['ner']:.1f}")
    return nlp


print("Fine-tuning the pretrained en_core_web_sm NER on the dictionary labels:")
nlp_ner = train_ner(train_texts, train_ents, pretrained=True)
nlp_ner.to_disk(os.path.join(OUT_DIR, "hummus_ner_model"))
print("Training a blank NER model on the same labels, for comparison:")
nlp_scratch = train_ner(train_texts, train_ents, pretrained=False)


def model_ner(nlp, texts, label_map=None):
    """(start, end, label) per text. Without a label_map only the six project labels are kept: the pretrained model's
    own labels (PERSON, ORG, ...) are dropped."""
    out = []
    for doc in nlp.pipe(texts, batch_size=256):
        ents = set()
        for e in doc.ents:
            label = label_map.get(e.label_) if label_map else (e.label_ if e.label_ in LABELS else None)
            if label:
                ents.add((e.start_char, e.end_char, label))
        out.append(ents)
    return out


model_gold_preds = model_ner(nlp_ner, gold_texts)
scratch_gold_preds = model_ner(nlp_scratch, gold_texts)

# General-purpose baseline: the pretrained spaCy NER as it comes, mapped onto the project labels. Only TIME, DATE and
# EVENT have a rough counterpart; there is no label for ingredients, dishes, equipment or diner groups.
nlp_general = load_pretrained(exclude=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
GENERAL_MAP = {"TIME": "DURATION", "DATE": "OCCASION", "EVENT": "OCCASION"}
general_preds = model_ner(nlp_general, gold_texts, GENERAL_MAP)

# ==========================================
# 5. COMPARISON
# ==========================================
results = {"dictionary": entity_prf(gold_sets, dict_gold_preds), "domain model": entity_prf(gold_sets, model_gold_preds),
           "from scratch": entity_prf(gold_sets, scratch_gold_preds), "general spaCy": entity_prf(gold_sets, general_preds)}

comparison = pd.concat(results, axis=1)
print(f"\n--- Entity-level P/R/F1 on {len(gold_texts)} hand-written gold sentences ---")
with pd.option_context("display.width", 250, "display.max_columns", None):
    print(comparison.round(3))

f1_table = pd.DataFrame({name: df["f1"] for name, df in results.items()}).reset_index().melt(
    id_vars="label", var_name="system", value_name="F1"
)
plt.figure(figsize=(11, 5))
sns.barplot(data=f1_table, x="label", y="F1", hue="system")
plt.title("Entity-level F1 by label (pilot gold set)")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

# Error inspection: what each system misses or invents on the gold set.
for name, preds in (("dictionary", dict_gold_preds), ("domain model", model_gold_preds)):
    print(f"\n--- {name}: sample errors ---")
    shown = 0
    for text, g, p in zip(gold_texts, gold_sets, preds):
        for s, e, l in sorted(g - p):
            print(f"MISSED   {l:<18} {text[s:e]!r}")
            shown += 1
        for s, e, l in sorted(p - g):
            print(f"SPURIOUS {l:<18} {text[s:e]!r}")
            shown += 1
        if shown >= 25:
            break

# ==========================================
# 6. ENTITY FREQUENCIES ON REVIEW CLAUSES
# ==========================================
# Clauses not used for training, so the model's counts reflect unseen text.
unseen_idx = clause_texts.index.difference(train_sample.index)
df_apply = df_clauses.loc[unseen_idx].sample(min(N_APPLY_CLAUSES, len(unseen_idx)), random_state=SEED)
apply_texts = df_apply["sentences"].astype(str).tolist()
dict_ents = dictionary_ner(apply_texts)
model_ents = model_ner(nlp_ner, apply_texts)

id_cols = [c for c in ("review_row", "clause_idx", "member_id", "recipe_id") if c in df_apply.columns]
rows = []
for (_, meta), text, ents in zip(df_apply[id_cols].iterrows(), apply_texts, model_ents):
    for s, e, label in sorted(ents):
        rows.append({**meta.to_dict(), "clause": text, "start": s, "end": e, "text": text[s:e], "label": label})
df_entities = pd.DataFrame(rows, columns=id_cols + ["clause", "start", "end", "text", "label"])  # reused by later tasks
df_entities.to_csv(os.path.join(OUT_DIR, "task2_entities.csv"), index=False)

freq = pd.DataFrame({
    "dictionary": Counter(l for ents in dict_ents for _, _, l in ents),
    "domain model": Counter(l for ents in model_ents for _, _, l in ents),
}).reindex(LABELS).fillna(0).astype(int)
print(f"\n--- Entity counts over {len(apply_texts)} clauses ---")
print(freq)
freq.plot(kind="bar", figsize=(10, 5), title="Entities found per label")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.show()

print("\n--- Most frequent entities per label (domain model) ---")
for label in LABELS:
    top = df_entities.loc[df_entities["label"] == label, "text"].str.lower().value_counts().head(12)
    print(f"{label}: { {k: int(v) for k, v in top.items()} }")

# Terms the model tags that the dictionary does not contain: candidates for lexicon expansion.
print("\n--- Model-only entity candidates (not in any lexicon) ---")
known = {t.lower() for entries in LEXICONS.values() for t in entries}
novel = df_entities[~df_entities["text"].str.lower().isin(known) & (df_entities["label"] != "DURATION")]
for label, group in novel.groupby("label"):
    print(f"{label}: { {k: int(v) for k, v in group['text'].str.lower().value_counts().head(10).items()} }")

# ==========================================
# 7. PILOT ANNOTATION SHEET (200 reviews)
# ==========================================
pilot = df_clauses[df_clauses["review_row"].isin(pilot_rows)].copy()
pilot_ents = dictionary_ner(pilot["sentences"].astype(str).tolist())
pilot["predicted_entities"] = [json.dumps([[s, e, l] for s, e, l in ents]) for ents in pilot_ents]
pilot["corrected_entities"] = ""  # fill in by hand: [[start, end, LABEL], ...]
pilot.to_csv(os.path.join(OUT_DIR, "task2_pilot_annotation.csv"), index=False)
print(f"\nSaved {len(pilot)} clauses from 200 reviews for manual correction in {OUT_DIR}.")
