import re
import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

print("==================================================")
print(" TASK 2: Named Entity Recognition (NER) ")
print("==================================================")

# 1. Load Reviews Dataset
print("\n[1/4] Loading Reviews Dataset...")
df_reviews = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "raiyanahmed/hummus",
    "pp_reviews.csv",
    pandas_kwargs={"low_memory": False}
)

TEXT_COL = "text"
df_reviews = df_reviews.dropna(subset=[TEXT_COL]).copy()
df_reviews[TEXT_COL] = df_reviews[TEXT_COL].astype(str).str.strip()

# ----------------------------------------------------
# 2. DICTIONARY-BASED NER MATCHING
# ----------------------------------------------------
print("\n[2/4] Setting up Dictionary-Based NER...")
FOOD_DICTIONARY = {
    "INGREDIENT_DISH": ["garlic", "onion", "olive oil", "chicken", "spicy food", "hummus", "salt", "pepper", "butter"],
    "COOKING_EQUIPMENT": ["slow cooker", "instant pot", "oven", "skillet", "pan", "pot", "air fryer", "blender"],
    "OCCASION": ["weekend dinners", "weeknights", "weekdays", "christmas", "thanksgiving", "birthday", "special occasion", "holidays"],
    "DINER_GROUP": ["children", "kids", "kiddos", "toddlers", "husband", "wife", "guests", "family", "friends", "partner"],
    "DURATION": ["20 minutes", "30 mins", "1 hour", "quick"]
}

def dictionary_ner(text):
    entities = []
    for label, terms in FOOD_DICTIONARY.items():
        for term in terms:
            pattern = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                entities.append((match.group(), label, match.start(), match.end()))
    return entities

# ----------------------------------------------------
# 3. DOMAIN-ADAPTED SPACY NER MODEL
# ----------------------------------------------------
print("\n[3/4] Building Domain-Adapted SpaCy Model (Rule-Based Adapter)...")
nlp = spacy.load("en_core_web_sm")

# Add EntityRuler before standard NER to capture custom domains
ruler = nlp.add_pipe("entity_ruler", before="ner")

patterns = []
# Convert dictionary to SpaCy EntityRuler patterns
for label, terms in FOOD_DICTIONARY.items():
    for term in terms:
        patterns.append({"label": label, "pattern": term})

# Add custom durations pattern using regex in spaCy
patterns.extend([
    {"label": "DURATION", "pattern": [{"TEXT": {"REGEX": r"^\d+$"}}, {"LOWER": {"IN": ["mins", "minutes", "hours", "hrs", "min"]}}]},
    {"label": "DINER_GROUP", "pattern": [{"LOWER": {"IN": ["my", "our"]}}, {"LOWER": {"IN": ["children", "kids", "husband", "wife", "family", "guests"]}}]}
])

ruler.add_patterns(patterns)

def domain_adapted_ner(text):
    doc = nlp(text)
    return [(ent.text, ent.label_, ent.start_char, ent.end_char) for ent in doc.ents]

# ----------------------------------------------------
# 4. COMPARISON DEMONSTRATION & DATASET BATCH TEST
# ----------------------------------------------------
print("\n[4/4] Comparing Dictionary vs. Domain-Adapted NER on Sample Reviews...")

samples = [
    "I love spicy food, but this was too hot for my children. Great for weekend dinners, but too time-consuming for weekdays.",
    "Used a slow cooker for 30 mins, perfectly cooked for our guests on Christmas."
]

for i, text in enumerate(samples, 1):
    print(f"\n--- Sample Review {i} ---")
    print("Text:", text)
    print("\nDictionary-Based NER Results:")
    print(dictionary_ner(text))
    print("\nDomain-Adapted NER Results:")
    print(domain_adapted_ner(text))

# Sample 100 reviews from HUMMUS dataset to evaluate extraction
print("\n>>> Extracting Entities from 100 Dataset Reviews...")
sample_df = df_reviews.head(100).copy()
sample_df["dict_entities"] = sample_df[TEXT_COL].apply(dictionary_ner)
sample_df["domain_entities"] = sample_df[TEXT_COL].apply(domain_adapted_ner)

print("\nTask 2 Execution Completed Successfully!")
print("Sample extracted entities from Dataset:")
for idx, row in sample_df.head(3).iterrows():
    print(f"\nReview #{idx + 1}: {row[TEXT_COL][:80]}...")
    print("Extracted Entities (Domain-Adapted):", row["domain_entities"])