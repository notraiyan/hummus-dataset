import re
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.tokenize import sent_tokenize
from kagglehub import KaggleDatasetAdapter
import kagglehub

# Task 1: Exploratory Data Analysis and Text Preparation

for resource in ("punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)

# HUMMUS pp_reviews.csv stores the review body in `text`.
TEXT_COL = "text"
RATING_COL = "rating"
DATE_COL = "submit_date"


# 1. LOAD DATASETS

print("Loading datasets...")
df_members = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "raiyanahmed/hummus",
    "pp_members.csv",
    pandas_kwargs={"low_memory": False}
)

df_recipes = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "raiyanahmed/hummus",
    "pp_recipes.csv",
    pandas_kwargs={"low_memory": False}
)

df_reviews = kagglehub.dataset_load(
    KaggleDatasetAdapter.PANDAS,
    "raiyanahmed/hummus",
    "pp_reviews.csv",
    pandas_kwargs={"low_memory": False}
)

ID_COL = "review_id" if "review_id" in df_reviews.columns else None


# 2. MISSING VALUES & TEXT CLEANING

for name, df in [("Reviews", df_reviews), ("Members", df_members), ("Recipes", df_recipes)]:
    print(f"\n--- Missing Values in {name} ---")
    print(df.isnull().sum())

n_total = len(df_reviews)
df_reviews = df_reviews.dropna(subset=[TEXT_COL]).copy()
df_reviews[TEXT_COL] = df_reviews[TEXT_COL].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
df_reviews = df_reviews[df_reviews[TEXT_COL] != ""].copy()
print(f"\nReviews with text: {len(df_reviews)} / {n_total} ({len(df_reviews) / n_total:.1%})")


# 3. REVIEW LENGTH ANALYSIS

df_reviews["review_length"] = df_reviews[TEXT_COL].str.split().str.len()

print("\n--- Review Length Statistics (words) ---")
print(df_reviews["review_length"].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df_reviews["review_length"], bins=50, kde=True, color="teal")
plt.title("Distribution of Review Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.xlim(0, df_reviews["review_length"].quantile(0.95))  # cap at 95th percentile
plt.show()


# 4. RATING DISTRIBUTION

real_mask = df_reviews[RATING_COL].isin([1, 2, 3, 4, 5])
print("\n--- Rating Composition of Reviews With Text (%) ---")
print(df_reviews[RATING_COL].value_counts(normalize=True).sort_index() * 100)
print(f"\nImputed (2.5) share: {(df_reviews[RATING_COL] == 2.5).mean():.2%}")

print("\n--- Explicit Star Rating Distribution (%, 1-5 only) ---")
print(df_reviews.loc[real_mask, RATING_COL].value_counts(normalize=True).sort_index() * 100)

plt.figure(figsize=(8, 5))
sns.countplot(x=RATING_COL, data=df_reviews[real_mask], hue=RATING_COL, palette="viridis", legend=False)
plt.title("Distribution of Explicit Ratings (1-5)")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.show()


# 5. CONTEXTUAL EXPRESSIONS FREQUENCY

context_patterns = {
    "weeknight/weekday": r"\b(?:week ?nights?|week ?days?|school nights?|work nights?)\b",
    "weekend": r"\b(?:weekends?|sunday dinners?|saturday)\b",
    "children/kids": r"\b(?:for|with|my|our) (?:the |my |our )?(?:children|kids|kiddos?|toddlers?|sons?|daughters?|grandkids?|grandchildren)\b",
    "partner/spouse": r"\b(?:my|our) (?:husband|wife|partner|boyfriend|girlfriend|hubby|dh|fiance)\b",
    "guests/company": r"\b(?:guests?|company|dinner party|potluck|crowd)\b",
    "special occasion": r"\b(?:special occasions?|holidays?|anniversary|birthday|celebrations?|christmas|thanksgiving|easter)\b",
    "time-constrained": r"\b(?:quick|in a hurry|short on time|busy (?:night|day|weeknight)s?|time[- ]consuming)\b",
}

context_counts = {
    name: int(df_reviews[TEXT_COL].str.contains(pattern, case=False, regex=True).sum())
    for name, pattern in context_patterns.items()
}

count_df = pd.DataFrame.from_dict(context_counts, orient="index", columns=["Review Count"])
count_df["Percentage (%)"] = count_df["Review Count"] / len(df_reviews) * 100
count_df = count_df.sort_values("Review Count", ascending=False)
print("\n--- Frequency of Contextual Expressions ---")
print(count_df)

plt.figure(figsize=(10, 5))
sns.barplot(x=count_df.index, y=count_df["Review Count"], hue=count_df.index, palette="magma", legend=False)
plt.title("Frequency of Contextual Expressions in Reviews")
plt.ylabel("Number of Reviews")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.show()


# 6. SENTENCE / CLAUSE SPLITTING

CONTRAST_PATTERN = re.compile(r"(?=\b(?:but|although|however|yet|even though)\b)|(?<!even )(?=\bthough\b)", re.IGNORECASE)

def split_into_sentences(text):
    """Split into sentences, then into clauses before contrast words. Negations are never removed."""
    if pd.isna(text):
        return []
    clauses = []
    for sentence in sent_tokenize(str(text)):
        for clause in CONTRAST_PATTERN.split(sentence):
            clause = clause.strip(",; ")
            if clause:
                clauses.append(clause)
    return clauses

df_reviews["clauses"] = df_reviews[TEXT_COL].apply(split_into_sentences)

keep_cols = [c for c in (ID_COL, "recipe_id", "member_id", DATE_COL, RATING_COL, "clauses") if c and c in df_reviews.columns]
df_clauses = df_reviews[keep_cols].explode("clauses").rename(columns={"clauses": "sentences"})
df_clauses = df_clauses.dropna(subset=["sentences"])
df_clauses["clause_idx"] = df_clauses.groupby(level=0).cumcount()
df_clauses = df_clauses.reset_index(names="review_row")

print(f"\nClauses: {len(df_clauses)} from {len(df_reviews)} reviews")
print(df_clauses.head(10))