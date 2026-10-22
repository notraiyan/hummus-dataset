import html
import re
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from nltk.tokenize import sent_tokenize

# Task 1: Exploratory Data Analysis and Text Preparation

# Run the dataset loader module first and use it here

for resource in ("punkt", "punkt_tab"):
    nltk.download(resource, quiet=True)

# HUMMUS pp_reviews.csv stores the review body in `text`.
TEXT_COL = "text"
RATING_COL = "rating"
# pp_reviews.csv has no submit_date; last_modified_date is the only review timestamp (an edit can move it later).
DATE_COL = "last_modified_date"
ID_COL = "review_id" if "review_id" in df_reviews.columns else None

# ==========================================
# 1. MISSING VALUES
# ==========================================
for name, df in [("Reviews", df_reviews), ("Members", df_members), ("Recipes", df_recipes)]:
    print(f"\n--- Missing Values in {name} ---")
    print(df.isnull().sum())

# Review text holds HTML: escaped characters ("&#039;", "&amp;"), line breaks as <br/> tags (some escaped as
# "&lt;br/&gt;") and a few formatting tags. Unescape it and keep line breaks as "\n", which the clause splitter uses as
# sentence boundaries; other whitespace is collapsed.
BREAK_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
FORMAT_TAG = re.compile(r"</?(?:a|b|i|u|em|strong|p|span|div|font)\b[^>]*>", re.IGNORECASE)


def clean_text(text):
    text = FORMAT_TAG.sub("", BREAK_TAG.sub("\n", html.unescape(str(text))))
    return re.sub(r"\s*\n\s*", "\n", re.sub(r"[^\S\n]+", " ", text)).strip()


# HUMMUS adds authorship rows (rating 6) that have no review text; they cannot be analysed as text.
n_total = len(df_reviews)
df_reviews = df_reviews.dropna(subset=[TEXT_COL]).copy()
df_reviews[TEXT_COL] = df_reviews[TEXT_COL].map(clean_text)
df_reviews = df_reviews[df_reviews[TEXT_COL] != ""].copy()
df_reviews[DATE_COL] = pd.to_datetime(df_reviews[DATE_COL], errors="coerce")
print(f"\nReviews with text: {len(df_reviews)} / {n_total} ({len(df_reviews) / n_total:.1%})")

# ==========================================
# 2. REVIEW LENGTH ANALYSIS
# ==========================================
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

# ==========================================
# 3. RATING DISTRIBUTION
# ==========================================
# HUMMUS preprocessing: rating 0 (no rating) -> 2.5, authorship -> 6. Only 1-5 are real star ratings.
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

# ==========================================
# 4. CONTEXTUAL EXPRESSIONS FREQUENCY
# ==========================================
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

# ==========================================
# 5. SENTENCE / CLAUSE SPLITTING
# ==========================================
# Zero-width lookahead keeps the contrast word at the start of its clause; "even" stays attached to "though".
CONTRAST_PATTERN = re.compile(r"(?=\b(?:but|although|however|yet|even though)\b)|(?<!even )(?=\bthough\b)", re.IGNORECASE)


def split_into_sentences(text):
    """Split into lines and sentences, then into clauses before contrast words. Negations are never removed."""
    if pd.isna(text):
        return []
    clauses = []
    for line in str(text).splitlines():
        for sentence in sent_tokenize(line):
            for clause in CONTRAST_PATTERN.split(sentence):
                clause = clause.strip(",; ")
                if clause:
                    clauses.append(clause)
    return clauses


df_reviews["clauses"] = df_reviews[TEXT_COL].apply(split_into_sentences)

# One row per clause; keep ids and clause position so later tuples can cite evidence spans.
keep_cols = [c for c in (ID_COL, "recipe_id", "member_id", DATE_COL, RATING_COL, "clauses") if c and c in df_reviews.columns]
df_clauses = df_reviews[keep_cols].explode("clauses").rename(columns={"clauses": "sentences"})
df_clauses = df_clauses.dropna(subset=["sentences"])
df_clauses["clause_idx"] = df_clauses.groupby(level=0).cumcount()
df_clauses = df_clauses.reset_index(names="review_row")

print(f"\nClauses: {len(df_clauses)} from {len(df_reviews)} reviews")
print(df_clauses.head(10))
