import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Exploratory Data Analysis and Text Preparation


# ==========================================
# 1. DATA PREPARATION & MISSING VALUES
# ==========================================
# Identify standard columns
text_col = 'review' if 'review' in df_reviews.columns else 'review_text' if 'review_text' in df_reviews.columns else 'text'
rating_col = 'rating' if 'rating' in df_reviews.columns else 'stars' if 'stars' in df_reviews.columns else None

print("\n--- Missing Values in Reviews ---")
print(df_reviews.isnull().sum())

print("\n--- Missing Values in Members ---")
print(df_members.isnull().sum())

print("\n--- Missing Values in recipes ---")
print(df_recipes.isnull().sum())

# Drop rows without text
df_reviews = df_reviews.dropna(subset=[text_col])
df_reviews[text_col] = df_reviews[text_col].astype(str)

# ==========================================
# 2. REVIEW LENGTH ANALYSIS
# ==========================================
df_reviews['review_length'] = df_reviews[text_col].apply(lambda x: len(x.split()))

print("\n--- Review Length Statistics ---")
print(df_reviews['review_length'].describe())

plt.figure(figsize=(10, 5))
sns.histplot(df_reviews['review_length'], bins=50, kde=True, color='teal')
plt.title('Distribution of Review Lengths')
plt.xlabel('Number of Words')
plt.ylabel('Frequency')
plt.xlim(0, df_reviews['review_length'].quantile(0.95)) # Cap at 95th percentile
plt.show()

# ==========================================
# 3. RATING DISTRIBUTION
# ==========================================
if rating_col:
    print("\n--- Rating Distribution (%) ---")
    print(df_reviews[rating_col].value_counts(normalize=True).sort_index() * 100)

    plt.figure(figsize=(8, 5))
    sns.countplot(
        x=rating_col, 
        data=df_reviews, 
        hue=rating_col, 
        palette='viridis', 
        legend=False
    )
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.show()

# ==========================================
# 4. CONTEXTUAL EXPRESSIONS FREQUENCY
# ==========================================
context_patterns = {
    "weeknight/weekday": r"\b(weeknight|weekday)s?\b",
    "for my children/kids": r"\b(for my children|for my kids|for the kids|for my son|for my daughter)\b",
    "special occasion": r"\b(special occasion|holiday|anniversary|celebration)\b"
}

context_counts = {}
for context_name, pattern in context_patterns.items():
    mentions = df_reviews[text_col].apply(lambda x: 1 if re.search(pattern, x, re.IGNORECASE) else 0).sum()
    context_counts[context_name] = mentions

print("\n--- Frequency of Contextual Expressions ---")
count_df = pd.DataFrame.from_dict(context_counts, orient='index', columns=['Review Count'])
count_df['Percentage (%)'] = (count_df['Review Count'] / len(df_reviews)) * 100
print(count_df.sort_values(by='Review Count', ascending=False))

# Plot contextual expressions
plt.figure(figsize=(8, 5))
sns.barplot(
    x=count_df.index, 
    y=count_df['Review Count'], 
    hue=count_df.index, 
    palette='magma', 
    legend=False
)
plt.title('Frequency of Contextual Expressions in Reviews')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()


# ==========================================
# 5. Split Reviews into Sentences
# ==========================================

import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK sentence tokenizer (run once)
# nltk.download('punkt')

def split_into_sentences(text):
    """
    Splits text into sentences, and further splits into clauses based on 
    contrast words, preserving the contrast words and negations.
    """
    if pd.isna(text):
        return []
        
    # 1. Standard sentence splitting (preserves all negations like 'not', 'never')
    sentence_tokens = sent_tokenize(str(text))
    
    sentences = []
    
    # 2. Split clauses at contrast words (but, although, however, yet)
    # The regex lookahead (?=\b...\b) splits BEFORE the word, 
    # ensuring the word remains in the resulting clause.
    contrast_pattern = r'(?=\b(?:but|although|however|yet|though)\b)'
    
    for sentence in sentence_tokens:
        # Split the sentence based on the pattern
        sub_sentences = re.split(contrast_pattern, sentence, flags=re.IGNORECASE)
        
        # Clean whitespace and punctuation at the edges
        for sub_sentence in sub_sentences:
            cleaned_sentence = sub_sentence.strip(',; ')
            if cleaned_sentence:
                sentences.append(cleaned_sentence)
                
    return sentences

# ==========================================
# TEST THE FUNCTION
# Identify the correct text column dynamically
text_col = 'review' if 'review' in df_reviews.columns else 'review_text' if 'review_text' in df_reviews.columns else 'text'

# Create a new DataFrame with one clause per row
exploded_reviews = df_reviews.copy()

# Apply the function using the dynamically found column name (text_col)
exploded_reviews['sentences'] = exploded_reviews[text_col].apply(split_into_sentences)

# 'Explode' the list so each clause gets its own row
df_clauses = exploded_reviews.explode('sentences').reset_index(drop=True)

# Drop empty clauses
df_clauses = df_clauses[df_clauses['sentences'].str.len() > 0]

print(df_clauses[['recipe_id', 'sentences']].head())