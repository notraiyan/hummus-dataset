# Installed before transformers is first imported (Task 3), so that Task 10 can load its relevance judge in 4-bit.
!pip install -q bitsandbytes

import kagglehub
from kagglehub import KaggleDatasetAdapter

# 1. Load Members Data
df_members = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "raiyanahmed/hummus",
  "pp_members.csv",
  pandas_kwargs={"low_memory": False}
)
print("Members first 5 records:\n", df_members.head(), "\n")

# 2. Load Recipes Data
# (This is usually the file with the most columns and the source of the warning)
df_recipes = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "raiyanahmed/hummus",
  "pp_recipes.csv",
  pandas_kwargs={"low_memory": False}
)
print("Recipes first 5 records:\n", df_recipes.head(), "\n")

# 3. Load Reviews Data
df_reviews = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "raiyanahmed/hummus",
  "pp_reviews.csv",
  pandas_kwargs={"low_memory": False}
)
print("Reviews first 5 records:\n", df_reviews.head())