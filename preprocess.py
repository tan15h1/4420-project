"""
DS 4420 Final Project - Data Preprocessing
Tanishi Datta & Shruthi Palaniappan

Parses RAW_recipes.csv (non-standard format), cleans ingredient and tag data,
and saves a clean CSV that both models can load directly.

Output: recipes_clean.csv
  - ingredients: list of cleaned ingredients
  - tags: list of cleaned tags
"""

import re
import ast
import pandas as pd
from collections import Counter


with open("RAW_recipes.csv", "r", errors="replace") as f:
    content = f.read()

# Ingredients: each entry looks like: ""['ing1', 'ing2', ...]"",N"
# Tags: the submitted date (e.g. 9/16/05) always appears right before the tags list
ing_pattern = re.compile(r'""(\[[^\]]*\])"",(\d+)"')
tag_pattern = re.compile(r',\d{1,2}/\d{1,2}/\d{2},""(\[[^\]]*\])""')

ing_matches = list(ing_pattern.finditer(content))
tag_matches = list(tag_pattern.finditer(content))
print(f"  Found {len(ing_matches)} ingredient lists, {len(tag_matches)} tag lists")

# Pair each ingredient list with the nearest preceding tags list
# Both lists are already sorted by position in the file, so we walk
# through them together in one pass (O(n)) instead of searching each time
recipes = []
tag_idx = 0

for ing_match in ing_matches:
    ing_pos = ing_match.start()

    # Advance tag_idx as long as the next tag still comes before this ingredient
    while tag_idx + 1 < len(tag_matches) and tag_matches[tag_idx + 1].start() < ing_pos:
        tag_idx += 1

    # Make sure the current tag actually precedes this ingredient
    if tag_matches[tag_idx].start() >= ing_pos:
        continue

    try:
        ingredients = ast.literal_eval(ing_match.group(1))
        tags = ast.literal_eval(tag_matches[tag_idx].group(1))
        if len(ingredients) == int(ing_match.group(2)):
            recipes.append({"ingredients": ingredients, "tags": tags})
    except (ValueError, SyntaxError):
        pass

print(f"  Extracted {len(recipes)} recipes")
n = len(recipes)

# ──────────────────────────────────────────────
# 2. Data Cleaning 
# ──────────────────────────────────────────────

# Clean each ingredient - remove tabs, strip whitespace, lowercase
def clean_ingredients(ingredients):
    cleaned = []
    for i in ingredients:
        i = i.replace("\t", "")
        i = i.strip().lower()
        if len(i) > 1 and any(c.isalpha() for c in i):
            cleaned.append(i)
    return cleaned

# Only keep tags that are useful for ingredient substitution
ALLOWED_TAGS = {
    # Cuisine
    "north-american", "american", "european", "asian", "italian",
    "french", "mexican", "greek", "indian", "chinese", "japanese",
    "thai", "mediterranean", "spanish", "canadian", "african",
    # Diet / health
    "vegetarian", "vegan", "low-carb", "low-fat", "low-sodium",
    "low-calorie", "low-cholesterol", "low-protein", "healthy",
    "high-protein", "high-in-something", "very-low-carbs", "gluten-free",
    # Dish type
    "main-dish", "desserts", "side-dishes", "appetizers", "breakfast",
    "brunch", "lunch", "snacks", "salads", "soups-stews", "breads",
    "beverages", "one-dish-meal", "condiments-etc", "cookies-and-brownies",
    "cakes", "sweets",
    # Ingredient-based
    "meat", "vegetables", "fruit", "pasta", "chicken", "beef", "pork",
    "seafood", "fish", "eggs-dairy", "cheese", "beans", "potatoes",
    "chocolate", "poultry", "eggs",
    # Style / occasion
    "comfort-food", "holiday-event", "dinner-party", "kid-friendly",
    "spicy", "sweet", "savory", "seasonal", "summer", "fall",
    "winter", "christmas", "weeknight",
}

def clean_tags(tags):
    cleaned = []
    for tag in tags:
        tag = tag.strip().lower()
        if tag in ALLOWED_TAGS:
            cleaned.append(tag)
    return cleaned


for recipe in recipes:
    recipe["ingredients"] = clean_ingredients(recipe["ingredients"])
    recipe["tags"] = clean_tags(recipe["tags"])

# Drop recipes with fewer than 3 ingredients
recipes = [r for r in recipes if len(r["ingredients"]) >= 3]
print(f"  Remove < 3 ingredients: {n - len(recipes)} removed, {len(recipes)} remaining")
n = len(recipes)

# Remove duplicate recipes 
seen = set()
unique = []
for recipe in recipes:
    key = tuple(sorted(recipe["ingredients"]))
    if key not in seen:
        seen.add(key)
        unique.append(recipe)
recipes = unique
print(f"  Remove duplicates: {n - len(recipes)} removed, {len(recipes)} remaining")

print(f"\nAfter cleaning: {len(recipes)} recipes")

# ──────────────────────────────────────────────
# 3. SAVE AS DATAFRAME
# ──────────────────────────────────────────────
# Pickle preserves the DataFrame exactly - lists stay as lists, no parsing needed on load
# Load it back with: df = pd.read_pickle("recipes_clean.pkl")

df = pd.DataFrame(recipes)
df.to_pickle("recipes_clean.pkl")

