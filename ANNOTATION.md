# Annotation guidelines

These guidelines define the labels for Project 29: which food and context words a review clause mentions, which opinions it expresses, whose opinions those are, and under which conditions they hold.

## Who labelled what

- **The 1,200 reviews** (the 200-review pilot and 1,000 further reviews in `annotation_sheet.csv`) were pre-annotated by LLM, following these guidelines.
- **The author checks and corrects them** in `annotation_checked.csv`: fix or delete wrong labels, add missing ones, and write `y` in `checked` for every clause checked. Only checked clauses are used as gold; report them as machine pre-annotation checked by the author, with the number of clauses checked.
- **Topic ratings** are done by the author; a second rater is optional (`topic_ratings_B.csv`). Relevance in Task 10 is judged by an LLM and explanations are checked automatically in Task 9 (see the README).

Each row is one clause (column `clause`). The full review is in `review`, only to work out what "it", "they" or "he" refer to. Label only what the clause itself says, and leave a list empty (`[]`) when there is nothing to mark.

## Entities (`entities`)

A JSON list of `[span, LABEL]`, with the span copied exactly from the clause:

```
[["chicken", "INGREDIENT"], ["slow cooker", "COOKING_EQUIPMENT"], ["30 minutes", "DURATION"]]
```

| Label | Mark | Do not mark |
|---|---|---|
| `INGREDIENT` | foods used as ingredients or eaten as such: "sour cream", "cumin" | "ingredients" in general |
| `DISH` | the dish or a named dish: "lasagna", "chicken soup", "cookies" | "recipe", "it", "this", course words ("dish", "side dish", "meal", "dessert", "treat", "snack") |
| `COOKING_EQUIPMENT` | tools and appliances: "crock pot", "skillet", "oven" | |
| `DURATION` | lengths of time: "2 hours", "overnight" | dates, "next time" |
| `OCCASION` | occasions: "weeknight", "Christmas", "potluck", "party" | meal times ("breakfast", "dinner") |
| `DINER_GROUP` | who eats: "kids", "husband", "guests", "family" | the reviewer ("I", "we") |

Mark the longest name ("sour cream", not "cream"), without "the" or "my" and without preparation words ("chopped", "sliced", "diced", "minced", "grated", "melted", "softened", "cooked"). Keep words that name a kind of product ("canned beans", "frozen peas", "ground beef", "fresh basil").

## Opinions (`opinions`)

One entry per opinion about an aspect of the recipe:
`[span, ASPECT, sentiment, holder, context, condition, negated]`

- **span:** the shortest piece of the clause that contains the opinion, with the aspect word when the clause names it: "too hot", "bland", "easy to make", "flavor was great".
- **ASPECT:** `TASTE`, `TEXTURE`, `PREP_EFFORT`, `TIME`, `INGREDIENT_AVAILABILITY`, `HEALTHINESS` or `SUITABILITY`. Healthiness is what the reviewer perceives, not the nutrition data. Suitability is how well the recipe fits an occasion the clause names (see Decisions).
- **sentiment:** `positive`, `negative` or `neutral`, as the clause states it. "not too spicy" is positive; "didn't have X" is negative; neutral is an aspect mentioned without a judgement.
- **holder:** `reviewer` (also when nobody else is named), `children`, `partner`, `family` (the household as a group, "we", or other relatives), `guests` (friends, company, people at a party) or `unknown`.
- **context:** `weekday`, `weekend`, `holiday`, `gathering`, `special occasion`, `general` (none named) or `unknown` (two different ones).
- **condition:** `""`, `"condition"` (holds only if: "quick if you use canned beans") or `"exception"` (does not hold in some situation: "except at breakfast", "but not if you add cream").
- **negated:** `true` if a negation applies to the opinion word ("not too spicy", "I don't think it was bland"), otherwise `false`.

Example, for the clause "but this was too hot for my children.":

```
[["too hot", "TASTE", "negative", "children", "general", "", false]]
```

Generic praise that names no aspect ("great recipe"), instructions or changes without a judgement ("I added more garlic"), and things other than the recipe ("my oven runs hot") get no opinion.

## Decisions (the pre-annotation follows these; check against them too)

- Liking or disliking a food itself ("I love broccoli") names no aspect: no opinion. Liking or disliking a flavour quality ("I love spicy food", "too sweet for me") is TASTE.
- A change to a time or an aspect without a judgement is neutral: "leave it a bit longer" (TIME, neutral), "to make it healthier" (HEALTHINESS, neutral). A plain step time ("bake for 8 hours") gets no opinion.
- Sauces, frostings, glazes and gravies used in another dish are INGREDIENT; when the sauce itself is the recipe, it is DISH.
- "light" is HEALTHINESS only when it means fewer calories or less fat ("a light version", "made them light with low-fat cheese"); otherwise it is TEXTURE ("light and fluffy", "a very light dessert").
- Smell or aroma alone ("smells wonderful") is not TASTE: no opinion.
- A task that was hard to do ("too runny to work with") is PREP_EFFORT, negative, in addition to any TEXTURE opinion.
- When the clause names only someone vague ("the non-spicy eaters enjoyed them"), the holder is `unknown`.
- A suggestion of what would be better is negative on the aspect it names, with no condition ("would be tastier with ground turkey", "better if it was a little saucier"). With no aspect named ("pecans would have been better") it gets no opinion.
- A substitution because the recipe's ingredient was not at hand ("used walnuts since that's all I had", "used what I had on hand") is INGREDIENT_AVAILABILITY, negative. Having something to use up that led to making the recipe ("I had leftover rice, so I made this") is positive.
- Suitability for an occasion is SUITABILITY: a judgement of how well the recipe fits an occasion the clause names ("great for weekend dinners", "perfect for a potluck", "a hit at our Christmas party", "not a weeknight meal", "will be making this every Thanksgiving"), including praise the clause ties to the occasion ("made it for Thanksgiving and everyone loved it"). The span is the judgement with the occasion; the context is that occasion; the holder is whoever judges it, usually the reviewer (people it was served to are the audience, unless the clause gives their verdict: "the guests said it was perfect for a party" is `guests`). No SUITABILITY for praise without an occasion ("great recipe"), a meal time alone ("great for lunch"), an occasion without a judgement ("made it for Christmas"), or an occasion outside the context values ("camping"). When the judgement names another aspect ("too time-consuming for weekdays"), label that aspect, with the occasion as its context, not SUITABILITY.
- The context is an occasion only when the clause names it ("leftovers for lunch" is `general`). The condition is `condition` or `exception` only when the clause itself limits the opinion ("if", "when", "once", "unless", "without", "best made a day ahead"). A past account of what happened has none ("I had cooked rice, so it was quick").

## Topic ratings (Task 3)

Fill `label` and `interpretable_1to3` in Task 3's `topics_for_human_rating_*.csv`:

- `1`: no theme
- `2`: mixed
- `3`: clear theme

Save it as `topic_ratings_A.csv`.
