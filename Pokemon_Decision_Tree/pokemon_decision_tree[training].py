# train.py — Builds a Decision Tree from examples and saves it
# Run with: python train.py
#
# A Decision Tree works by asking a series of yes/no questions about your data.
# At each step it finds the single question (feature + threshold) that best
# separates your examples into clean groups, then recurses on each group.
# The result is a tree of nested if/else rules, learned entirely from data.

import pickle   # for saving the trained tree to disk
import os       # for creating the output folder


# ── DATA ──────────────────────────────────────────────────────────────────────
# Our training set — 10 hand-labelled Pokémon battle scenarios.
# Each row is: [my_hp, type_advantage, opp_hp, correct_decision]
#
#   my_hp          — your Pokémon's current HP (0–100)
#   type_advantage — 1 if you have a type advantage, 0 if you don't
#   opp_hp         — the opponent Pokémon's current HP (0–100)
#   decision       — what a smart trainer would do in this situation

data = [
    [80, 1, 60, "Attack"],   # healthy + type advantage → attack
    [70, 1, 50, "Attack"],
    [90, 1, 40, "Attack"],
    [60, 0, 20, "Attack"],   # no advantage but opp is almost dead → finish it
    [70, 0, 15, "Attack"],
    [55, 0, 80, "Switch"],   # no advantage + healthy opponent → cut your losses
    [80, 0, 90, "Switch"],
    [20, 0, 70, "Switch"],   # your HP is dangerously low → switch out
    [15, 1, 60, "Switch"],   # even with type advantage, 15 HP is too risky
    [25, 0, 65, "Switch"],
]

# Split the data into inputs (X) and labels (y).
# X is a list of [my_hp, type_advantage, opp_hp] lists.
# y is a list of strings: "Attack" or "Switch".
X = [row[:3] for row in data]
y = [row[3]  for row in data]


# ── GINI IMPURITY ─────────────────────────────────────────────────────────────
# Gini impurity measures how "mixed" a group of labels is.
#
#   0.0  → perfectly pure  (all labels are the same, e.g. all "Attack")
#   0.5  → maximally mixed (50% Attack, 50% Switch)
#
# Formula: 1 - Σ (proportion of class c)²
#
# Example:
#   labels = ["Attack", "Attack", "Switch"]
#   p(Attack) = 2/3,  p(Switch) = 1/3
#   gini = 1 - (2/3)² - (1/3)² = 1 - 0.444 - 0.111 = 0.444
#
# We want splits that produce groups with gini close to 0.

def gini(labels):
    total = len(labels)
    score = 1.0
    for label in set(labels):                        # iterate over unique classes
        proportion = labels.count(label) / total     # fraction of this class
        score -= proportion ** 2                     # subtract squared proportion
    return score                                     # lower = purer


# ── BEST SPLIT ────────────────────────────────────────────────────────────────
# Finds the single (feature, threshold) pair that produces the lowest
# weighted Gini impurity across the two resulting groups.
#
# For each feature (column) and each unique value in that column as a candidate
# threshold, we split the data into:
#   left  — rows where feature value ≤ threshold
#   right — rows where feature value >  threshold
#
# We score the split using weighted Gini:
#   score = (|left| * gini(left) + |right| * gini(right)) / |total|
#
# The split with the lowest score wins — it creates the purest groups.

def best_split(X, y):
    best = (None, None, 999.0)   # (best_feature, best_threshold, best_score)

    for feature in range(len(X[0])):                    # try each feature column
        for threshold in set(row[feature] for row in X): # try each unique value as a threshold

            # Split labels based on this feature/threshold question
            left  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
            right = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

            # Skip if one side is empty — this isn't a real split
            if not left or not right:
                continue

            # Weighted Gini score for this split
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)

            # Keep it if it's the best we've seen
            if score < best[2]:
                best = (feature, threshold, score)

    return best[0], best[1]   # return the winning feature index and threshold


# ── BUILD TREE ────────────────────────────────────────────────────────────────
# Recursively builds the decision tree by:
#   1. Checking if we should stop (pure group, or too deep)
#   2. Finding the best split on the current data
#   3. Splitting the data and recursing on each half
#
# Each node in the tree is a plain Python dict:
#
#   Leaf node  → {"answer": "Attack"}
#                 returned when a group is pure or depth limit is hit
#
#   Split node → {"feature": 1, "threshold": 0.5,
#                 "left": <subtree>, "right": <subtree>, "answer": None}
#                 represents a question: "is feature[1] <= 0.5?"

def build(X, y, depth=0):

    # ── BASE CASE 1: Pure group ───────────────────────────────────────────────
    # If every label in this group is the same, there's nothing left to split.
    # Return a leaf node with that single answer.
    if len(set(y)) == 1:
        return {"answer": y[0]}

    # ── BASE CASE 2: Depth limit reached ─────────────────────────────────────
    # Letting the tree grow indefinitely would make it memorise the training data
    # (overfitting). Capping depth forces it to generalise.
    # At the limit we return the majority class as a best-guess leaf.
    if depth >= 3:
        return {"answer": max(set(y), key=y.count)}

    # ── FIND BEST SPLIT ───────────────────────────────────────────────────────
    feature, threshold = best_split(X, y)

    # If no valid split exists (e.g. all rows are identical), fall back to majority
    if feature is None:
        return {"answer": max(set(y), key=y.count)}

    # ── PARTITION DATA ────────────────────────────────────────────────────────
    # Divide rows into left (≤ threshold) and right (> threshold) groups
    left_X  = [X[i] for i in range(len(X)) if X[i][feature] <= threshold]
    left_y  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
    right_X = [X[i] for i in range(len(X)) if X[i][feature] >  threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

    # ── RECURSE ───────────────────────────────────────────────────────────────
    # Build subtrees for each partition, incrementing depth each time.
    # This continues until every group is pure or the depth cap is hit.
    return {
        "feature"  : feature,             # which feature this node tests (0, 1, or 2)
        "threshold": threshold,            # the value it compares against
        "left"     : build(left_X, left_y,   depth + 1),   # subtree for ≤ threshold
        "right"    : build(right_X, right_y, depth + 1),   # subtree for >  threshold
        "answer"   : None,                 # None means this is a split node, not a leaf
    }


# ── TRAIN ─────────────────────────────────────────────────────────────────────
# Kick off the recursive build on the full dataset.
# 'tree' will be the root node dict — the whole model lives in this structure.
tree = build(X, y)


# ── SAVE ──────────────────────────────────────────────────────────────────────
# pickle serialises the Python dict (and all its nested dicts) to a binary file.
# Loading it back gives you the exact same structure, ready to predict with.

os.makedirs("Trained_Models", exist_ok=True)   # create folder if it doesn't exist

name = input("Name your model: ").strip() or "model"
path = "Trained_Models/" + name + ".pkl"

with open(path, "wb") as f:
    pickle.dump(tree, f)          # wb = write binary

print("Saved to", path)
