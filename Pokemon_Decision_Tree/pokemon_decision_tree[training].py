# train.py — Builds the tree from examples and saves it
# Run with: python train.py

import pickle
import os

# ── DATA ──────────────────────────────────────────────────────
# Each row: [my_hp, type_advantage, opp_hp, answer]
# type_advantage: 1 = yes, 0 = no

data = [
    [80, 1, 60, "Attack"],
    [70, 1, 50, "Attack"],
    [90, 1, 40, "Attack"],
    [60, 0, 20, "Attack"],   # opp almost dead → attack anyway
    [70, 0, 15, "Attack"],
    [55, 0, 80, "Switch"],   # no advantage + opp healthy → switch
    [80, 0, 90, "Switch"],
    [20, 0, 70, "Switch"],   # low HP → switch
    [15, 1, 60, "Switch"],
    [25, 0, 65, "Switch"],
]

X = [row[:3] for row in data]   # inputs
y = [row[3]  for row in data]   # answers


# ── GINI — measures how mixed a group is (0 = pure, 0.5 = messy) ──
def gini(labels):
    total = len(labels)
    score = 1.0
    for label in set(labels):
        score -= (labels.count(label) / total) ** 2
    return score


# ── FIND BEST SPLIT — tries every question, keeps the best one ──
def best_split(X, y):
    best = (None, None, 999.0)

    for feature in range(len(X[0])):
        for threshold in set(row[feature] for row in X):

            left  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
            right = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

            if not left or not right:
                continue

            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)

            if score < best[2]:
                best = (feature, threshold, score)

    return best[0], best[1]


# ── BUILD TREE — splits data recursively until groups are pure ──
def build(X, y, depth=0):
    # Stop if all answers are the same
    if len(set(y)) == 1:
        return {"answer": y[0]}

    # Stop if too deep
    if depth >= 3:
        return {"answer": max(set(y), key=y.count)}

    feature, threshold = best_split(X, y)

    if feature is None:
        return {"answer": max(set(y), key=y.count)}

    left_X  = [X[i] for i in range(len(X)) if X[i][feature] <= threshold]
    left_y  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
    right_X = [X[i] for i in range(len(X)) if X[i][feature] >  threshold]
    right_y = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

    return {
        "feature"   : feature,
        "threshold" : threshold,
        "left"      : build(left_X, left_y,  depth + 1),
        "right"     : build(right_X, right_y, depth + 1),
        "answer"    : None,
    }


# ── TRAIN ─────────────────────────────────────────────────────
tree = build(X, y)

# ── SAVE ──────────────────────────────────────────────────────
os.makedirs("Trained_Models", exist_ok=True)

name = input("Name your model: ").strip() or "model"
path = "Trained_Models/" + name + ".pkl"

with open(path, "wb") as f:
    pickle.dump(tree, f)

print("Saved to", path)