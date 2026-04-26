# predict.py — Loads a trained Decision Tree and gives battle advice
# Run with: python predict.py
#
# The model is a plain Python dict (nested dicts, actually).
# Prediction works by walking down the tree from the root:
#   - At each split node, ask the node's question about the input
#   - Go left if the answer is ≤ threshold, right if it's > threshold
#   - Stop when you hit a leaf node — its "answer" is the prediction

import pickle   # for loading the saved model
import os       # for listing files in the models folder


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
# Scan the Trained_Models folder and let the user pick one.
# Each .pkl file is a serialised decision tree saved by train.py.

models = [f for f in os.listdir("Trained_Models") if f.endswith(".pkl")]

for i, name in enumerate(models):
    print(f"[{i+1}] {name}")

choice = int(input("Pick a model: ")) - 1         # user enters 1-based, convert to 0-based
path   = "Trained_Models/" + models[choice]

with open(path, "rb") as f:                       # rb = read binary
    tree = pickle.load(f)                         # deserialise back into a Python dict


# ── PREDICT ───────────────────────────────────────────────────────────────────
# Recursively walks the tree until it reaches a leaf, then returns the answer.
#
# node — the current tree node (a dict)
# row  — the input to classify: [my_hp, type_advantage, opp_hp]
#
# How it works at each node:
#   If node["answer"] is not None  → this is a leaf, return the answer
#   Otherwise                      → this is a split node:
#       check row[node["feature"]] against node["threshold"]
#       go left  if ≤ threshold
#       go right if >  threshold
#       recurse on the chosen child

def predict(node, row):

    # ── Leaf node ─────────────────────────────────────────────────────────────
    # "answer" is set on leaf nodes (None on split nodes).
    # When we reach a leaf, we've followed the decision path to its conclusion.
    if node["answer"] is not None:
        return node["answer"]

    # ── Split node ────────────────────────────────────────────────────────────
    # Ask the question this node represents:
    #   "Is feature[node['feature']] ≤ node['threshold']?"
    #
    # feature indices map to:
    #   0 → my_hp
    #   1 → type_advantage
    #   2 → opp_hp
    if row[node["feature"]] <= node["threshold"]:
        return predict(node["left"],  row)    # answer is ≤ threshold → go left
    else:
        return predict(node["right"], row)    # answer is >  threshold → go right


# ── INTERACTIVE PREDICTION LOOP ───────────────────────────────────────────────
# Keep asking for battle stats and printing decisions until the user quits.

while True:
    # Collect the three features the tree was trained on
    my_hp    = float(input("\nYour HP (0-100):              "))
    type_adv =   int(input("Type advantage? (1=yes, 0=no): "))
    opp_hp   = float(input("Opponent HP (0-100):           "))

    # Build the feature vector in the same order as the training data:
    # [my_hp, type_advantage, opp_hp]
    result = predict(tree, [my_hp, type_adv, opp_hp])

    print("\n→ Decision:", result)

    # Ask if the user wants to run another prediction
    if input("\nTry again? (y/n): ").lower() != "y":
        break
