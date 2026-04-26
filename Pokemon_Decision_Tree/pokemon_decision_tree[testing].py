# predict.py — Loads a trained model and gives battle advice
# Run with: python predict.py

import pickle
import os

# ── LOAD MODEL ────────────────────────────────────────────────
models = [f for f in os.listdir("Trained_Models") if f.endswith(".pkl")]

for i, name in enumerate(models):
    print(f"[{i+1}] {name}")

choice = int(input("Pick a model: ")) - 1
path   = "Trained_Models/" + models[choice]

with open(path, "rb") as f:
    tree = pickle.load(f)


# ── PREDICT — walks down the tree and returns the answer ──────
def predict(node, row):
    if node["answer"] is not None:
        return node["answer"]
    if row[node["feature"]] <= node["threshold"]:
        return predict(node["left"],  row)
    else:
        return predict(node["right"], row)


# ── ASK AND PREDICT ───────────────────────────────────────────
while True:
    my_hp    = float(input("\nYour HP (0-100):              "))
    type_adv =   int(input("Type advantage? (1=yes, 0=no): "))
    opp_hp   = float(input("Opponent HP (0-100):           "))

    result = predict(tree, [my_hp, type_adv, opp_hp])
    print("\n→ Decision:", result)

    if input("\nTry again? (y/n): ").lower() != "y":
        break