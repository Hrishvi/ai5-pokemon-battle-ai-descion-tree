 pokemon-battle-tree

> A Decision Tree classifier built from scratch in Python that decides whether to **Attack** or **Switch** in a Pokémon battle — written in pure Python, zero dependencies.

---

## What is this?

Wanted to learn how decision trees actually work before touching scikit-learn. So I built one by hand, trained it on Pokémon battle scenarios, and wired up a little CLI to use it.

The problem is intentionally tiny: given your HP, your opponent's HP, and whether you have a type advantage — should you attack or switch? Three features, two outcomes, ten training examples. Small enough to trace through by hand, which is kind of the point.

---

## Features

- ✅ Decision tree built entirely from scratch — pure Python, no libraries
- ✅ Gini impurity to evaluate every possible split
- ✅ Recursive tree construction with configurable max depth
- ✅ Save & load trained models with `pickle`
- ✅ Interactive CLI for training and prediction

---

## How it works

Training data:

| My HP | Type Advantage | Opp HP | Decision |
|-------|---------------|--------|----------|
| 80    | Yes           | 60     | Attack   |
| 20    | No            | 70     | Switch   |
| 60    | No            | 20     | Attack   |
| 15    | Yes           | 60     | Switch   |
| ...   | ...           | ...    | ...      |

At each node, the algorithm tries every possible split across every feature and threshold, scores each one using **Gini impurity**, and picks the split that produces the purest groups. It recurses until all leaf nodes are pure (one class only) or the depth limit is hit.

---

## Getting started

```bash
git clone https://github.com/yourusername/pokemon-battle-tree.git
cd pokemon-battle-tree
python pokemon_decision_tree_training_.py
```

Name your model when prompted — it saves to `Trained_Models/`. Then:

```bash
python pokemon_decision_tree_testing_.py
```

Pick a model from the list, enter the battle stats, and get a decision.

---

## Example output

```
Your HP (0-100):              75
Type advantage? (1=yes, 0=no): 1
Opponent HP (0-100):          40

→ Decision: Attack
```

---

## Project structure

```
pokemon-battle-tree/
├── pokemon_decision_tree_training_.py   # Build and save a tree
├── pokemon_decision_tree_testing_.py    # Load a model and predict
├── Trained_Models/
│   └── Pokemon_Decision_Tree.pkl        # Pre-trained example model
└── README.md
```

---

## Why bother building it from scratch?

Because `DecisionTreeClassifier().fit(X, y)` tells you nothing about what's actually happening. Building it yourself forces you to answer questions like:

- What does "best split" actually mean, and best by what measure?
- Why do we use Gini impurity instead of just counting errors?
- What stops the tree from memorising the training data entirely?
- What even *is* a leaf node in memory?

Once you've written the `build()` function yourself, those questions have concrete answers.

---

## Concepts covered

- Decision trees (recursive binary splitting)
- Gini impurity as a purity measure
- Greedy split selection (exhaustive threshold search)
- Depth-limited tree growth to prevent overfitting
- Model serialisation with `pickle`

---

## Related project

If you liked this, the follow-up repo [`ai6-pokemon-battle-ai-random-forest`](https://github.com/Hrishvi/ai6-pokemon-battle-ai-random-forest) extends this into a full **Random Forest** — bootstrap sampling, feature subsampling, majority vote across multiple trees, confidence scores.

---

## License

MIT
