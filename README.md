# ARC Prize 2025 – Neuro-Symbolic Solver

This project is focused on solving tasks from the [ARC (Abstraction and Reasoning Corpus)](https://github.com/fchollet/ARC) using a combination of **symbolic algorithms** (like A*, BFS, DSLs) and **neural models**, integrating them in a neuro-symbolic framework. The goal is to develop systems that generalize from few examples and mimic human-like reasoning.


## 🧠 Project Structure

| File | Description |
|------|-------------|
| `A_arc.py` | first file created , my first rough implementation of the solution  |
| `a-star.py`, `a-star2.py` | A* search-based solvers  , dosent work very well  |
| `bfs.py`, `bfs2.py`, `bfs3.py` | BFS-based solvers, works well though dosent provide correct soln and takes long time  |
| `dsl.py`, `dsl2.py` | my dsl implementations |
| `neuro+mcts.py`, `neuro+mcts2.py`| Neural model combined with MCTS for task planning |
| `neurosymbolic_reinforce.py` | Torch-based neural-symbolic integration |
| `nuerosymbolic_tf.py` | tf based neuro-symbolic pipeline |
| `neurosymbolics_tf.ipynb` | same thing using to run in colab  |
| `neurosymbolic_RLA2C.py` | Torch-based neural-symbolic ,updatation using RL_A2C |
| `nuerosymb_q_learning.py` | Torch-based neural-symbolic ,updatation using q-learning (greedy-epsilon) |
| `submission.py`, `submission.json` | Code and metadata for ARC submission format |
| `Trash.py` | Scratch or deprecated experiments |
| `IDEAS.md` | Notes and brainstorming for models and approaches |
| `.gitignore` | Standard Git ignore rules |
| `README.md` | This file |







**To be added .**
---

