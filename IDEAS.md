## Present Idea

- pre-training the RL model using supervised alg with simple dataset to give it a basic understanding of its role
    -   could also be usefull assess performance of model in terms of its hyperparameters
- add remove dsl & also some other dsls
- use masked obj extraction to also indicate location of objects and no need to worry about shape problem
- positional encoding
- use masked fitting where object shape can be incredebly complex in shape

---

## other 
- A model that creates its own DSLs
- 2 CNNs o/p are mutlipled together to similar to a transformer to get relevant pos of i/p in o/p .

- partial/full comparison
    - Start with one-by-one + partial comparisons (faster for most tasks).
    - Fall back to full-grid checks if partial matches succeed but the full solution fails.
    - Use object masks to avoid redundant computations (critical for speed).

## collected Ideas

**DSLs**

- color change recognition from input-1to input-2  & input to output

**DSLs from Kaggle Solutions:**    
- [ARCathon](https://github.com/arcathon/)
- [michaelhodel](https://github.com/michaelhodel/arc-dsl/tree/main)

    
 **Visual Reasoning Engines:** 
 - [rival](https://github.com/msamsami/rival) abstract grid operations.
- [pyreason](https://github.com/lab-v2/pyreason)
- [LTNtorch](https://github.com/tommasocarraro/LTNtorch)


**General-Purpose Tools:**
- **Scheme/LISP:** For symbolic reasoning (e.g., miniKanren).
- **Probabilistic DSLs:** (e.g., Gen.jl) for uncertain reasoning

**Prev Solutions**

https://github.com/SimonOuellette35/GridCoder2024 code for grid coder


