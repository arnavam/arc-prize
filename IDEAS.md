## Present Idea

- reduce the training loss
- make more robust scoring by adding  error of  miss placing + error not placing
- use more refined early-stopping
- use diff RL algorithms or variants .
- create a resoning type advanced pre-training datasts ( like if yello move top etc..)
- add 'remove' dsl , pre-train using that  & also add some other dsls

- positional encoding
- use masked fitting where object shape can be incredebly complex in shape
- make the pretraining more robust by evaluating on unseen data.
- make evaluation on unseen target i.e, the target shouldnt shown or replace with current during placement
- if no of combination and placment runs out we need to create another task. 
- None values during placement in supervised_learning
## cause of concern 
- overlapping  of image for some reason ( might be from intermediate but could also be from beginner)
- task of placing being created in intermediate side
- nan values being created using RL training

## optmizations to perform
- 


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


