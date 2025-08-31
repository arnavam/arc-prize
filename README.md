# ARC Prize 2025 – Neuro-Symbolic Solver

This project is focused on solving tasks from the [ARC (Abstraction and Reasoning Corpus)](https://github.com/fchollet/ARC) using a combination of **symbolic algorithms**  and **neural models**, integrating them in a neuro-symbolic framework. The goal is to develop systems that generalize from few examples and mimic human-like reasoning.



## 📂 File Structure

|Category|File / Directory|Description|
|---|---|---|
|**Core Logic & Execution**|`Arc_Prize_soln.py`|The main script that implements the core RL solution logic and generates outputs.|
||`Arc_Prize_pretraining.py`|Script for supervised pre-training machine learning models on ARC-like tasks.|
||`d_models/`|A directory containing the dl models architecture and training code|
||`dsl.py`|Defines the custom Domain-Specific Language (DSL) for grid transformations.|
|**Helper Modules**|`helper.py`|A general-purpose module with utility functions used across the project.|
||`helper_arc.py`|Contains specific helper functions for handling ARC task data structures |
||`helper_env.py`|Contains specific helper functions  related to  the environment|
|**Data & Models**|`weights/`|Directory containing the saved weights for our trained models.|
||`generated_training_data.pkl`|A pickled file containing synthetically generated data used to train the models.|
|**Output & Logs**|`data_loader/`|Contains images of the generated datets|
||`classifier_predictions/`|Default directory for storing the output predictions images from the classifier model.|
||`likelihood_predictions/`|Default directory for storing the output images from the likelihood model.|
||`app.log`|A log file for tracking execution, debugging information, and errors.|
|**Miscellaneous**|`IDEAS.md`|A document containing brainstorming notes and future ideas.|
||`submission.py`|A helper script used to format and package the final solution for submission.|
||`.gitignore`|Specifies which files and directories to exclude from Git version control.|
||`__pycache__/`|Directory for Python's cached bytecode.|
||`arc-prize-2025/`|Directory  holding data to the 2025 ARC Prize.|
||`README.md`|This file.|





**To be added .**
---

