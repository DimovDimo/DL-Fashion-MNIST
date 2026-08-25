# Start Here

> Author: **Dimo Dimov**

This document introduces the project and explains how it was developed.

It covers two things in particular:

1. **Why the Fashion-MNIST dataset was chosen**: including the two datasets that were evaluated and rejected on data-quality grounds.
2. **How the repository is organized**: what each numbered folder contains and in which order the material should be read.

> This is an **educational** project, not a clinical or commercial one. It is nevertheless documented to the standard that would be expected of a research study.

---

## Contents

- [1. Project Overview](#1-project-overview)
- [2. Dataset Selection: Why Fashion-MNIST?](#2-dataset-selection-why-fashion-mnist)
  - [2.1 First candidate: ECG heartbeat categorization (rejected)](#21-first-candidate-ecg-heartbeat-categorization-rejected)
  - [2.2 Second candidate: MedMNIST (rejected)](#22-second-candidate-medmnist-rejected)
  - [2.3 Final choice: Fashion-MNIST](#23-final-choice-fashion-mnist)
- [3. Getting Started](#3-getting-started)
- [4. Repository Structure](#4-repository-structure)
- [5. Notebook Version History](#5-notebook-version-history)
- [6. License](#6-license)
- [References](#references)

---

## 1. Project Overview

This repository contains a complete, end-to-end image-classification study built on the **Fashion-MNIST** dataset: exploratory data analysis, classical machine-learning baselines, deep-learning models, model ensembling, a full evaluation protocol, saved model artifacts, an interactive web application, and a scientific write-up of the results.

Rather than presenting only the final notebook, the project documents its **entire development history**: the dataset candidates that were considered and rejected, the preregistration written before the experiments were run, each revision of the analysis notebook, and the modular code that the notebook was refactored into.

The recommended way to read the project is folder by folder, in numerical order, starting from this document.

---

## 2. Dataset Selection: Why Fashion-MNIST?

Choosing the dataset took considerably longer than expected. Two medically themed datasets were evaluated in depth and both were **rejected because of data leakage**: the problem where information about the target variable reaches the model through channels other than genuine biological signal, inflating validation metrics and making results meaningless.

| Candidate | Outcome | Reason for rejection |
| --- | --- | --- |
| **ECG heartbeat** (Kaggle) | Rejected | Intra-patient split: individual heartbeats from the same patient appear in both train and test |
| **DermaMNIST** (MedMNIST) | Rejected | Multiple images of the same lesion span the splits, despite the authors' stated leakage controls |
| **PathMNIST** (MedMNIST) | Rejected | Patch-level rather than patient-level split, plus strong color and compression artifacts that leak the labels |
| **Fashion-MNIST** | **Selected** | No grouping structure to leak through, a fixed and widely used train/test split, and no domain-specific confounders |

### 2.1 First candidate: ECG heartbeat categorization (rejected)

The project began as an **ECG heartbeat categorization** study ([DimovDimo/ECG-Heartbeat](https://github.com/DimovDimo/ECG-Heartbeat)), using the [Kaggle *Heartbeat* dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat). It was abandoned during the data-validation stage, because the dataset suffers from leakage.

The core problem is what is known as an **intra-patient split**. Individual heartbeats recorded from the same person are extremely similar to one another: they share the same cardiac morphology, electrode placement, and recording characteristics. In this dataset, records were assigned to the train and test partitions **at the level of an individual heartbeat, chosen at random, rather than at the level of the patient**.

The consequence is that the same patient can contribute beats to both partitions. A model is then free to learn the idiosyncratic signature of a specific recording instead of the clinically meaningful features that distinguish beat classes, and the resulting test scores no longer measure generalization.

No standardized, community-accepted procedure existed for constructing a proper patient-level (**inter-patient**) split for this dataset, so an alternative dataset was sought instead.

### 2.2 Second candidate: MedMNIST (rejected)

The next candidate was **[MedMNIST](https://medmnist.com/)**, a collection of standardized biomedical image datasets. It appeared promising at first, but investigation revealed that several of its sub-datasets carry their own data problems.

#### DermaMNIST

The MedMNIST authors state that official or patient-level splits were used across the collection in order to prevent leakage. For **DermaMNIST**, however, this was not achieved correctly. The underlying source data has a **multi-image-per-lesion** structure: several photographs can exist for a single lesion. When images are assigned to splits individually, different images of the *same lesion* end up on opposite sides of the boundary: reintroducing exactly the leakage the split was meant to prevent.

#### PathMNIST

Among all MedMNIST sub-datasets, **PathMNIST** had the fewest problems, so it became the leading candidate. It too proved unsuitable, for two distinct reasons.

**1. Patient-level leakage.**
PathMNIST is derived from **NCT-CRC-HE-100K**, which contains 100,000 image patches taken from 86 patients with colorectal cancer. The training and validation partitions are split at a ratio of approximately 9:1, but the split is performed **at the level of individual image patches rather than at the level of patients or slides**.

As a result, the same slide and the same patient can appear in both partitions. Patches drawn from a single slide are highly correlated with one another (identical staining, similar processing, comparable tissue morphology) so the model learns to recognize characteristics specific to particular patients instead of the general features of each tissue class. Validation metrics are therefore optimistically inflated, and any hyperparameter tuning performed against them is unreliable.

**2. Shortcut learning and confounding factors (the more serious problem).**
Beyond the split, the dataset contains strong **non-biological biases that leak label information**. A model does not need to understand histopathology at all in order to score well on it:

- **Insufficient color normalization.** Macenko's method was not applied rigorously enough, leaving clear per-class color signatures in the data. Mean RGB intensities alone reach **above 50 %** accuracy across the nine classes, and a color histogram reaches **above 82 %** accuracy: with no deep learning involved whatsoever.
- **JPEG compression artifacts.** The compression artifacts differ systematically between classes and are straightforward for a model to detect.
- **Corrupted and overexposed patches.** Some patches are damaged by errors in dynamic-range handling, and the damage correlates with class membership.

### 2.3 Final choice: Fashion-MNIST

Because both medically themed candidates turned out to be compromised, the decision was made to select a dataset that is as free of data problems as possible. **[Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)** was chosen.

Fashion-MNIST is a benchmark of 70,000 grayscale 28×28 images of fashion articles across 10 categories, with a fixed, officially published 60,000 / 10,000 train–test split. It has no patient or slide structure that could leak across the split, no staining or scanner confounders, and it is widely used enough that reported results can be compared directly against published baselines.

> The dataset-rejection analysis above is retained deliberately. Recognizing leakage is a core skill in applied machine learning, and documenting why a dataset was *not* used is as informative as documenting why one *was*.

---

## 3. Getting Started

A `requirements.txt` file is provided in the root of the project. Installing the dependencies listed there before running any code is strongly recommended.

---

## 4. Repository Structure

The project lives in the **`Project`** folder, which contains numbered sub-folders. Reading them in numerical order is the recommended approach, since each one builds on the previous.

| Folder | Contents |
| --- | --- |
| **1. Start Here** | This folder: the presentation and orientation document for the project. |
| **2. OSF-Style Preregistration** | A preregistration written in the style used by the Open Science Framework. It is **not registered with OSF**, because this is an educational rather than a research project: but it was prepared to the same professional standard as a genuine research preregistration. |
| **3. Unified Notebook** | `DL-Fashion-MNIST.ipynb`, which contains the entire project in a single file. See the [version history](#5-notebook-version-history) below. |
| **4. Split Notebooks** | The project refactored into separate `.ipynb` files. `restructure.py` is the tool that performs the split automatically. The `src` folder holds the original, unmodified code and is preserved as a reference point in case other code changes. The `notebooks` folder holds the resulting `.ipynb` files, which were created and kept in sync with paired `.py` files using [Jupytext](https://github.com/mwouts/jupytext) in `py:percent` format. |
| **5. Trained Models** | The saved, already-trained models. |
| **6. Gradio App** | A [Gradio](https://www.gradio.app/) web application, launched with `app.py`. |
| **7. Exported Formats** | The project exported into several alternative file formats. |
| **8. Scientific Paper** | A scientific article written about the project. It is **not published and has not undergone peer review**; it exists to present the project in a scholarly form. |
| **9. Readme Resources** | The files referenced by `README.md`. |

---

## 5. Notebook Version History

The unified notebook `DL-Fashion-MNIST.ipynb` went through four versions:

| Version | Changes |
| --- | --- |
| **v1** | The initial version of the project. |
| **v2** | Added exploratory data analysis, machine learning, deep learning, ensembling, and evaluation of results. |
| **v3** | Added functions for saving trained models. |
| **v4** | Fixed a bug in one unit test. See the note below. |

> **On the unit test count.** The v4 fix concerned a unit test that failed because **10 duplicate images were found across the Fashion-MNIST training and test sets**. Those 10 duplicates were removed, after which the test passed.
>
> As a consequence, the project notes refer both to **27 unit tests** (the number passing before the fix) and to **28 unit tests**, which is the full suite, all passing. The two figures are the same suite measured before and after the v4 correction, not two different test suites.

---

## 6. License

This project is released under the **MIT License**.

---

## References

1. Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* [arXiv:1708.07747](https://arxiv.org/abs/1708.07747)
2. Fazeli, S. *Heartbeat* dataset. Kaggle. [https://www.kaggle.com/datasets/shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
3. Yang, J., Shi, R., Wei, D., et al. (2021). *MedMNIST v2: A Large-Scale Lightweight Benchmark for 2D and 3D Biomedical Image Classification.* [arXiv:2110.14795](https://arxiv.org/abs/2110.14795) · [https://medmnist.com/](https://medmnist.com/)
4. Kather, J. N., et al. (2018). *100,000 histological images of human colorectal cancer and healthy tissue* (NCT-CRC-HE-100K). Zenodo. [https://doi.org/10.5281/zenodo.1214456](https://doi.org/10.5281/zenodo.1214456)
5. Wouts, M. *Jupytext: Jupyter notebooks as plain text scripts.* [https://github.com/mwouts/jupytext](https://github.com/mwouts/jupytext)
6. Gradio. [https://www.gradio.app/](https://www.gradio.app/)
7. Open Science Framework. [https://osf.io/](https://osf.io/)

---
*Created as part of the SoftUni Deep Learning Course (July 2026).*