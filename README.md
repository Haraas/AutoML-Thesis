# AutoML-Thesis

Code, data, and figures accompanying the diploma thesis
**"Database Performance Optimization using AutoML compared to existing ML4DB techniques."**

**Author:** Hara Stoumpou · **Supervisor:** Prof. Dimitrios Tsoumakos

## Overview

This repository contains the full material needed to reproduce the experiments of the
thesis. The work applies automated hyperparameter optimization (TPE, SMAC, and Random
Search) to the AVGDL learned query-optimization method within the BAO framework, and
evaluates it on the JOB-extend workload (1,697 IMDB queries) against the PostgreSQL
optimizer. Under a validation-based, multi-seed evaluation protocol, automated tuning
shows no statistically significant improvement over the manually tuned baseline; the
dominant effect on performance is the sensitivity of the BAO online learning loop to the
training seed.

## Repository structure

```
AutoML-Thesis/
├── Notebooks/        end-to-end notebook (.ipynb) and a plain-Python export (.py)
├── data/
│   ├── search/       per-optimizer search runs (SMAC/, TPE/, Random_Search/):
│   │                 six raw result files and six ranked "_final" files per optimizer
│   └── reruns/       multi-seed re-evaluation:
│                     Reruns.csv, Summary_of_Reruns.csv, Reruns.xlsx
├── figures/          generated diagrams used in the thesis
├── references/       Git submodule: upstream AVGDL / qp_evaluation code (commit 022ecd4)
└── .gitmodules
```

## Reproducing the results

The experiments were run on Google Colab (NVIDIA T4 GPU). Open the notebook in
`Notebooks/` and run it top to bottom; it downloads the upstream `qp_evaluation` code and
the data and reproduces the searches and the re-evaluation. The notebook defines the full
pipeline (the `BanditOptimizer` driver, the `run_avgdl_full` evaluator, and the
`run_experiment` / `run_experiment_smac` / `run_experiment_random` wrappers). All result
files are also stored under `data/`, so every table and figure can be regenerated without
re-running the computationally expensive search. The `references/` submodule pins the
upstream code at commit `022ecd4` for provenance; to fetch it locally, clone with
`--recurse-submodules`.

## Upstream work

The method and evaluation framework build on the comparative study of Zhao et al.
(PVLDB, 2023) and the BAO framework of Marcus et al. (SIGMOD, 2021). The upstream code is
included as the `references/` submodule, pinned at commit `022ecd4`.

## Release

The version accompanying the thesis is tagged **v1.0-thesis**.
