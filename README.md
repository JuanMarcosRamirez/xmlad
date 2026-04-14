# FINALITY Bootcamp 2026 Repository
## Robust & Explainable Anomaly Detection for Communication Networks
### From Statistical Foundations to Trustworthy AI Systems

[Vincenzo Mancuso](https://networks.imdea.org/es/team/imdea-networks-team/people/vincenzo-mancuso/) and [Juan Marcos Ramirez](https://juanmarcosramirez.github.io/)

## Abstract

This repository supports a hands-on bootcamp on **data preprocessing, anomaly detection, and explainable machine learning** with a focus on **telecom and network-performance data**. The material is organized as a sequence of Jupyter notebooks that guide participants from data inspection and preprocessing, rule-based and multivariate anomaly detection, to model-based interpretation and explainability. The goal of the bootcamp is not only to run the provided notebooks, but also to help participants **create their own modified notebook versions**, test alternative parameters, compare methods, and build intuition for how parameter choices affect anomaly-detection performance.

This repository is designed for:
- PhD students
- communication engineers
- researchers and practitioners who want a practical entry point into anomaly detection workflows for structured datasets.

---

## Repository goals

By working through this repository, participants should be able to:

- inspect and clean structured datasets for anomaly-detection tasks,
- compute descriptive statistics and visualize KPI distributions,
- understand why transformations such as `log10` can improve analysis,
- compare rule-based detectors such as **IQR**, **Z-score**, and **MAD**,
- apply multivariate methods such as **robust covariance**, **One-Class SVM**, **Isolation Forest**, and **Local Outlier Factor**,
- interpret confusion matrices, ROC curves, and standard evaluation metrics,
- understand the role of contamination, scaling, and anomaly scores,
- build and analyze a simple **residual-based anomaly-detection baseline**,
- connect model outputs with explainability and decision support.

---

## Who this repository is for

This repository is intended for participants who are comfortable with:
- Python basics,
- tabular data analysis,
- Jupyter Notebook workflows,
- core machine-learning concepts.

It is especially suitable for:
- doctoral students in data science, signal processing, AI, or telecommunications,
- communication engineers interested in KPI analysis,
- applied researchers who need a practical anomaly-detection workflow.

---

## Repository structure


```text
xmlad/
├── 01_preprocessing.ipynb
├── 02_anomaly_detection.ipynb
├── 03_explainability.ipynb
├── datasets/
├── srcs/
│   ├── __init__.py
│   ├── anmdet.py
│   ├── misc.py
│   ├── models.py
│   ├── plot_functions.py
│   └── preprocessing_functions.py
├── LICENSE
└── README.md
