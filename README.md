# User Engagement Prediction Platform

Production-oriented machine learning pipeline for predicting user engagement on large-scale interaction data using historical behavioral signals, TensorFlow, TFX, TensorFlow Transform, and Apache Beam.

## Overview

This project builds an end-to-end engagement prediction workflow on top of the Outbrain Click Prediction dataset.

The system covers:

- large-scale dataset construction
- contextual and content feature engineering
- leakage-safe historical CTR features
- chronological model evaluation
- TensorFlow model training
- TensorFlow Data Validation
- TensorFlow Transform preprocessing
- Apache Beam execution
- TFX-based training pipeline

The current pipeline has been evaluated on more than **1 million interaction records**.

## Results

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| Chronological baseline | 0.6077 | 0.2645 |
| Historical Logistic Regression | 0.6832 | 0.3471 |
| TensorFlow Neural Network | **0.6908** | **0.3614** |

Historical engagement features improved:

- ROC-AUC by **+0.0755**
- PR-AUC by **+0.0826**

The TensorFlow model further improved:

- ROC-AUC by **+0.0076**
- PR-AUC by **+0.0143**

## Architecture

```text
Raw Interaction Data
        |
        v
Dataset Construction
        |
        v
Feature Engineering
        |
        +-- Context Features
        +-- Content Features
        +-- User History
        +-- Historical CTR Features
        |
        v
Chronological Train / Validation Split
        |
        v
TensorFlow Data Validation
        |
        v
TensorFlow Transform
        |
        v
Apache Beam
        |
        v
TensorFlow Training
        |
        v
TFX Pipeline
