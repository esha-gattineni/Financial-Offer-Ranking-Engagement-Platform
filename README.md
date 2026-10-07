# Financial Offer Ranking & Engagement Platform

A production-oriented machine learning platform that predicts which financial experience a customer is most likely to engage with next - such as a **credit card offer, personal loan, rewards promotion, savings product, or personalized financial insight**.

The system combines historical engagement signals, contextual features, TensorFlow modeling, automated validation, real-time serving, and load testing into one end-to-end workflow.

> **1M+ interaction records · 0.6915 ROC-AUC · 0.3630 PR-AUC · 795 req/s · 43 ms p95 · 0% failures**

---

## Overview

Banks often have multiple relevant experiences they could show a customer at the same time:

- Credit card upgrade
- Personal loan offer
- Cashback promotion
- Savings or CD offer
- Rewards redemption opportunity
- Personalized financial insight
- Retention or lifecycle message

The challenge is not simply deciding what offers exist.

The harder problem is:

> **Which financial experience is most relevant to this customer, in this context, right now?**

This project addresses that problem by building a real-time engagement scoring system that predicts the probability of customer interaction and can be used to rank candidate experiences before they are shown.

---

## System Architecture

```text
Raw Customer Interaction Data
            │
            ▼
     Dataset Construction
            │
            ▼
 Context + Product Features
            │
            ▼
 Historical Engagement Features
            │
            ▼
Chronological Train / Validation Split
            │
            ▼
      TensorFlow Training
            │
            ▼
       Model Evaluation
            │
            ▼
   Automated Validation Gate
            │
       PASS │ FAIL
            │
            ▼
       SavedModel Export
            │
            ▼
    TensorFlow Serving
            │
            ▼
         FastAPI
            │
            ▼
Real-Time Engagement Probability
            │
            ▼
    Insight Ranking
```

---

## Example Use Case

A customer opens a mobile banking application.

The platform may have several candidate experiences available:

```text
Customer opens app
        ↓
Candidate experiences
        ↓
┌───────────────────────────────┐
│ Credit card upgrade           │
│ Personal loan offer           │
│ Cashback reward               │
│ Savings promotion             │
│ Personalized spending insight │
└───────────────────────────────┘
        ↓
Engagement scoring model
        ↓
Probability assigned to each option
        ↓
Rank experiences
        ↓
Surface the most relevant one
```

The same scoring layer can be used across multiple customer-facing banking experiences while keeping the underlying ranking logic consistent.

---

## Model Performance

| Model | ROC-AUC | PR-AUC |
| --- | ---: | ---: |
| Chronological Baseline | 0.6077 | 0.2645 |
| Historical Logistic Regression | 0.6832 | 0.3471 |
| TensorFlow Model | **0.6915** | **0.3630** |

The largest improvement came from introducing leakage-safe historical engagement features across offers, campaigns, products, and prior customer interactions.

---

## Serving Performance

The complete **FastAPI → TensorFlow Serving** inference path was load-tested with concurrent HTTP traffic.

### Test Configuration

- **Total requests:** 5,000
- **Concurrent clients:** 25
- **Successful responses:** 100%

### Results

| Metric | Result |
| --- | ---: |
| Throughput | **795.3 req/s** |
| Average latency | **31.2 ms** |
| p50 latency | **30.2 ms** |
| p95 latency | **43.2 ms** |
| p99 latency | **62.7 ms** |
| Failure rate | **0%** |

The serving path remained comfortably below the project target of **100 ms p95 latency**.

---

## Banking Applications

This architecture can support ranking and personalization across several financial experiences.

### Credit Cards

- Card upgrades
- Cashback products
- Travel-rewards cards
- Balance-transfer promotions
- Credit-line offers

### Lending

- Personal loans
- Auto loans
- Refinance opportunities
- Credit-line increases

### Rewards

- Cashback opportunities
- Merchant rewards
- Travel redemptions
- Points promotions

### Savings & Deposits

- Savings products
- CDs
- Promotional deposit offers
- Account upgrades

### Financial Insights

- Spending insights
- Savings recommendations
- Account activity alerts
- Personalized budgeting suggestions

### Lifecycle Engagement

- Customer onboarding
- Retention
- Re-engagement
- Cross-sell
- Next-best-action recommendations

---

## Automated Model Validation

Every retrained model must pass quality thresholds before it is exported for serving.

```text
ROC-AUC >= 0.68
PR-AUC  >= 0.34
```

The validation flow:

```text
Retrained Model
      │
      ▼
Model Evaluation
      │
      ▼
Validation Gate
      │
   ┌──┴──┐
   │     │
 PASS   FAIL
   │     │
   ▼     ▼
Export  Keep Existing Model
```

This prevents an underperforming candidate model from replacing the current serving version.

---

## Real-Time Serving

The final TensorFlow model is exported as a SavedModel and served using TensorFlow Serving.

```text
Client Request
      │
      ▼
FastAPI :8000
      │
      ▼
TensorFlow Serving :8501
      │
      ▼
TensorFlow Model
      │
      ▼
Engagement Probability
```
---

## Automated Retraining

The retraining workflow uses the existing processed 1M-record dataset.

```text
Existing Dataset
      │
      ▼
TensorFlow Retraining
      │
      ▼
Candidate Evaluation
      │
      ▼
Validation Gate
      │
   ┌──┴──┐
   │     │
 PASS   FAIL
   │     │
   ▼     ▼
Export  Reject Candidate
```

The workflow handles:

- Model retraining
- ROC-AUC and PR-AUC evaluation
- Validation threshold checks
- SavedModel export only after a successful validation

---


Prediction:

```bash
curl -X POST \
  http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @serving/api_test_request.json
```

---

## Testing

API tests are implemented using PyTest.

Run:

```bash
PYTHONPATH=. python -m pytest -v
```

Tests cover:

- Root endpoint
- Input validation
- Health endpoint
- Missing feature handling

---

## Tech Stack

**Machine Learning:** TensorFlow, scikit-learn, TFX, TensorFlow Transform, TensorFlow Data Validation

**Data:** Python, Pandas, NumPy, PyArrow, Parquet, Apache Beam

**Serving:** FastAPI, TensorFlow Serving, REST APIs

**Infrastructure:** Docker, Docker Compose

**Performance:** C++, HTTP load testing, p50/p95/p99 latency analysis

**Testing:** PyTest

---
## Core Idea

The platform moves from:

> **“What financial products or offers are available?”**

to:

> **“Which financial experience is most relevant to this customer right now?”**

The project focuses not only on model accuracy, but also on the engineering required to make an ML model usable in a real system: **feature correctness, leakage prevention, validation, retraining, serving, testing, and low-latency inference**.
