**OlistXP — End-to-End A/B Experimentation Platform**

An end-to-end A/B testing and experimentation platform built on Databricks, PySpark, and Delta Lake — simulating a production-grade experimentation system used by companies like Amazon, Netflix, and Meta.

The project tests whether a **simplified checkout flow** improves conversion rate and revenue for an e-commerce platform, using the real-world Olist Brazilian E-Commerce dataset.

**[View Live Dashboard →](https://public.tableau.com/app/profile/tan.p5354/viz/OlistXP_AB_Test_Dashboard/Dashboard2#1)**
---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [What Each Layer Does](#what-each-layer-does)
6. [Experiment Design](#experiment-design)
7. [Statistical Methodology](#statistical-methodology)
8. [Results](#results)
9. [Dashboard](#dashboard)
10. [Simulation Limitations](#simulation-limitations)
11. [How This Works in a Real Organization](#real-world)
12. [How to Run](#how-to-run)
13. [Future Extensions](#future-extensions)
14. [Tech Stack](#tech-stack)

---

## Project Overview

OlistXP simulates a real product experiment where users are randomly assigned to:

- **Control Group** → existing multi-step checkout flow
- **Treatment Group** → simplified 3-step checkout flow

The goal is to determine whether the simplified checkout causes a statistically significant improvement in conversion rate and revenue, without harming other business metrics.

**Business Question:** Does simplifying the checkout flow increase the percentage of users who complete their purchase, and does the overall revenue impact justify shipping the change to all users?

---

## Dataset

**Source:** [Olist Brazilian E-Commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — Kaggle

**Size:** ~100,000 real orders placed between 2016 and 2018 across 9 relational tables.

| Table | Rows | Description |
|---|---|---|
| olist_orders_dataset | 99,441 | Order lifecycle and timestamps |
| olist_customers_dataset | 99,441 | Customer geography and IDs |
| olist_order_items_dataset | 112,650 | Items, prices, freight per order |
| olist_order_payments_dataset | 103,886 | Payment methods and values |
| olist_order_reviews_dataset | 104,162 | Customer satisfaction scores |
| olist_products_dataset | 32,951 | Product catalog and categories |
| olist_sellers_dataset | 3,095 | Seller information |
| olist_geolocation_dataset | 1,000,163 | Zip code coordinates |
| product_category_name_translation | 71 | Portuguese to English category names |

---

## Architecture

```
Raw CSVs (9 Olist tables)
        │
        ▼
┌─────────────────────────────────────────┐
│  BRONZE LAYER                           │
│  Raw ingestion → Delta Lake             │
│  Schema enforcement · Audit columns     │
│  Corrupt row flagging                   │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  SILVER LAYER                           │
│  Cleaning · Joining · Feature Engineering│
│  9 tables → 1 master order_fact table   │
│  99,441 rows · 32 columns               │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  GOLD LAYER                             │
│  Experiment Assignment · Simulation     │
│  Hash-based user bucketing              │
│  Treatment effect simulation            │
│  User-level metrics table               │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  STATISTICAL ANALYSIS                   │
│  SRM check · z-test · t-test            │
│  Confidence intervals · p-values        │
│  CUPED variance reduction               │
│  Segmentation analysis                  │
└─────────────────────┬───────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────┐
│  TABLEAU DASHBOARD                      │
│  KPI cards · CVR comparison             │
│  Revenue analysis · Segmentation        │
│  Statistical summary table              │
└─────────────────────────────────────────┘
```

---

## Project Structure

```
olist-xp/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 02_bronze_silver_layer.py     ← Bronze ingestion + Silver transformation
│   ├── 03_gold_experiment.py         ← Experiment assignment + simulation
│   └── 04_statistical_analysis.py   ← Full statistical test suite
│
└── docs/
    └── dashboard_screenshot.png      ← Tableau dashboard preview
```

---

## What Each Layer Does

### Bronze Layer
The Bronze layer is a faithful, unchanged copy of the raw source data stored in Delta Lake format.

**What we did:**
- Read each of the 9 CSV files individually with explicit schema definitions — never using `inferSchema` to prevent incorrect type assumptions
- Added three audit columns to every table: `_ingestion_timestamp`, `_source_file`, and `_is_corrupt`
- Flagged rows where the primary key is null using `_is_corrupt` without deleting any rows — Bronze is an archive
- Saved all 9 tables to Delta Lake with `mode("overwrite")` for idempotent writes
- Verified all 9 tables against expected row counts — all passed

**Key design decision:** Bronze never deletes or modifies data. Even corrupt rows are kept and flagged. This ensures full auditability and the ability to re-process from source at any time.

---

### Silver Layer
The Silver layer cleans each table individually, then joins them into one master analytics table.

**What we did:**

**Cleaning:**
- Reviews: dropped 2,236 rows with null `order_id` (unusable without an order reference), used `try_cast` to convert `review_score` from string to integer (some rows contained timestamps instead of scores due to a source data error), replaced invalid score of 0 with null
- Order items: aggregated from 112,650 rows to 98,666 rows — one row per order (summing price, freight, counting items)
- Payments: aggregated from 103,886 rows to 99,440 rows — one row per order (summing payment values)
- Products: renamed two misspelled columns (`product_name_lenght` → `product_name_length`), left joined English category translations

**Joining:**
- Joined all cleaned tables into one master `order_fact` table using left joins throughout — preserving all 99,441 orders even when some tables had no matching rows
- Discovered and fixed duplicate order_ids in reviews that caused row inflation from 99,441 to 99,992 — deduplicated using a window function keeping the most recent review per order

**Feature Engineering:**
Added 8 new columns to support experiment analysis:

| Feature | Description |
|---|---|
| `is_converted` | 1 if order_status = delivered, else 0 |
| `days_to_delivery` | Days from purchase to delivery |
| `is_late` | 1 if delivered after estimated date |
| `total_order_value` | total_item_price + total_freight |
| `freight_ratio` | total_freight / total_item_price |
| `is_weekend` | 1 if order placed on Saturday or Sunday |
| `order_hour` | Hour of day the order was placed |
| `is_first_order` | 1 if this is the customer's first ever order (window function) |

**Final Silver table:** 99,441 rows × 32 columns saved as `silver/order_fact_featured`

---

### Gold Layer
The Gold layer filters to the experiment window, assigns users to groups, simulates treatment effects, and builds the final user-level metrics table.

**What we did:**

**Filtering:**
- Filtered Silver master table to January 1–31 2018 (experiment window)
- 99,441 orders → 7,013 experiment orders → 6,918 unique eligible users

**Hash-based Experiment Assignment:**
- Used MD5 hashing for deterministic, reproducible assignment
- Formula: `hash(customer_unique_id + experiment_id + salt) % 100`
- Bucket 0–49 → treatment, Bucket 50–99 → control
- Same user always gets the same group — consistent across sessions and devices
- Registered as a Spark UDF for distributed execution across the cluster
- Result: control 3,470 users (50.2%) / treatment 3,448 users (49.8%)

**Treatment Effect Simulation:**
- `BASELINE_CVR = 0.75` — 75% of control users convert
- `TREATMENT_CVR_LIFT = 0.04` — treatment users have a 79% conversion threshold
- `TREATMENT_AOV_CHANGE = -0.02` — treatment order values reduced by 2% (simulating removal of upsell prompts)
- Used `rand(seed=42)` for reproducible randomness — same results every run
- Each user gets a random number between 0 and 1; if it falls below their group's threshold they convert

**User Metrics Table:**
Aggregated to one row per user:
- `converted` — did the user convert at all during the experiment (max)
- `total_revenue` — total revenue from all their orders
- `num_orders`, `avg_review_score`, `avg_delivery_days`, `late_delivery_rate`

---

### Statistical Analysis Layer
Full hypothesis testing suite following industry best practices.

**Tests run in order:**

**1. SRM Check (always first)**
Chi-square test on observed split vs expected 50/50. If p < 0.01, all results are invalid and analysis stops. Our result: p = 0.7914 — no SRM detected.

**2. Two-proportion z-test on CVR (primary metric)**
Tests whether the difference in conversion rates between groups is statistically significant.

**3. Welch's t-test on AOV (secondary metric)**
Tests whether average order value differs between groups. Welch's variant used because revenue distributions have unequal variance between groups.

**4. Welch's t-test on Revenue per User (guardrail metric)**
Most important business metric — captures the combined effect of CVR and AOV. All users included with non-converted revenue set to zero.

**5. CUPED Variance Reduction**
Applied pre-experiment covariate adjustment using `avg_review_score` to reduce metric variance and improve statistical power. Variance reduced by 0.8% — result remained significant.

**6. Segmentation Analysis**
Broke down CVR lift by user segments to check consistency of treatment effect across subgroups.

---

## Experiment Design

| Parameter | Value |
|---|---|
| Experiment ID | checkout_simplification_v1 |
| Hypothesis | Simplified checkout reduces friction and increases CVR |
| Primary metric | Conversion Rate (CVR) |
| Secondary metrics | Average Order Value (AOV) |
| Guardrail metrics | Revenue per User, Review Score |
| Randomization unit | customer_unique_id (user level) |
| Traffic split | 50% Control / 50% Treatment |
| Experiment window | January 1–31 2018 |
| Significance level | α = 0.05 (two-sided) |
| Assignment method | MD5 hash-based deterministic bucketing |

**Why user-level randomization?**
We assign at the user level not the order level because we are testing the checkout experience on people, not individual transactions. A user who places three orders should experience the same checkout all three times.

**Why hash-based assignment?**
Random assignment is non-deterministic — the same user could land in different groups on different runs. Hash-based assignment guarantees the same user always maps to the same bucket, making the experiment reproducible and preventing cross-contamination.

**Why left joins in Silver?**
Left joins preserve all 99,441 orders even when some tables have no matching row. Inner joins would silently drop orders with missing payment or review data, biasing the dataset toward only "perfect" orders.

**Why filter to experiment window in Gold, not Silver?**
Silver is a general-purpose analytics layer reusable for any analysis — yearly dashboards, category analysis, seller performance. Filtering to January 2018 in Gold keeps Silver flexible. Additionally, CUPED requires pre-experiment data from November–December 2017 which would be lost if Silver was filtered.

---

## Statistical Methodology

### Hypothesis

```
H₀: CVR_treatment = CVR_control   (no effect)
H₁: CVR_treatment ≠ CVR_control   (two-sided)
```

We use a two-sided test because we make no assumption about direction — the simplified checkout could theoretically hurt conversion if users valued the removed features.

### Why SRM check runs first
Sample Ratio Mismatch invalidates all statistical conclusions regardless of p-values. If the assignment pipeline has a bug causing a 60/40 split instead of 50/50, the groups are not comparable and no amount of statistical testing can fix that. SRM check is the gate that must pass before any analysis proceeds.

### Why CUPED
CUPED (Controlled-experiment Using Pre-Experiment Data) reduces metric variance by regressing out pre-experiment user behavior, improving statistical power without requiring more users. This is standard practice at Booking.com, Netflix, and Airbnb.

### Why Welch's t-test instead of Student's t-test
Revenue distributions are right-skewed and the two groups have unequal variance. Welch's t-test does not assume equal variance, making it more appropriate and robust for this use case.

---

## Results

| Metric | Control | Treatment | Lift | P-value | Result |
|---|---|---|---|---|---|
| SRM check | 50.2% | 49.8% | — | 0.7914 | PASS |
| Conversion Rate | 75.36% | 78.80% | +3.44pp (+4.56%) | 0.000668 | SIGNIFICANT |
| Average Order Value | R$146.70 | R$160.36 | +R$13.66 | 0.008459 | SIGNIFICANT |
| Revenue per User | R$109.79 | R$125.62 | +R$15.83 | 0.000214 | SIGNIFICANT |
| CUPED Revenue | — | — | — | 0.000353 | SIGNIFICANT |

**95% Confidence Interval on CVR lift: [+1.46pp, +5.42pp]**

The entire confidence interval is above zero — we are 95% confident the true effect is positive.

### Segmentation Results

| Segment | Control CVR | Treatment CVR | Lift | Users |
|---|---|---|---|---|
| Single order users | 75.1% | 78.6% | +3.5pp | 6,825 |
| Repeat buyers | 92.9% | 100.0% | +7.1pp | 93 |
| On-time delivery | 75.6% | 78.6% | +3.0pp | 6,468 |
| Late delivery users | 72.4% | 82.0% | +9.6pp | 450 |

Treatment effect is positive and consistent across all segments. Users who experienced late deliveries show the largest lift — removing checkout friction has the biggest impact on already-frustrated users.

**Recommendation: Ship the simplified checkout to 100% of users.**

---

## Dashboard

Built in Tableau Public. The dashboard tells the complete experiment story across 5 views:

- KPI Summary — total users, CVR comparison, revenue comparison
- Conversion Rate by Variant — bar chart with clear visual lift
- Revenue per User by Variant — business impact visualization
- Segmentation Analysis — lift breakdown by user segment
- Statistical Summary Table — all test results with significance indicators

![Dashboard Preview](docs/dashboard_screenshot.png)

---

## Simulation Limitations

This project uses **simulated treatment effects**, which is a fundamental limitation of working with historical data where no real experiment was conducted.

**What is simulated:**
- The direction of the effect (we assumed treatment would be better)
- The magnitude of the lift (we chose 4% CVR improvement)
- The baseline conversion rate (we set 0.75 based on delivered order ratio)
- The AOV change (we set -2% based on upsell removal logic)

**What is real:**
- The dataset (100,000 real Brazilian e-commerce orders)
- The user IDs and their behavior patterns
- The statistical methodology and pipeline
- The engineering architecture

**Why this is still valid for a portfolio project:**
The value of this project is not in the specific numbers — those are predetermined by simulation assumptions. The value is in demonstrating a complete understanding of experiment design, statistical methodology, and production-grade data engineering. In a real role, the pipeline and methodology would be identical — the only difference is real user behavior would replace the simulated effects.

---

## How This Works in a Real Organization <a name="real-world"></a>

In a real organization, the process would never start with assumptions. It would follow a discovery-first approach:

**Step 1 — Quantitative research before building anything**
Analyse existing funnel data to find where users are actually dropping off. If 85% of users reach the payment page but only 65% complete the order, the problem is in payment — not earlier steps. Data tells you where to look before you decide what to change.

**Step 2 — Qualitative research**
Talk to users who abandoned checkout. Watch session recordings. Numbers tell you what is happening — users tell you why. "I got confused by too many options" points to simplification. "I wanted to see more product suggestions" would point in the opposite direction entirely.

**Step 3 — Form multiple competing hypotheses**
Never assume one direction. In our scenario, the alternative hypothesis — that users actually value product suggestions and removing them would hurt conversion — is equally reasonable and should be tested separately.

**Step 4 — Define success AND failure criteria upfront**
Before running any experiment, write down exactly what would constitute success and what would constitute failure. This prevents changing the definition of success after seeing the data (p-hacking).

**Step 5 — Let the data decide**
Run the experiment with no predetermined outcome. The data tells you which hypothesis is correct. A negative result is not a failure — it is valuable information that prevents shipping a harmful change.

**Step 6 — Guardrail metrics catch unexpected downsides**
Even if CVR improves, guardrail metrics like revenue per user and satisfaction score must not regress. If CVR goes up but revenue per user drops 10%, the experiment does not ship regardless of the primary metric result.

---

## How to Run

### Prerequisites
- Databricks account (Community Edition works)
- Olist dataset downloaded from Kaggle
- Python 3.x with scipy, pandas, numpy

### Setup
```bash
git clone https://github.com/YourUsername/olist-xp.git
cd olist-xp
```

### Notebook execution order
Run notebooks in this exact order:

```
1. notebooks/02_bronze_silver_layer.py
   → Upload Olist CSVs to Databricks volume first
   → Update BASE_PATH to match your volume path
   → Runs Bronze ingestion + Silver transformation
   → Creates silver/order_fact_featured

2. notebooks/03_gold_experiment.py
   → Update SILVER_PATH and GOLD_PATH
   → Runs experiment assignment + simulation
   → Creates gold/user_experiment_metrics

3. notebooks/04_statistical_analysis.py
   → Update GOLD_PATH
   → Runs full statistical test suite
   → Exports CSVs for Tableau dashboard
```

### Path configuration
Update these paths in each notebook to match your Databricks volume:
```python
BASE_PATH   = "dbfs:/Volumes/workspace/default/ab_esting/"
BRONZE_PATH = BASE_PATH + "bronze/"
SILVER_PATH = BASE_PATH + "silver/"
GOLD_PATH   = BASE_PATH + "gold/"
```

---

## Future Extensions

**Statistical enhancements:**
- Sensitivity analysis with different baseline CVR values (0.30, 0.50, 0.70) to test robustness of conclusions
- Sequential testing with alpha-spending functions to allow safe early stopping
- Bayesian A/B testing approach using Beta-Binomial conjugate model
- Multi-armed bandit to replace static 50/50 split with dynamic traffic allocation

**Engineering enhancements:**
- Databricks Workflows to orchestrate notebooks into an automated pipeline with scheduling and failure alerting
- Great Expectations for automated data quality checks at each layer
- MLflow experiment tracking for logging parameters and metrics
- Pre-experiment CUPED using actual purchase history instead of review scores

**Business extensions:**
- Long-term holdout analysis measuring 90-day retention effect
- Interaction detection between concurrent experiments
- Experiment ramp-up strategy (1% → 5% → 20% → 50% → 100%)
- Personalization experiment using recommendation algorithm testing

---

## Tech Stack

| Component | Technology |
|---|---|
| Compute | Databricks Community Edition |
| Storage | Delta Lake (Parquet) |
| Processing | PySpark 3.x, Spark SQL |
| Statistics | Python — scipy, statsmodels, numpy, pandas |
| Dashboard | Tableau Public |
| Version control | GitHub |
| Dataset | Olist Brazilian E-Commerce (Kaggle) |

---

## Resume Bullets

**For Product Data Scientist roles:**
- Designed and implemented an end-to-end A/B experimentation platform on Databricks, processing 100K+ e-commerce transactions using PySpark and Delta Lake with hash-based deterministic user assignment ensuring experiment integrity
- Applied CUPED variance reduction and ran a full statistical test suite including two-proportion z-test, Welch's t-test, SRM detection, and segmentation analysis — detecting a statistically significant 3.44pp CVR lift (p=0.0007) with 95% CI [+1.46pp, +5.42pp]

**For Analytics Engineer roles:**
- Architected a medallion data lakehouse (Bronze → Silver → Gold) on Delta Lake using PySpark, joining 9 relational tables with 100K+ records into an analytics-ready experiment metrics table, with schema enforcement, data quality flagging, and idempotent Delta writes

**For Data Engineer roles:**
- Built a scalable ETL pipeline on Databricks ingesting multi-table Olist e-commerce data into a Delta Lake medallion architecture with explicit schema enforcement, audit columns, SRM-validated experiment assignment, and automated row count verification

---

*Project by Thanishka | OlistXP — A/B Experimentation Platform*
*Dataset: Olist Brazilian E-Commerce | Kaggle CC BY-NC-SA 4.0***
