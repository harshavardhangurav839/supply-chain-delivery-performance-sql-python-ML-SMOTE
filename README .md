# Supply Chain Delivery Performance Analysis

> End-to-end order fulfillment analysis identifying root causes of chronic late deliveries, quantifying their financial impact, and building a predictive model to flag at-risk orders — using Python and Machine Learning.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Tools & Technologies](#tools--technologies)
- [Methods](#methods)
- [Data Cleaning & Preparation](#data-cleaning--preparation)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Key Insights](#key-insights)
- [Machine Learning Model](#machine-learning-model)
- [Project Structure](#project-structure)
- [How to Run This Project](#how-to-run-this-project)
- [Final Recommendations](#final-recommendations)
- [Author & Contact](#author--contact)

---

## Overview

This project analyzes the delivery operations of a global e-commerce company managing end-to-end order fulfillment across multiple regions. The analysis covers **172,765 orders** spanning January 2015 through January 2018, focusing on identifying root causes of chronic late deliveries, quantifying their financial impact, and building a data-driven predictive framework for improvement.

---

## Problem Statement

A global e-commerce company operating across multiple regions manages end-to-end order fulfillment for products like sporting goods, fitness equipment, outdoor gear, footwear, and apparel. The company faces inconsistent delivery performance where actual shipping times frequently deviate from scheduled delivery windows, leading to:

- Eroded customer trust and unpredictable order profitability
- Inability to make reliable delivery commitments to buyers at point of purchase
- Significant financial drain from delayed shipments

---

## Dataset

- Source: `DataCoSupplyChainDataset.csv` stored in `/Data/` folder
- **172,765 orders** from January 2015 to January 2018
- Features include: order region, shipping mode, customer segment, department, payment type, order status, scheduled shipment days, and order profit per order

---

## Tools & Technologies

- **Python** — Data ingestion, cleaning, EDA, visualization, and machine learning
- **Pandas & NumPy** — Data manipulation and feature engineering
- **Matplotlib & Seaborn** — Charts: bar plots, pie charts, heatmaps, line plots
- **Scikit-learn** — Random Forest Classifier, train/test split, classification report
- **Imbalanced-learn (SMOTE)** — Handling class imbalance in ML pipeline
- **GitHub** — Version control and project hosting

---

## Methods

1. **Data Ingestion** — Loaded the CSV dataset using Pandas with Latin-1 encoding
2. **Data Cleaning** — Dropped irrelevant columns (PII fields like customer email, password, name, street, zipcode; product image and description)
3. **Feature Engineering**:
   - Calculated `order processing time` = shipping date − order date
   - Calculated `Delay` = order processing time − scheduled shipment days
   - Created `Is_Delayed` flag (Delay > 0)
   - Created `Profitability_flag` (Profitable / Not Profitable / Break Even) from Order Profit Per Order
   - Extracted `order_month`, `order_day`, `order_hour` from order date
4. **Exploratory Data Analysis** — Profitability distribution, delay distribution, profit vs delay days analysis
5. **Bottleneck Detection** — Computed delay percentage across 6 operational dimensions: Region, Segment, Shipping Mode, Order Status, Payment Type, Department
6. **Root Cause Analysis** — Deep-dive into the highest-delay region (Central Africa) to rank top driver factors
7. **Time-Based Analysis** — Analyzed delay patterns by month, day of week, and hour of day
8. **Machine Learning** — Built a Random Forest classifier with frequency encoding, stratified split, and SMOTE oversampling to predict late delivery risk

---

## Data Cleaning & Preparation

- Removed PII and irrelevant columns (Customer Email, Password, Name, Street, Zipcode, Product Image, Product Description, Order Zipcode)
- Computed derived fields: processing time, delay days, delay flag, profitability flag, and time-based features
- Filtered and classified order profitability into three tiers: Profitable (> 0), Not Profitable (< 0), Break Even (= 0)

---

## Exploratory Data Analysis

**Profitability Distribution:**
- 80.7% of orders are profitable
- 18.7% are loss-making — disproportionately concentrated among delayed shipments
- 0.6% break-even

**Delay Overview:**
- 54.71% of all orders are delivered late (94,523 out of 172,765)
- 31.0% of orders arrive exactly 1 day late — the single largest cohort
- Orders delayed by 1–4 days account for 54.7% of total volume
- Mean profit per order remains stable at ~$20–$23 across all delay levels, meaning the financial problem is driven by **volume**, not individual order economics

**Bottleneck Detection across 6 dimensions:**
- **Shipping Mode** is the #1 lever — First Class: 100% delay rate; Second Class: 79.8%; Standard Class: 39.8%; Same Day: 0%
- **Regional spread is narrow (55–59%)** — confirms a company-wide systemic issue, not a localised failure
- **Customer segments are nearly identical (54.5–55.4%)** — no segment receives preferential service
- **Health & Beauty (56.9%) and Pet Shop (56.6%)** lead by department

**Time-Based Patterns:**
- Peak delay months: August & September (55.4%), December (55.2%) — mid-year promotions and Q4 holiday surge
- Day-of-week variance is minimal (< 1 percentage point spread)
- Intra-day peaks: Hour 21 (57.1%), Hour 11 (56.7%), Hour 12 (56.0%)

---

## Key Insights

| Key Finding | Metric |
|---|---|
| Overall Late Delivery Rate | **54.71%** |
| On-Time Delivery Rate | **45.29%** |
| Total Orders Analyzed | **172,765** |
| Total Profit (profitable orders) | **$7.5M** |
| Profit at Risk (delayed orders) | **$2.1M** |
| Predictive Model Accuracy (RF) | **74%** |

- **First Class shipping has a 100% delay rate** — operating in complete contradiction to its brand promise
- **$2.1M profit is at risk** from delayed orders alone
- **90th percentile delay = 3 days** — the problem is systemic and process-driven, not catastrophic
- **Payment Review orders** show an 80.0% delay rate in Central Africa — a secondary but significant bottleneck
- **Outdoors (61.3%) and Golf (60.8%)** departments in Central Africa show significantly elevated delay rates

---

## Machine Learning Model

A supervised classification model was built to predict whether an individual order will be delivered late (`Late_delivery_risk = 1`).

**Pipeline:**
- Features used: Type, Scheduled Shipment Days, Category, Customer Segment, Shipping Mode, Order Status, Department, Region, order_month, order_day, order_hour
- Frequency encoding applied to all categorical variables
- Stratified 80/20 train/test split
- SMOTE oversampling to address class imbalance (balanced training set: 79,182 per class)

**Random Forest Classifier — Test Set Performance (34,553 records):**

| Metric | Class 0 (On-time) | Class 1 (Late) | Overall (Weighted) |
|---|---|---|---|
| Precision | 0.68 | 0.78 | **0.74** |
| Recall | 0.72 | 0.75 | **0.74** |
| F1-Score | 0.70 | 0.77 | **0.74** |
| Accuracy | — | — | **0.74** |

- **74% overall accuracy** — correctly classifies nearly 3 in 4 orders
- **78% precision on late orders** — when the model flags an order as late, it is correct 78% of the time
- **75% recall on late orders** — captures 75% of all actual late deliveries

---

## Project Structure

```
Supply chain analysis/
│
├── README.md
├── .gitignore
├── Report.docx
│
├── Notebook/                         # Jupyter notebooks
│   └── supply_chain_analysis.ipynb
│
├── Scripts/                          # Python scripts
│   ├── Data_preprocessing.py
│   ├── EDA.py
│   └── Model_evaluation.py
│
├── Images/                           # Charts and visualizations
│   ├── Bottleneck detection.png
│   ├── Delay distribution.png
│   ├── Delay percentage for days,month,week.png
│   ├── Profitability distribution.png
│   └── Root cause analysis.png
│
└── Data/                             # Raw dataset (not tracked in git)
    └── DataCoSupplyChainDataset.csv
```

---

## How to Run This Project

**1. Clone the repository:**

```bash
git clone https://github.com/harshavardhangurav839/supply-chain-delivery-performance-sql-python-ML-SMOTE.git

```

**2. Install required libraries:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```

**3. Place the dataset:**

```
Copy DataCoSupplyChainDataset.csv into the Data/ folder
```

**4. Open and run the notebook:**

```
Notebook/supply_chain_analysis.ipynb
```

Run all cells in order — data loading → EDA → bottleneck detection → root cause analysis → time analysis → ML model

---

## Final Recommendations

| Priority | Recommendation |
|---|---|
| 🔴 **CRITICAL** | Immediately audit First Class & Second Class shipping — 100% and 79.8% delay rates are unacceptable. Suspend or reprice until resolved. |
| 🟠 **HIGH** | Deploy the Random Forest predictive model (74% accuracy) into the order management system to flag high-risk orders at confirmation |
| 🟠 **HIGH** | Resolve payment processing bottlenecks — automate escalation for PAYMENT_REVIEW orders stalled beyond 2 hours |
| 🟡 **MEDIUM** | Build seasonal surge capacity plans for August, October, and December peak delay months |
| 🟡 **MEDIUM** | Default eligible orders to Standard Class shipping (39.8% delay rate vs 100% for First Class) |
| 🟡 **MEDIUM** | Conduct warehouse audit for Outdoors and Golf departments in Central Africa (60–61% delay rates) |
| 🟢 **LOW** | Review pricing and discounts on the 18.7% loss-making order share |
| 🟢 **LOW** | Retrain the model quarterly with carrier-level data, weather events, and warehouse utilization features |

---

## Author & Contact

**Harshavardhan Gurav**

- 📧 Email: harshavardhangurav839@gmail.com
- 💼 LinkedIn: [Harshavardhan Gurav](https://www.linkedin.com/in/harshavardhan-gurav-a2ba0a20a?utm_source=share_via&utm_content=profile&utm_medium=member_android)
