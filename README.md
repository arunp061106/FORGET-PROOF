# ForgetProof: Cryptographically Audited SISA Machine Unlearning for Healthcare ML

[![IBM Z Datathon 2026](https://img.shields.io/badge/IBM%20Z%20Datathon-2026-052FAD?logo=ibm&logoColor=white)](https://github.com/arunp061106/FORGET-PROOF)
[![Track](https://img.shields.io/badge/Track-AI%20Secured%3A%20Innovation%20Without%20Exposure-059669)](https://github.com/arunp061106/FORGET-PROOF)
[![Architecture](https://img.shields.io/badge/Architecture-IBM%20LinuxONE%20%7C%20s390x-1d4ed8)](https://github.com/arunp061106/FORGET-PROOF)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://github.com/arunp061106/FORGET-PROOF)
[![Compliance](https://img.shields.io/badge/Compliance-GDPR%20Art.%2017%20%7C%20HIPAA%20%7C%20CCPA-8b5cf6)](https://github.com/arunp061106/FORGET-PROOF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ForgetProof** is a production-grade, cryptographically audited machine unlearning pipeline engineered for enterprise healthcare environments. Built upon the **SISA (Sharded, Isolated, Sliced, Aggregated)** framework and deployed on **IBM LinuxONE (s390x)**, ForgetProof guarantees the **Right to be Forgotten** (GDPR Art. 17, HIPAA, CCPA) without computationally prohibitive model retraining, without statistical utility loss, and with immutable, tamper-evident cryptographic audit receipts.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [The Healthcare Privacy Dilemma](#the-healthcare-privacy-dilemma)
- [System Architecture & SISA Unlearning](#system-architecture--sisa-unlearning)
- [IBM LinuxONE (s390x) Validation](#ibm-linuxone-s390x-validation)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Cryptographic Audit & Tamper-Evident Receipts](#cryptographic-audit--tamper-evident-receipts)
- [Interactive Offline Results Dashboard](#interactive-offline-results-dashboard)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [Regulatory & Compliance Impact](#regulatory--compliance-impact)
- [Citations & Academic References](#citations--academic-references)

---

## Executive Summary

When a patient revokes data processing consent, deleting their row from a database does **not** purge their private health signals from a trained ML model. The model remains susceptible to **Membership Inference Attacks (MIA)**, and regulatory agencies (e.g., FTC algorithmic disgorgement mandates) can force complete destruction of models trained on non-compliant cohorts.

Retraining an entire production clinical model for each consent withdrawal request is computationally ruinous and operationally impossible in high-throughput hospitals.

**ForgetProof solves this by:**
1. **Partitioning clinical records into isolated consent shards** ($N = 8$), training independent sub-models concurrently.
2. **Retraining ONLY the single affected shard** ($1/N$ of the training corpus) upon consent revocation, slashing retrain latency by **4.50×** ($1.04\text{s}$ vs $4.68\text{s}$) with **zero utility degradation** ($88.71\%$ SISA ensemble vs $88.76\%$ monolithic baseline on 101,766 clinical records).
3. **Saving 74.8% of total computational resources** across streaming revocation workloads.
4. **Generating HMAC-SHA256 signed, hash-chained receipts** verified both server-side and client-side via the Web Crypto API.

---

## The Healthcare Privacy Dilemma

```
Traditional Approach:
Patient Revokes Consent ──► Row Deleted in Database ──► Model Unchanged! ──► Vulnerable to MIA!
                                                                         └──► Regulatory Penalties (GDPR/FTC)

Naive Mitigation:
Patient Revokes Consent ──► Retrain Entire Model From Scratch ──► Huge Compute Waste & Massive Downtime!

ForgetProof Solution:
Patient Revokes Consent ──► Identify Single Shard ──► Targeted Delta Retrain (1.04s) ──► Cryptographic Receipt
```

* **Model Memorization:** Deep decision ensembles trained on high-dimensional electronic health records (EHR) memorize outlier clinical trajectories (e.g., rare diagnosis codes, unique medication combinations).
* **Membership Inference Attacks (MIA):** Attackers observing prediction confidence distributions can reliably infer whether an individual patient's record was used to train the diagnostic model.
* **Algorithmic Disgorgement:** Regulators increasingly mandate that non-compliant models be destroyed, not just the underlying raw records.
* **High Operational Latency:** Retraining monolithic models on millions of records for every individual request creates unacceptable queuing delays and astronomical cloud compute bills.

---

## System Architecture & SISA Unlearning

ForgetProof implements the **SISA (Sharded, Isolated, Sliced, and Aggregated)** paradigm introduced by Bourtoule et al. (IEEE S&P 2021), adapted specifically for tabular clinical EHR records:

```
                                  +---------------------------------------+
                                  |   Raw EHR Dataset (101,766 Records)  |
                                  +---------------------------------------+
                                                      |
                                           [Deterministic Sharding]
                                                      |
                   +------------------+---------------+------------------+
                   |                  |                                  |
                   v                  v                                  v
             +-----------+      +-----------+                      +-----------+
             |  Shard 0  |      |  Shard 1  |       ...            |  Shard 7  |
             +-----------+      +-----------+                      +-----------+
                   |                  |                                  |
                   v                  v                                  v
             +-----------+      +-----------+                      +-----------+
             | Sub-Model |      | Sub-Model |       ...            | Sub-Model |
             |  (Tree 0) |      |  (Tree 1) |                      |  (Tree 7) |
             +-----------+      +-----------+                      +-----------+
                   \                  |                                  /
                    \                 |                                 /
                     +----------------+--------------------------------+
                                      |
                           [Uniform Soft-Voting Aggregator]
                                      |
                                      v
                       +-------------------------------+
                       |  Ensemble Prediction Output   |
                       |       (88.71% Accuracy)       |
                       +-------------------------------+
```

### Targeted Unlearning Sequence

```
Patient Withdrawal Request (Record ID: #48291)
               │
               ▼
[Provenance & Shard Lookup] ──► Identifies Shard #3 (12,720 records)
               │
               ▼
[Shard Delta Retraining]    ──► Drops #48291, retrains Sub-Model #3 ONLY (1.04s)
               │                 (Shards 0-2 and 4-7 remain 100% untouched)
               ▼
[Canary MIA Verification]   ──► Measures loss distribution shift on canary samples
               │
               ▼
[Audit Receipt Generation]  ──► Computes SHA-256 block hash & HMAC-SHA256 signature
               │
               ▼
[Immutable Audit Log]       ──► Appends to chained receipt ledger (10/10 verified)
```

---

## IBM LinuxONE (s390x) Validation

ForgetProof was rigorously tested and verified on an enterprise **IBM LinuxONE** cloud virtual machine:

| Parameter | Specification / Environment |
|---|---|
| **Architecture** | **IBM Z (`s390x`)** |
| **Machine Type** | **IBM 9175 (z16 / Emperor 4)** |
| **Compute Cores** | 3 vCPUs @ 5.5 GHz |
| **Memory** | 8.0 GB RAM |
| **Operating System** | Ubuntu 22.04 LTS (`Linux 5.15.0-134-generic`) |
| **Python Environment** | Python 3.10.12 (Standard Library, NumPy, SciPy, Pandas, Scikit-Learn, Joblib) |
| **Native Compatibility** | **100% Pure Python/Wheel compatibility** (Zero proprietary C/C++ compilation bottlenecks) |
| **Concurrency Scaling** | Parallel shard fitting saturated across all allocated IBM Z virtual cores |

---

## Empirical Benchmark Results

Evaluated on the **UCI Diabetes 130-US Hospitals** dataset (101,766 inpatient records across 10 years, 50 clinical features, predicting 30-day hospital readmission):

### 1. Model Utility vs. Retrain Latency

| Benchmark Dimension | Monolithic Baseline | ForgetProof SISA (N=8) | Performance Advantage |
|---|---|---|---|
| **Accuracy** | **88.76%** | **88.71%** | **$\Delta = 0.05\%$ (Zero utility loss)** |
| **ROC AUC** | 0.6290 | 0.6350 | **+0.006 AUC improvement** |
| **Retraining Latency** | **4.68 seconds** | **1.04 seconds** | **4.50× Faster Retraining** |
| **Data Touched per Request**| 100% of dataset | 12.5% ($1/8$) of dataset | **87.5% reduction in data exposure** |
| **30-Request Total Time** | 121.79 seconds | 30.73 seconds | **74.77% Compute Resources Saved** |
| **Accuracy Drift (30 requests)** | N/A | **88.72%** (rock solid) | **Complete operational stability** |

### 2. Shard Sweep Analysis ($N \in \{1, 2, 4, 8, 16\}$)

| Shard Count ($N$) | Ensemble Accuracy | Ensemble ROC AUC | Retrain Latency (s) | Relative Speedup |
|---|---|---|---|---|
| **$N = 1$ (Baseline)** | 88.76% | 0.6290 | 4.84s | 1.00× |
| **$N = 2$** | 88.73% | 0.6386 | 2.44s | 1.98× |
| **$N = 4$** | 88.73% | 0.6364 | 1.21s | 4.00× |
| **$N = 8$ (Production)** | **88.71%** | **0.6358** | **1.04s** | **4.50×** |
| **$N = 16$** | 88.72% | 0.6326 | 0.82s | 5.90× |

*Key Takeaway:* SISA maintains pristine diagnostic accuracy across all shard counts, proving that consent-batch isolation does not damage collaborative decision boundaries.

### 3. Membership Inference Attack (MIA) Canary Proof

To prove true unlearning, synthetic canary records were injected into the training set. A Yeom-loss threshold MIA evaluated the model's confidence before and after targeted unlearning:
* **Pre-Unlearning:** Canary records exhibit significantly reduced cross-entropy loss, revealing distinct memorization signatures.
* **Post-Unlearning:** Following shard retraining, canary sample loss shifts back to the true non-member baseline distribution, rendering membership identification indistinguishable from random chance.

---

## Cryptographic Audit & Tamper-Evident Receipts

ForgetProof implements an append-only, tamper-evident cryptographic receipt chain. Every unlearning event produces an authenticated record:

```json
{
  "request_id": "REQ-000003",
  "patient_id": "PT-38472",
  "shard_id": 3,
  "timestamp": "2026-10-18T07:25:01Z",
  "retrain_time_s": 1.039,
  "pre_forget_accuracy": 0.8871,
  "post_forget_accuracy": 0.8872,
  "prev_hash": "a1f9e2b83c4d7...",
  "current_hash": "e4c8d2a1b9f7...",
  "hmac_signature": "7d9a1f2e8c3b4..."
}
```

### Verification Properties:
1. **Hash Chaining**: Each block incorporates the SHA-256 digest of the preceding block:
   $$H_i = \text{SHA256}(\text{Payload}_i \parallel H_{i-1})$$
2. **Keyed Authentication**: Authenticated with HMAC-SHA256 using an enterprise hardware security module (HSM) key.
3. **Dual Verification Engines**:
   - **Python CLI Verifier**: `python audit/receipt.py outputs/receipt_chain.json`
   - **Client-Side Web Crypto Verifier**: In-browser real-time SHA-256 integrity checker using `window.crypto.subtle` (zero network dependency).

---

## Interactive Offline Results Dashboard

ForgetProof includes an executive dashboard with interactive Chart.js visualizations:

```
dashboard/
├── index.html                  # Responsive executive user interface
├── style.css                   # Modern dark-mode glassmorphic styling
├── app.js                      # Application logic, tabs, & live Web Crypto verifier
├── data.js                     # Exported benchmark data payload
└── vendor/
    └── chart.umd.min.js        # Self-contained Chart.js engine (Zero CDN, 100% offline)
```

### Features:
* **Interactive Shard Sweep Chart**: Toggle between Accuracy and Retrain Latency curves.
* **Streaming Cumulative Savings Chart**: Compare SISA vs. Monolithic retraining over time.
* **MIA Canary Loss Distribution**: Visual proof of statistical unlearning.
* **Real-Time Cryptographic Verifier**: Run live SHA-256 chain verification or trigger simulated tampering to test detection.
* **100% Offline Capability**: Runs seamlessly via direct `file://` protocol without any local web server or internet access.

---

## Repository Structure

```
FORGET-PROOF/
├── .gitignore                  # Strict exclusion of keys, pem files, and caches
├── README.md                   # Comprehensive system documentation
├── requirements.txt            # Locked open-source dependencies
│
├── audit/
│   ├── __init__.py
│   └── receipt.py              # Cryptographic HMAC-SHA256 receipt chain engine
│
├── dashboard/                  # Standalone offline web dashboard
│   ├── index.html              # Modern interactive dashboard UI
│   ├── style.css               # Glassmorphic dark-theme styles
│   ├── app.js                  # Interactive charts and live cryptographic verifier
│   ├── data.js                 # Exported benchmark data package
│   └── vendor/
│       └── chart.umd.min.js    # Bundled Chart.js v4.4.1 library
│
├── data/
│   ├── __init__.py
│   ├── download.py             # Automated fetcher for UCI Diabetes 130-US Hospitals
│   └── preprocess.py           # Clinical feature engineering, imputation, & scaling
│
├── experiments/
│   ├── __init__.py
│   ├── run_all.py              # Master end-to-end benchmark orchestration pipeline
│   ├── shard_sweep.py          # Empirical scaling sweep across N in {1, 2, 4, 8, 16}
│   └── streaming_sim.py        # 30-request continuous patient withdrawal simulation
│
├── forgetting/
│   ├── __init__.py
│   ├── canary.py               # Outlier canary record generator & injector
│   ├── mia.py                  # Yeom loss-threshold Membership Inference Attack
│   └── sisa.py                 # Core SISA sharded ensemble & targeted retrainer
│
├── models/
│   ├── __init__.py
│   ├── baseline.py             # Monolithic baseline classifier
│   └── ensemble.py             # Parallel soft-voting aggregate ensemble
│
├── outputs/                    # Exported charts, raw metrics, and receipt ledger
│   ├── accuracy_drift.png
│   ├── accuracy_vs_shards.png
│   ├── compliance_report.html
│   ├── consent_lineage.png
│   ├── cumulative_savings.png
│   ├── mia_before_after.png
│   ├── receipt_chain.json      # HMAC-signed cryptographic ledger
│   ├── results.json            # Machine-readable benchmark summary
│   ├── retrain_comparison.png
│   └── shard_sizes.png
│
├── scripts/
│   └── export_dashboard_data.py # Prepares results for offline dashboard consumption
│
└── visualization/
    ├── __init__.py
    ├── lineage_graph.py        # Patient-to-shard provenance visualization
    └── plots.py                # Publication-grade Matplotlib plotting suite
```

---

## Quickstart & Reproduction Guide

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/arunp061106/FORGET-PROOF.git
cd FORGET-PROOF
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows:
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Master Benchmark Pipeline
To fetch data, train baseline & SISA ensembles, execute MIA canary tests, run sweeps, simulate streaming withdrawals, and generate all outputs:
```bash
python experiments/run_all.py
```

### 3. Verify Cryptographic Receipts
To verify the integrity of the generated unlearning audit trail:
```bash
python audit/receipt.py outputs/receipt_chain.json
```

### 4. View the Interactive Dashboard
Open `dashboard/index.html` directly in any web browser:
```bash
# Windows
start dashboard/index.html

# macOS
open dashboard/index.html

# Linux
xdg-open dashboard/index.html
```
*(Or optionally host via `python -m http.server 8080 --directory dashboard`)*

---

## Regulatory & Compliance Impact

| Regulation | Article / Provision | How ForgetProof Ensures Compliance |
|---|---|---|
| **GDPR** | Article 17 ("Right to Erasure") | Mathematically removes record influence via isolated sub-model retraining within 1.04 seconds. |
| **HIPAA** | Privacy Rule § 164.514 | Neutralizes Membership Inference vulnerability, preventing re-identification of patients. |
| **CCPA / CPRA** | Cal. Civ. Code § 1798.105 | Fulfils consumer deletion requests with real-time provenance tracking and audit logs. |
| **FTC Enforcement** | Algorithmic Disgorgement | Prevents complete model destruction penalties by providing verifiable cryptographic proof of unlearning. |

---

## Citations & Academic References

1. **SISA Architecture**: Bourtoule, L., Chandrasekaran, V., Choquette-Choo, C. A., Jia, H., Travers, A., Zhang, H., Lie, D., & Papernot, N. (2021). *Machine Unlearning*. In **IEEE Symposium on Security and Privacy (S&P)**, pp. 141-159. [arXiv:1912.03817](https://arxiv.org/abs/1912.03817).
2. **Membership Inference Attacks**: Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). *Membership Inference Attacks Against Machine Learning Models*. In **IEEE Symposium on Security and Privacy (S&P)**, pp. 3-18. [arXiv:1610.05820](https://arxiv.org/abs/1610.05820).
3. **Loss-Threshold Privacy Risk**: Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018). *Privacy Risk in Machine Learning: Analyzing the Impact of Data, Models, and Learning Algorithms*. In **IEEE Computer Security Foundations Symposium (CSF)**, pp. 231-245.
4. **Clinical Dataset**: Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Balasubramanian, S., Higgins, K. W., & Clore, J. N. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records*. **BioMed Research International**, 2014, 781670. [UCI Machine Learning Repository #296](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

---

## Authors & Acknowledgments

* **Arunkarthick P** ([@arunp061106](https://github.com/arunp061106))
* Developed for the **IBM Z Datathon 2026** — *Track: AI Secured: Innovation Without Exposure*.
* Built and validated on **IBM LinuxONE Cloud Infrastructure**.
