# ForgetProof: Cryptographically Audited SISA Machine Unlearning for Healthcare ML

[![IBM Z Datathon 2026](https://img.shields.io/badge/IBM%20Z%20Datathon-2026-052FAD?logo=ibm&logoColor=white)](https://github.com/arunp061106/FORGET-PROOF)
[![Track](https://img.shields.io/badge/Track-AI%20Secured%3A%20Innovation%20Without%20Exposure-059669)](https://github.com/arunp061106/FORGET-PROOF)
[![Architecture](https://img.shields.io/badge/Architecture-IBM%20LinuxONE%20%7C%20s390x-1d4ed8)](https://github.com/arunp061106/FORGET-PROOF)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://github.com/arunp061106/FORGET-PROOF)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **ForgetProof** is a modular research pipeline for audited machine unlearning in healthcare environments. Built upon the **SISA (Sharded, Isolated, Sliced, Aggregated)** framework and evaluated on **IBM LinuxONE (s390x)**, ForgetProof supports the **Right to be Forgotten** (GDPR Art. 17, HIPAA, CCPA) without computationally prohibitive model retraining, demonstrating comparable predictive performance (ROC AUC 0.635 vs 0.629; accuracy 88.71% vs 88.76%; the majority-class rate is 88.74%, so accuracy alone is not informative), and generating tamper-evident cryptographic audit receipts.

---

## Table of Contents
- [Executive Summary](#executive-summary)
- [The Healthcare Privacy Dilemma](#the-healthcare-privacy-dilemma)
- [System Architecture & SISA Unlearning](#system-architecture--sisa-unlearning)
- [IBM LinuxONE (s390x) Validation](#ibm-linuxone-s390x-validation)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [Cryptographic Audit & Tamper-Evident Receipts](#cryptographic-audit--tamper-evident-receipts)
- [Limitations & Open Challenges](#limitations--open-challenges)
- [Interactive Offline Results Dashboard](#interactive-offline-results-dashboard)
- [Repository Structure](#repository-structure)
- [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)
- [Regulatory & Compliance Impact](#regulatory--compliance-impact)
- [Citations & Academic References](#citations--academic-references)

---

## Executive Summary

When a patient revokes data processing consent, deleting their row from a database does **not** purge their private health signals from a trained ML model. The model remains susceptible to **Membership Inference Attacks (MIA)**, and regulatory agencies (e.g., FTC algorithmic disgorgement mandates) can force destruction of models trained on non-compliant cohorts.

Retraining an entire clinical model from scratch for each consent withdrawal request is computationally prohibitive and operationally difficult in continuous clinical workflows.

**ForgetProof addresses this by:**
1. **Partitioning clinical records into isolated consent shards** ($N = 8$), training independent sub-models concurrently.
2. **Retraining ONLY the single affected shard** ($1/N$ of the training corpus) upon consent revocation, reducing retrain latency by **4.50×** ($1.04\text{s}$ measured single-shard retrain vs $4.68\text{s}$ full retrain baseline) while maintaining comparable utility (ROC AUC 0.635 vs 0.629; accuracy 88.71% vs 88.76%; the majority-class rate is 88.74%, so accuracy alone is not informative).
3. **Saving 74.8% of computational resources** across a simulated 30-request streaming revocation workload ($30.73\text{s}$ total SISA retrain vs $121.79\text{s}$ full retrain).
4. **Generating HMAC-SHA256 signed, hash-chained receipts** verified both server-side and client-side via the Web Crypto API.

---

## The Healthcare Privacy Dilemma

```
Traditional Approach:
Patient Revokes Consent ──► Row Deleted in Database ──► Model Unchanged! ──► Vulnerable to MIA!
                                                                         └──► Regulatory Scrutiny

Naive Mitigation:
Patient Revokes Consent ──► Retrain Entire Model From Scratch ──► High Compute Cost & Downtime

ForgetProof Approach:
Patient Revokes Consent ──► Identify Single Shard ──► Targeted Delta Retrain (1.04s) ──► Cryptographic Receipt
```

* **Model Memorization:** Decision tree ensembles trained on high-dimensional electronic health records (EHR) can memorize outlier clinical trajectories (e.g., rare diagnosis codes, unique medication combinations).
* **Membership Inference Attacks (MIA):** Attackers observing prediction confidence distributions can attempt to infer whether an individual patient's record was used to train the diagnostic model.
* **Algorithmic Disgorgement:** Regulators increasingly mandate that non-compliant models be discarded when consent is revoked, not just the underlying raw records.
* **High Operational Latency:** Retraining monolithic models on entire datasets for every individual request creates queuing delays and high computational overhead.

---

## System Architecture & SISA Unlearning

ForgetProof implements the **SISA (Sharded, Isolated, Sliced, and Aggregated)** paradigm introduced by Bourtoule et al. (IEEE S&P 2021), adapted for tabular clinical EHR records:

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
Patient Withdrawal Request (Record ID: #102765)
               │
               ▼
[Provenance & Shard Lookup] ──► Identifies Shard #3 (~9.8k records)
               │
               ▼
[Shard Delta Retraining]    ──► Drops #102765, retrains Sub-Model #3 ONLY (1.04s measured)
               │                 (Shards 0-2 and 4-7 remain untouched)
               ▼
[Canary MIA Verification]   ──► Measures loss distribution shift on canary samples
               │
               ▼
[Audit Receipt Generation]  ──► Computes SHA-256 block hash & HMAC-SHA256 signature
               │
               ▼
[Tamper-Evident Audit Log]  ──► Appends to chained receipt ledger (10/10 verified)
```

---

## IBM LinuxONE (s390x) Validation

ForgetProof was benchmarked on an enterprise **IBM LinuxONE** cloud virtual machine. The values below are directly verified from instance command outputs (`uname -a`, `free -h`, `lscpu`):

| Parameter | Instance Value |
|---|---|
| **Architecture** | `s390x` (IBM/S390, 64-bit, Big Endian) |
| **Machine Type** | `9175` |
| **Virtualization / Hypervisor** | `z/VM 7.3.0` (Full virtualization) |
| **Compute Cores** | 3 vCPUs @ 5508 MHz (dynamic & static) |
| **Memory** | 7.8 GiB total (1.2 GiB used, 6.5 GiB available, 0B swap) |
| **Operating System / Kernel** | `Linux forgetproof 5.15.0-56-generic #62-Ubuntu SMP` (Ubuntu 22.04 LTS) |
| **Python Environment** | Python 3.10.12 (Standard Library, NumPy, SciPy, Pandas, Scikit-Learn, Joblib) |
| **Wheel Compatibility** | Pure Python & standard prebuilt wheels (no unsupported native C extensions) |
| **Parallelization** | Concurrent shard fitting across allocated IBM Z virtual cores via `joblib` |

---

## Empirical Benchmark Results

Evaluated on the **UCI Diabetes 130-US Hospitals** dataset (101,766 inpatient records across 10 years, 50 clinical features, predicting 30-day hospital readmission):

### 1. Model Utility vs. Retrain Latency

| Benchmark Dimension | Monolithic Baseline | ForgetProof SISA (N=8) | Performance Context |
|---|---|---|---|
| **Accuracy** | 88.76% | 88.71% | ROC AUC 0.635 vs 0.629 (comparable); accuracy 88.71% vs 88.76%; the majority-class rate is 88.74%, so accuracy alone is not informative. |
| **ROC AUC** | 0.6290 | 0.6350 | ROC AUC 0.635 vs 0.629 (comparable); accuracy 88.71% vs 88.76%; the majority-class rate is 88.74%, so accuracy alone is not informative. |
| **Retraining Latency** | **4.68 seconds** | **1.04 seconds** | **4.50× Faster Retraining** (measured wall-clock time) |
| **Data Touched per Request**| 100% of dataset | 12.5% ($1/8$) of dataset | 87.5% reduction in data processed per withdrawal |
| **30-Request Simulation Time** | 121.79 seconds | 30.73 seconds | **74.77% Compute Resources Saved** |
| **Post-Unlearning Accuracy** | N/A | **88.72%** | Stable accuracy across 30 sequential revocations |

### 2. Shard Sweep Analysis ($N \in \{1, 2, 4, 8, 16\}$)

| Shard Count ($N$) | Ensemble Accuracy | Ensemble ROC AUC | Retrain Latency (s) | Relative Speedup |
|---|---|---|---|---|
| **$N = 1$ (Baseline)** | 88.76% | 0.6290 | 4.68s (measured full retrain) | 1.00× |
| **$N = 2$** | 88.73% | 0.6386 | ~2.34s [estimated (full / 2)] | ~2.0× [estimated] |
| **$N = 4$** | 88.73% | 0.6364 | ~1.17s [estimated (full / 4)] | ~4.0× [estimated] |
| **$N = 8$ (Production)** | **88.71%** | **0.6358** | **1.04s (measured)** | **4.50× (measured)** |
| **$N = 16$** | 88.72% | 0.6326 | ~0.29s [estimated (full / 16)] | ~16.0× [estimated] |

> *Note on Shard Sweep Latency:* Single-shard retrain latency was directly measured for N=8 at 1.04s (4.50× speedup vs 4.68s baseline). Per-shard retrain times for N=2, 4, and 16 were not separately measured in the sweep and are estimated as full training time / N.

### 3. Membership Inference Attack (MIA) Canary Evaluation

On a canary test, the removed canaries' loss moved toward the non-member distribution after shard retraining. This is one small test and not proof of forgetting. Verifying forgetting is an open research problem.

---

## Cryptographic Audit & Tamper-Evident Receipts

ForgetProof implements an append-only, tamper-evident cryptographic receipt chain. Every unlearning event produces an authenticated record:

```json
{
  "request_id": "f0802ee2-d3c6-4502-b150-c7678c883f38",
  "record_id": 102765,
  "shard_id": 0,
  "timestamp": "2026-10-18T07:21:19.794242+00:00",
  "retrain_time_seconds": 1.0272,
  "accuracy_before": 0.8871,
  "accuracy_after": 0.8872,
  "prev_hash": "GENESIS",
  "hash": "2518e63775b0cf5defe931f8d138177e2d6990edbb679f8a191ef3d955ce4c37",
  "signature": "78826336ec8b7e049fe73a8f2ad36514da7c6b31701c69bbbf90a97ea98da55b"
}
```
*(Illustrative receipt schema, real record ID #102765 from `outputs/receipt_chain.json`)*

### Verification Properties:
1. **Hash Chaining**: Each block incorporates the SHA-256 digest of the preceding block:
   $$H_i = \text{SHA256}(\text{Payload}_i \parallel H_{i-1})$$
2. **Keyed Authentication**: Authenticated with HMAC-SHA256 using a configurable secret key (hardware security module / HSM integration planned).
3. **Dual Verification Engines**:
   - **Python CLI Verifier**: `python audit/receipt.py outputs/receipt_chain.json`
   - **Client-Side Web Crypto Verifier**: In-browser real-time SHA-256 integrity checker using `window.crypto.subtle` (zero network dependency).

---

## Limitations & Open Challenges

1. **Simulated Consent Events**: The 30 withdrawal requests were simulated for evaluation; operational hospital consent streams would experience variable arrival distributions and burst dynamics.
2. **Small Canary Test**: Privacy evaluation was conducted on a single small canary test. Verifying unlearning across arbitrary data distributions remains an active, open research problem.
3. **Modest Predictive Power**: The underlying classification task achieves ROC AUC 0.635 vs baseline 0.629. Because the majority-class rate is 88.74%, raw accuracy alone is uninformative.
4. **Estimated Sweep Latencies**: Single-shard retrain latency was directly measured only for N=8 (1.04s). Retrain times for N=2, 4, and 16 are estimated as full time / N.
5. **Hardware Security Extensibility**: IBM Secure Execution for Linux and Hardware Security Module (HSM) signing are planned architectural enhancements, not run in this benchmark.

---

## Interactive Offline Results Dashboard

ForgetProof includes an executive dashboard with interactive Chart.js visualizations:

```
dashboard/
├── index.html                  # Responsive user interface
├── style.css                   # Modern dark-mode glassmorphic styling
├── app.js                      # Application logic, tabs, & live Web Crypto verifier
├── data.js                     # Exported benchmark data payload
└── vendor/
    └── chart.umd.min.js        # Self-contained Chart.js engine (Zero CDN, 100% offline)
```

### Features:
* **Interactive Shard Sweep Chart**: Displays measured N=8 bar alongside estimated shard configurations.
* **Streaming Cumulative Savings Chart**: Compare SISA vs. Monolithic retraining over time.
* **MIA Canary Loss Distribution**: Overlapping loss histograms with contextualized test set imbalance notes.
* **Real-Time Cryptographic Verifier**: Run live SHA-256 chain verification or trigger simulated tampering to test detection.
* **100% Offline Capability**: Runs seamlessly via direct `file://` protocol without any local web server or internet access.

---

## Repository Structure

```
FORGET-PROOF/
├── .gitignore                  # Strict exclusion of keys, pem files, and caches
├── README.md                   # System documentation
├── requirements.txt            # Dependencies
│
├── audit/
│   ├── __init__.py
│   ├── compliance.py           # Internal compliance scoring metric
│   └── receipt.py              # Cryptographic HMAC-SHA256 receipt chain engine
│
├── dashboard/                  # Standalone offline web dashboard
│   ├── index.html              # Interactive dashboard UI
│   ├── style.css               # Styling
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
│   ├── run_all.py              # Master benchmark orchestration pipeline
│   ├── shard_sweep.py          # Scaling sweep across N in {1, 2, 4, 8, 16}
│   └── streaming_sim.py        # 30-request patient withdrawal simulation
│
├── forgetting/
│   ├── __init__.py
│   ├── canary.py               # Outlier canary record generator & injector
│   ├── forget_engine.py        # Core SISA sharded ensemble & targeted retrainer
│   └── membership_inference.py # Yeom loss-threshold Membership Inference Attack
│
├── models/
│   ├── __init__.py
│   ├── baseline.py             # Monolithic baseline classifier
│   ├── ensemble.py             # Parallel soft-voting aggregate ensemble
│   └── shard_trainer.py        # Shard training utilities
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
    └── plots.py                # Matplotlib plotting suite
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
To fetch data, train baseline & SISA ensembles, evaluate canary loss shifts, run sweeps, simulate streaming withdrawals, and generate all outputs:
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

> **Disclaimer:** ForgetProof provides technical unlearning mechanisms and cryptographic audit logging to support compliance workflows; it **does not constitute legal compliance** or formal certification under GDPR, HIPAA, or CCPA.

| Regulation | Article / Provision | How ForgetProof Supports Workflows |
|---|---|---|
| **GDPR** | Article 17 ("Right to Erasure") | Supports removal of patient records via isolated sub-model retraining within 1.04 seconds (measured for N=8). |
| **HIPAA** | Privacy Rule § 164.514 | Helps evidence that patient records removed from training shards move toward the non-member distribution. |
| **CCPA / CPRA** | Cal. Civ. Code § 1798.105 | Helps evidence consumer deletion fulfillment with provenance tracking and audit logs. |
| **FTC Enforcement** | Algorithmic Disgorgement | Helps evidence unlearning compliance to mitigate model destruction risks through cryptographic audit logging. |

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
* Benchmarked on **IBM LinuxONE Cloud Infrastructure**.
