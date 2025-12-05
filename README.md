# 🌌 TransientsHostRanking
A complete end-to-end pipeline for identifying and ranking host galaxies of astronomical transients (e.g., supernovae).  
This project combines **NOIRLab Data Lab queries**, **DLR-based host assignment**, and a **machine learning model** to produce high-quality host galaxy rankings.

---

## 🚀 Project Overview

The goal of this project is to:
1. Query host galaxy candidates around transient positions using **Legacy Survey DR10** via NOIRLab Data Lab.
2. Compute **Directional Light Radius (DLR)** and select host candidates based on a CF-test.
3. Build a training set from historical ZTF classifications.
4. Train a machine learning classifier to **rank host galaxy candidates**.
5. Apply the trained model to new survey data to produce final host rankings.

This pipeline is used in the context of a cosmology project involving  
**80,000–90,000 supernovae** and the identification of their host galaxies.

---

## 🧱 Repository Structure
TransientsHostRanking/
│
├── data/
│ ├── input/ # Input data (SN lists, DR4 host candidates)
│ ├── output/ # Model predictions and ranked hosts
│ ├── training_data/ # Training CSV files (candidates + ground truth)
│ ├── save/ # Simulation results or helper tables
│
├── models/
│ ├── hgb_model.pkl # Trained HistGradientBoostingClassifier
│ ├── feature_list.pkl # Matching feature list for inference
│
├── src/
│ ├── main.py # LS DR10 query, DLR, CF-test, host candidate creation
│ ├── train_model.py # Training of the ML model
│ ├── apply_model.py # Apply ML to DR4 candidates & create rankings
│
├── README.md # Project documentation
└── requirements.txt # Dependencies


---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/PouGitGudHub/TransientsHostRanking.git
cd TransientsHostRanking
```
---
Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```
---
Install dependencies:
```bash
pip install -r requirements.txt
```

---
##🔐 Data Lab Authentication

The project uses NOIRLab’s Data Lab API.

You can set your credentials in main.py or export them as environment variables:
```bash
export DATALAB_USER="your_username"
export DATALAB_PASS="your_password"

```
---
##🛰 1. Query Legacy Survey DR10 + Compute DLR (main.py)

This script:

- loads supernova coordinates

- queries LS DR10 using Data Lab

- extracts fluxes, morphology, and photometric redshift

- computes DLR following LS/DESI conventions

- performs the CF-Test to find the 98% efficiency DLR cut

outputs all host candidates to:

```bash
/data/input/CF_limit_hosts_combined_dr4.csv
```

Run:
```bash

