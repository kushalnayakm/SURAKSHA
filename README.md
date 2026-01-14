# 🛡️ SURAKSHA  
### Graph-Based Aadhaar Fraud Detection using R-GCN & Blockchain Audit

> **Fraud is detected as patterns, not isolated mistakes.**

---

## 📌 What is SURAKSHA?

SURAKSHA is an **AI-powered fraud detection system** designed for Aadhaar enrollment data.

Instead of checking records one by one, SURAKSHA **connects records together** and detects **coordinated fraud patterns** using:

- Graphs
- Graph Neural Networks (R-GCN)
- Blockchain-based audit trail
- Interactive frontend dashboard

---

## ❓ How is SURAKSHA Different?

Traditional systems ask:
> *Is this single record valid?*

SURAKSHA asks:
- Who enrolled whom?
- Where was it done?
- When was it done?
- Are many records behaving together?

👉 **Fraud appears as a pattern, not as a single error.**

---

## 🧠 Core Technologies

| Technology | Purpose |
|----------|---------|
| Python | Core implementation |
| NetworkX | Graph construction |
| PyTorch + R-GCN | Fraud pattern learning |
| Blockchain | Tamper-proof audit logs |
| Flask | Frontend dashboard |

---

## 📁 Project Structure

SURAKSHA/
│
├── api_data_aadhar_enrolment/ # Raw Aadhaar CSV files (ignored in Git)
│ └── *.csv
│
├── data/
│ └── processed_real_aadhaar_data.csv
│
├── blockchain/
│ ├── init.py # Blockchain audit logic
│ └── ledger.json # Immutable audit ledger
│
├── models/
│ ├── aadhaar_knowledge_graph.pkl
│ ├── best_suraksha_model.pt
│ ├── graph_info.pkl
│ └── training_metrics.pkl
│
├── outputs/
│ ├── fraud_detection_report.txt
│ ├── fraud_rings.csv
│ └── performance_metrics.json
│
├── load_real_aadhaar_data.py # STEP 1: Data cleaning
├── code2_graph_construction.py # STEP 2: Graph construction
├── code3_rgcn_training.py # STEP 3: Model training
├── code4_fraud_detection.py # STEP 4: Fraud detection
│
├── app.py # Frontend dashboard
├── .gitignore
└── README.md

yaml
Copy code

---

## ⚙️ System Requirements

Python : 3.9 – 3.11
RAM : 8 GB minimum
OS : Windows / Linux / macOS

yaml
Copy code

---

## 📦 Installation (One-Time Setup)

### 1️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
Activate the virtual environment:

bash
Copy code
venv\Scripts\activate

**
########2️⃣ Install Dependencies**

