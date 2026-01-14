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

## 📂 Project Structure

```text
SURAKSHA/
│
├── api_data_aadhar_enrolment/        # Raw Aadhaar CSV files (ignored in Git)
│
├── data/
│   └── processed_real_aadhaar_data.csv
│
├── blockchain/
│   ├── __init__.py                  # Blockchain audit logic
│   └── ledger.json                  # Immutable audit ledger
│
├── models/
│   ├── aadhaar_knowledge_graph.pkl
│   ├── best_suraksha_model.pt
│   ├── graph_info.pkl
│   └── training_metrics.pkl
│
├── outputs/
│   ├── fraud_detection_report.txt
│   ├── fraud_rings.csv
│   └── performance_metrics.json
│
├── load_real_aadhaar_data.py         # STEP 1: Data cleaning
├── code2_graph_construction.py       # STEP 2: Knowledge graph
├── code3_rgcn_training.py            # STEP 3: Model training
├── code4_fraud_detection.py          # STEP 4: Fraud detection
├── app.py                            # Frontend dashboard
│
├── .gitignore
└── README.md


---

## ⚙️ System Requirements

Python : 3.9 – 3.11
RAM : 8 GB minimum
OS : Windows / Linux / macOS

yaml
Copy code

---

📦 Installation (One-Time Setup)
1️⃣ Create Virtual Environment (Recommended)
python -m venv venv


Activate the virtual environment (Windows):

venv\Scripts\activate

2️⃣ Install Dependencies
pip install torch torch-geometric networkx pandas flask numpy


💡 GPU is optional. CPU works perfectly fine.

🚀 How to Run SURAKSHA (Step-by-Step)
🟦 STEP 1: Data Cleaning

Purpose:
Clean raw Aadhaar enrollment records.

Command:

python load_real_aadhaar_data.py


Output:

data/processed_real_aadhaar_data.csv

🟦 STEP 2: Knowledge Graph Construction

Purpose:
Convert Aadhaar records into a relationship graph.

Command:

python code2_graph_construction.py


Graph Nodes:

Person (Enrollment)

Operator

Enrollment Center

Location

Graph Relationships:

enrolled_by

located_at

temporal_proximity

shared_biometric

Outputs:

models/aadhaar_knowledge_graph.pkl
models/graph_info.pkl

🟦 STEP 3: R-GCN Model Training

Purpose:
Train a Relational Graph Convolutional Network to learn fraud patterns.

Command:

python code3_rgcn_training.py


Training Results:

Accuracy ≈ 86.4%

Training time ≈ 3 minutes

Outputs:

models/best_suraksha_model.pt
models/training_metrics.pkl

🟦 STEP 4: Fraud Detection (MAIN STEP)

Purpose:
Run inference on the full graph and detect fraud rings.

Command:

python code4_fraud_detection.py


Fraud Ring Logic:

Many suspicious enrollments

Same operator

Same time / location / biometric

⇒ Operator flagged as fraud ring leader

Outputs:

outputs/fraud_detection_report.txt
outputs/fraud_rings.csv
outputs/performance_metrics.json
blockchain/ledger.json

🔐 Blockchain Audit Trail

Every critical step is immutably logged:

Data loading

Graph creation

Model training

Fraud detection

Report generation

📌 If anyone tries to modify past records,
hash verification fails immediately.

🖥️ Frontend Dashboard (Demo)

No retraining needed.

Run Dashboard
python app.py

Open in Browser
http://localhost:5000

Dashboard Shows

Fraud rings

Operator confidence scores

Accuracy & detection time

Downloadable reports

🧠 One-Line Summary (For Judges)

“SURAKSHA transforms Aadhaar fraud detection from record-level checks to network-level intelligence using graph neural networks and blockchain auditing.”




