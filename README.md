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
Activate:

bash
Copy code
venv\Scripts\activate     # Windows
2️⃣ Install Dependencies
bash
Copy code
pip install torch torch-geometric networkx pandas flask numpy
GPU is optional. CPU works fine.
**
🚀 How to Run SURAKSHA (Step-by-Step)**
🟦 STEP 1: Data Cleaning
Input: Raw Aadhaar CSV files
Output: Cleaned dataset

bash
Copy code
python load_real_aadhaar_data.py
##✅ Output:

bash
Copy code
data/processed_real_aadhaar_data.csv
🟦 STEP 2: Knowledge Graph Construction
Converts Aadhaar records into a relationship graph.

bash
Copy code
python code2_graph_construction.py
Graph Nodes:

Person (Enrollment)

Operator

Center

Graph Relations:

enrolled_by

located_at

temporal_proximity

shared_biometric

✅ Outputs:

bash
Copy code
models/aadhaar_knowledge_graph.pkl
models/graph_info.pkl
🟦 STEP 3: R-GCN Model Training
Learns fraud behavior from graph structure.

bash
Copy code
python code3_rgcn_training.py
✅ Results:

Accuracy ≈ 86.4%

Training Time ≈ 3 minutes

Outputs:

bash
Copy code
models/best_suraksha_model.pt
models/training_metrics.pkl
🟦 STEP 4: Fraud Detection (MAIN STEP)
Runs inference on the full graph and detects fraud rings.

bash
Copy code
python code4_fraud_detection.py
Fraud Ring Logic:

If an operator has many suspicious enrollments

Same location / time / biometric

→ Operator is flagged as fraud ring leader

✅ Outputs:

bash
Copy code
outputs/fraud_detection_report.txt
outputs/fraud_rings.csv
outputs/performance_metrics.json
blockchain/ledger.json
🔐 Blockchain Audit Trail
Every action is logged immutably:

Data loading

Graph creation

Model training

Fraud detection

Report generation

👉 If anyone changes past data, hash verification fails.

🖥️ Frontend Dashboard (Demo)
No retraining needed.

bash
Copy code
python app.py
Open in browser:

arduino
Copy code
http://localhost:5000
Dashboard Features:
Fraud rings

Operator confidence scores

Accuracy & detection time

Downloadable reports

