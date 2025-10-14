# 🧠 OS Scheduler Simulator with ML & Cloud Integration — MVP PRD

## Overview
A minimal, end-to-end project that combines **Python**, **C++**, **Operating Systems concepts**, **Machine Learning**, and **Cloud technologies** into one cohesive system.

The system simulates OS-level **process scheduling**, evaluates multiple scheduling algorithms, and uses a **Machine Learning model** to predict the best scheduler for a given workload.

The goal: an educational and engineering playground that touches every layer of the stack without becoming unmanageably large.

---

## 🎯 Objectives
- Implement a **cross-language OS simulator** (Python + C++).
- Demonstrate **scheduling algorithms** (FCFS, SJF, RR, Priority).
- Use **ML** to predict which scheduler minimizes turnaround time.
- Store experiment results in **MongoDB**.
- Provide a **FastAPI web backend** and a **simple web dashboard (Next.js)**.
- Containerize with **Docker**, and optionally deploy to **AWS (ECS/EKS)**.
- Integrate with **GitHub + CI/CD** for testing and versioning.

---

## 🧩 Component Breakdown

| Component | Technology | Purpose |
|------------|-------------|----------|
| **Core Simulator** | C++ (Pybind11) | Fast simulation of CPU scheduling algorithms |
| **Python Backend** | FastAPI / Typer | Exposes API endpoints, handles logic |
| **Database** | MongoDB | Stores simulation history, metrics, and ML predictions |
| **ML Model** | scikit-learn | Predicts optimal scheduling algorithm |
| **Frontend** | Next.js or HTML/TypeScript | Visualizes workloads, results, and recommendations |
| **Testing** | Pytest + CTest | Ensures correctness of simulation & API |
| **Containerization** | Docker + Docker Compose | Local orchestration |
| **Cloud Deployment** | AWS ECS or EKS | Optional hosting for app |
| **Version Control** | Git + GitHub | Source control and CI/CD |

---

## ⚙️ MVP Feature List

### 1. Scheduling Simulation
- Implement core scheduling algorithms in C++:
  - FCFS
  - SJF (non-preemptive)
  - Round Robin
  - Priority Scheduling
- Expose results through Python via Pybind11.

### 2. Metrics
- Compute:
  - Average Wait Time
  - Average Turnaround Time
  - Average Response Time
- Generate a Gantt chart visualization (matplotlib or frontend).

### 3. Machine Learning Model
- Offline training script to generate workloads and label best algorithms.
- Train a Decision Tree / Random Forest on workload features.
- Store and load model (`models/scheduler_model.pkl`).
- Expose `/predict` endpoint in FastAPI.

### 4. MongoDB Integration
- Store simulation runs:
  ```json
  {
    "workload": [{ "pid": 1, "arrival": 0, "burst": 4 }, ...],
    "algorithm": "SJF",
    "results": { "avg_wait": 2.1, "avg_turnaround": 4.6 },
    "model_prediction": { "algo": "SJF", "confidence": 0.82 },
    "created_at": "2025-01-01T00:00:00Z"
  }
  ```

### 5. Frontend (MVP)
- Form to input workload (PID, arrival, burst, etc.).
- Buttons for:
  - “Run Simulation”
  - “Predict Best Algorithm”
- Simple chart of the timeline and metrics.

### 6. Containerization
- Dockerfile for backend.
- Dockerfile for Mongo.
- `docker-compose.yml` to run both locally.

### 7. Testing
- Unit tests (Pytest for backend, Catch2 for C++).
- Integration test: simulate workload → store → retrieve from DB.
- System tests

---

## 🧠 Architecture Diagram
```
┌────────────────────────────────────┐
│          Web UI (Next.js)          │
│   Input workload + view results    │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│       FastAPI Backend (Python)     │
│ - Routes /simulate, /predict       │
│ - Calls C++ scheduler via pybind11 │
│ - Logs results to MongoDB          │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│   C++ Scheduler Core (Pybind11)    │
│  - FCFS / SJF / RR / Priority      │
│  - Returns timeline + metrics      │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│  ML Model (scikit-learn)           │
│  - Predict best algorithm           │
│  - Retrain periodically             │
└────────────────────────────────────┘
                │
                ▼
┌────────────────────────────────────┐
│ MongoDB                            │
│ Stores workloads, predictions, etc │
└────────────────────────────────────┘
```

---

## 🚀 Development Roadmap (Milestones)

| Week | Milestone | Deliverable |
|------|------------|-------------|
| **1** | Core Setup | Repo, CMake, scikit-build-core, pybind11 bindings, FCFS algorithm |
| **2** | Add Algorithms + Tests | Add SJF, RR, Priority; Pytest suite; CLI working |
| **3** | API + Database | FastAPI + MongoDB + Docker Compose |
| **4** | Frontend MVP | Web dashboard for simulation & metrics |
| **5** | ML Integration | Train & serve prediction model via `/predict` |
| **6** | Cloud + CI/CD | Docker build pipeline, AWS deployment, GitHub Actions |

---

## ✅ Testing Strategy
- **Unit tests:**  
  Each algorithm validated against expected metrics.
- **Integration tests:**  
  API routes call simulator and verify DB writes.
- **ML evaluation:**  
  Accuracy > 70% on validation workloads.
- **Regression tests:**  
  Benchmark scheduler performance on fixed workloads.

---

## 🧰 Tech Stack Summary
| Layer | Tech |
|--------|------|
| Core | C++17, Pybind11, CMake, scikit-build-core |
| Backend | Python 3.11+, FastAPI, Typer, Pydantic |
| ML | scikit-learn, joblib, numpy, pandas |
| Database | MongoDB |
| Frontend | Next.js + TypeScript (or HTML/CSS) |
| Infra | Docker, Docker Compose, AWS ECS/EKS |
| CI/CD | GitHub Actions, pytest |

---

## 🔒 Future Expansion (Post-MVP)
- Add **preemptive schedulers** (SRTF, MLFQ).
- Add **memory & I/O simulation modules**.
- Support **user authentication** for personalized dashboards.
- Integrate **Prometheus/Grafana** for metrics monitoring.
- Deploy full stack with **Terraform + AWS RDS + ECS**.
