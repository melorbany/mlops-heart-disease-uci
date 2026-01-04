```markdown
<!-- File: reqs.md (token-optimized) -->

# MLOps Assign-1 — Spec (Heart Disease UCI)

## Goal
Binary classifier for heart disease risk + production-ready, reproducible, monitored API.

## Data
- Source: UCI Heart Disease (CSV)
- Target: binary (disease yes/no)
- Features: 14+ (age, sex, bp, chol, etc.)
- Provide: download script OR clear instructions

## Must-Do Tasks / Outputs
### 1) EDA + preprocessing
- Handle missing values
- Encode categoricals
- Plots: histograms + corr heatmap + class balance
- Save plots as artifacts

### 2) Features + models
- Final preprocessing pipeline: scaling + encoding (train/infer parity)
- Train ≥2 models (e.g., LogisticRegression, RandomForest)
- Tuning documented (minimal notes ok)
- Evaluate with CV + metrics: accuracy, precision, recall, ROC-AUC

### 3) Experiment tracking
- MLflow (or similar)
- Log for each run: params, metrics, artifacts, plots

### 4) Packaging + reproducibility
- Save final model (MLflow/pickle/ONNX)
- Provide `requirements.txt` or `env.yml`
- Inference must reuse same preprocessing pipeline

### 5) CI/CD + tests
- Unit tests (pytest/unittest) for data + model code
- CI (GitHub Actions or Jenkins): lint + tests + train
- Workflow artifacts/logs saved
- Pipeline fails on lint/test errors (clear logs)

### 6) Containerized serving
- Dockerized API (FastAPI/Flask)
- Endpoint: `POST /predict`
  - Input: JSON features
  - Output: `{prediction: 0|1, confidence: 0..1}`
- Must build + run locally
- Provide sample payload + curl command

### 7) Deployment (prod-style)
- Deploy Docker API to cloud OR Kubernetes (GKE/EKS/AKS/Minikube/Docker Desktop)
- Use k8s manifests OR Helm
- Expose via LoadBalancer OR Ingress
- Verify endpoint + include screenshots

### 8) Monitoring + logging
- Log API requests
- Simple monitoring: Prometheus+Grafana OR metrics/log dashboard
- Provide proof (screenshots/notes)

### 9) Documentation
Report (MD/PDF) includes:
- setup
- EDA/model choices
- MLflow summary
- architecture diagram
- CI/CD + deploy screenshots
- repo link
Also submit: ~10-page report in doc/docx.

## Deliverables (repo)
- Code (EDA/train/infer/API)
- `Dockerfile`
- `requirements.txt`/`env.yml`
- dataset download script/instructions (+ cleaned data if included)
- `tests/`
- CI workflow YAML/Jenkinsfile
- `k8s/` or `helm/`
- `screenshots/`
- report file(s)

## Extra deliverables
- Short end-to-end pipeline video
- Deployed API URL OR local access instructions

## Acceptance (binary)
- Fresh install works from deps file
- Docker build + local `/predict` works
- CI runs lint+tests+train; fails on errors
- MLflow logs runs
- k8s/cloud deployment reachable
- logging + monitoring demonstrated
```