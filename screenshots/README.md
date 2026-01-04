# Screenshots Guide

This directory should contain visual proof of your MLOps pipeline deployment, monitoring, and functionality.

## Required Screenshots

### 1. CI/CD Pipeline (`ci_cd_pipeline.png`)

**What to capture:** GitHub Actions workflow successful run

**Steps:**
1. Go to your GitHub repository
2. Click on "Actions" tab
3. Click on a successful workflow run (green checkmark)
4. Take a screenshot showing:
   - All stages completed successfully (green checkmarks)
   - Workflow name and duration
   - Branch name (master)

**Example layout:**
```
✓ Lint (1m 32s)
✓ Security (2m 15s)
✓ Data (1m 5s)
✓ Train Model (3m 12s)
✓ MLflow Report (0m 45s)
✓ Test (2m 8s)
✓ Validate Model (0m 12s)
✓ Docker Build (2m 34s)
✓ Docker Push (1m 23s)
✓ Cloud Deploy (0m 48s)
✓ Health Check (0m 15s)
✓ Predict API Test (0m 32s)
```

---

### 2. MLflow UI (`mlflow_ui.png`)

**What to capture:** MLflow experiments dashboard with multiple runs

**Steps:**
1. Start MLflow UI:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db
   ```
2. Open http://localhost:5000 in browser
3. Navigate to "heart-disease-classification" experiment
4. Take a screenshot showing:
   - List of runs (logreg, rf, final_model)
   - Metrics columns (accuracy, roc_auc, etc.)
   - Parameters columns
   - Start time and duration

**What to highlight:**
- Multiple model runs visible
- Sortable metrics table
- Experiment name clearly visible

---

### 3. FastAPI Swagger UI (`api_swagger.png`)

**What to capture:** Interactive API documentation

**Steps:**
1. Start API locally or access deployed version:
   ```bash
   python run_app.py
   # or visit deployed URL
   ```
2. Open http://localhost:8000/ in browser (Swagger UI at root)
3. Take a screenshot showing:
   - API title: "Heart Disease Prediction API"
   - All endpoints listed:
     - GET /health
     - POST /predict
     - GET /metrics
   - OpenAPI schema visible

**What to highlight:**
- Clean API interface
- All endpoints documented
- Schemas section (HeartFeatures, PredictionResponse)

---

### 4. API Prediction Test (`api_predict_test.png`)

**What to capture:** Successful prediction request and response

**Option A - Using Swagger UI:**
1. Open Swagger UI
2. Click "POST /predict"
3. Click "Try it out"
4. Paste sample data:
   ```json
   {
     "age": 63,
     "sex": 1,
     "cp": 3,
     "trestbps": 145,
     "chol": 233,
     "fbs": 1,
     "restecg": 0,
     "thalach": 150,
     "exang": 0,
     "oldpeak": 2.3,
     "slope": 0,
     "ca": 0,
     "thal": 1
   }
   ```
5. Click "Execute"
6. Take a screenshot showing:
   - Request body
   - Response code 200
   - Response body with prediction and probability

**Option B - Using curl + terminal:**
1. Run curl command:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1}'
   ```
2. Take a screenshot of terminal showing:
   - The curl command
   - JSON response with prediction and probability

---

### 5. Cloud Deployment (`cloud_deployment.png`)

**What to capture:** Proof that API is running on cloud infrastructure

**Option A - Cloud VM (current setup):**
1. SSH to your cloud server OR access cloud dashboard
2. Run:
   ```bash
   docker ps | grep heart-predict-api
   curl http://localhost:80/health
   ```
3. Take a screenshot showing:
   - Docker container running
   - Health check response
   - Server IP/hostname visible

**Option B - Kubernetes:**
1. Run:
   ```bash
   kubectl get pods,svc -l app=heart-predict-api
   kubectl logs <pod-name> --tail=20
   ```
2. Take a screenshot showing:
   - Pods in "Running" state
   - Service with EXTERNAL-IP assigned
   - Recent logs from pod

**Option C - Cloud Console:**
1. Login to your cloud provider dashboard (GCP, AWS, Azure)
2. Navigate to VM instances or Kubernetes clusters
3. Take a screenshot showing:
   - Running instance/cluster
   - Public IP address
   - Status: Running/Healthy

---

### 6. Metrics Endpoint (`metrics_endpoint.png`)

**What to capture:** API metrics showing predictions made

**Steps:**
1. Make a few prediction requests (at least 5-10)
2. Access metrics endpoint:
   ```bash
   curl http://localhost:8000/metrics
   # or visit in browser
   ```
3. Take a screenshot showing:
   - JSON response with counters:
     ```json
     {
       "total_requests": 47,
       "total_predictions": 42,
       "positive_predictions": 23,
       "negative_predictions": 19
     }
     ```

**What to highlight:**
- Non-zero values (proving API has been used)
- Positive and negative predictions both present

---

### 7. EDA Visualizations (`eda_plots.png`)

**What to capture:** Generated EDA plots

**Steps:**
1. Run EDA script:
   ```bash
   python -m src.eda.visualize
   ```
2. Open generated plots from `artifacts/eda/`:
   - `histograms.png`
   - `correlation_heatmap.png`
   - `class_balance.png`
3. Create a collage or take 3 separate screenshots showing all plots

**What to highlight:**
- Clear visualizations of data distributions
- Correlation matrix with color coding
- Class balance (showing ~54% disease, 46% no disease)

---

### 8. Test Coverage Report (`test_coverage.png`)

**What to capture:** pytest coverage report

**Steps:**
1. Run tests locally:
   ```bash
   pytest tests/ -v --cov=src --cov-report=html
   ```
2. Open `htmlcov/index.html` in browser
3. Take a screenshot showing:
   - Overall coverage percentage (should be >60%)
   - Coverage by module
   - Green bars for well-covered modules

**Alternative - Terminal output:**
```bash
pytest tests/ --cov=src --cov-report=term-missing
```
Screenshot terminal showing coverage table.

---

### 9. Docker Build Success (`docker_build.png`)

**What to capture:** Successful Docker build

**Steps:**
1. Build Docker image:
   ```bash
   docker build -f docker/Dockerfile -t heart-predict-api .
   ```
2. Take a screenshot showing:
   - Build steps completing
   - Final line: "Successfully tagged heart-predict-api:latest"
   - Build time

**Bonus:** Show image in Docker Desktop or via:
```bash
docker images | grep heart-predict-api
```

---

### 10. Model Artifacts (`model_artifacts.png`)

**What to capture:** Generated artifacts from training

**Steps:**
1. After running training, list artifacts:
   ```bash
   ls -lh models/
   ls -lh artifacts/
   ls -lh mlflow/
   ```
2. Take a screenshot showing:
   - `models/heart_model.pkl` (file size ~100-500 KB)
   - `artifacts/eda/` plots
   - `artifacts/metrics.json`
   - `mlflow/mlflow.db`

---

## Screenshot Naming Convention

Use these exact filenames for consistency:

1. `ci_cd_pipeline.png` - GitHub Actions workflow
2. `mlflow_ui.png` - MLflow experiments dashboard
3. `api_swagger.png` - FastAPI Swagger UI
4. `api_predict_test.png` - Prediction request/response
5. `cloud_deployment.png` - Deployed API proof
6. `metrics_endpoint.png` - API metrics JSON
7. `eda_plots.png` - EDA visualizations collage
8. `test_coverage.png` - pytest coverage report
9. `docker_build.png` - Docker build success
10. `model_artifacts.png` - Generated files listing

## Tips for High-Quality Screenshots

1. **Use high resolution:** 1920x1080 or higher
2. **Crop unnecessary UI:** Focus on relevant content
3. **Ensure text is readable:** Zoom in if needed
4. **Avoid sensitive data:** Hide API keys, personal info
5. **Include timestamps:** Shows recency of work
6. **Use consistent browser/terminal:** Professional appearance

## Screenshot Tools

**Windows:**
- Snipping Tool (built-in)
- Snip & Sketch (Win + Shift + S)
- ShareX (free, advanced)

**macOS:**
- Cmd + Shift + 4 (region)
- Cmd + Shift + 3 (full screen)
- Preview (for editing)

**Linux:**
- Flameshot (recommended)
- GNOME Screenshot
- Spectacle (KDE)

## Annotations (Optional)

Consider adding annotations to highlight key elements:
- Red arrows pointing to important metrics
- Red boxes around successful status indicators
- Text labels explaining what's shown

**Tools:**
- Microsoft Paint / Paint 3D
- GIMP (free)
- Snagit (paid)
- Annotate feature in macOS Preview

## After Capturing Screenshots

1. Place all screenshots in this `screenshots/` directory
2. Verify filenames match the convention above
3. Add to git:
   ```bash
   git add screenshots/*.png
   git commit -m "Add deployment and monitoring screenshots"
   ```
4. Reference screenshots in `docs/report.md` if needed

## Example Screenshot Checklist

Use this checklist to track progress:

- [ ] `ci_cd_pipeline.png` - GitHub Actions green checkmarks
- [ ] `mlflow_ui.png` - Experiment runs with metrics
- [ ] `api_swagger.png` - FastAPI documentation
- [ ] `api_predict_test.png` - Successful prediction
- [ ] `cloud_deployment.png` - Cloud/k8s pods running
- [ ] `metrics_endpoint.png` - API metrics JSON
- [ ] `eda_plots.png` - Histograms + heatmap + balance
- [ ] `test_coverage.png` - Coverage >60%
- [ ] `docker_build.png` - Build success message
- [ ] `model_artifacts.png` - Generated files

---

**Note:** Screenshots are REQUIRED for assignment acceptance. They provide proof that:
1. Your CI/CD pipeline works end-to-end
2. Your models trained successfully
3. Your API is deployed and functional
4. Your monitoring is operational

Allocate ~30-45 minutes to capture all required screenshots.
