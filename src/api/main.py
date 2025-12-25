from fastapi import FastAPI

app = FastAPI(title="Heart Disease Prediction API")


@app.get("/")
def home():
    return {"message": "Welcome to Heart Disease Prediction.","created_by":"Group 5"}

@app.get("/health")
def health():
    return {"status": "ok"}