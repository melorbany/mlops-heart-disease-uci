# MLOps Assignment 1 - Heart Disease Prediction

End-to-end **machine learning + MLOps** pipeline for predicting heart disease using the **UCI Heart Disease (processed) datasets** and serving predictions via a **FastAPI** web service.

The project:

- Converts the original UCI `processed.*.data` files into a unified CSV
- Cleans and preprocesses the data
- Trains a binary classifier to predict heart disease
- Exposes the model through a production-ready API

## Project Overview

This repository demonstrates how to take a classic ML dataset from **raw files to a deployed prediction service**, focusing on:

- **Reproducible data processing**
- **Config-driven modeling**
- **Clear separation between data, code, models and API**
- A structure that’s friendly to **CI/CD**, **Dockerization**, and **cloud deployment**

Technologies used:

- **Python**, **pandas**, **NumPy**
- **scikit-learn** for modeling
- **FastAPI** + **Uvicorn** for the REST API
- Standard tooling for packaging and testing


## Dataset

We use the **UCI Heart Disease** datasets in their *processed* form:

- `processed.cleveland.data`
- `processed.hungarian.data`
- `processed.switzerland.data`
- `processed.va.data`

Each file has the same 14 original columns:

1. `age`
2. `sex`
3. `cp` – chest pain type
4. `trestbps` – resting blood pressure
5. `chol` – serum cholestoral
6. `fbs` – fasting blood sugar
7. `restecg` – resting electrocardiographic results
8. `thalach` – maximum heart rate achieved
9. `exang` – exercise induced angina
10. `oldpeak` – ST depression induced by exercise
11. `slope` – slope of the peak exercise ST segment
12. `ca` – number of major vessels colored by fluoroscopy
13. `thal` – 3 = normal; 6 = fixed defect; 7 = reversible defect
14. `num` – original target: 0 (no disease) to 4 (severe disease)


## References

- UCI Machine Learning Repository – Heart Disease Data Set  
  (search for “Heart Disease” on the UCI site)
- scikit-learn documentation – https://scikit-learn.org/
- FastAPI documentation – https://fastapi.tiangolo.com/

