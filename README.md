# COM763 Portfolio Task 1: Breast Cancer ML Pipeline

This repository contains an end-to-end binary-classification project using the
Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.

## Files

- `COM763_Task1_Breast_Cancer_ML_Pipeline.ipynb` — executed development notebook
- `app.py` — Streamlit interface
- `model/breast_cancer_pipeline.joblib` — fitted preprocessing and model pipeline
- `sample_batch_input.csv` — sample CSV for batch testing
- `requirements.txt` — deployment dependencies
- `DEPLOYMENT_GUIDE.md` — local and Streamlit Community Cloud instructions

## Local run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Important

This is an educational demonstration and not a clinical diagnostic system.
