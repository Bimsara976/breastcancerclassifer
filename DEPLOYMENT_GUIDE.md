# Streamlit Community Cloud Deployment Guide

## 1. Test the project locally

Open a terminal in the project folder and run:

```bash
python -m venv .venv
```

Activate the environment.

**Windows**

```bash
.venv\Scripts\activate
```

**macOS/Linux**

```bash
source .venv/bin/activate
```

Install the dependencies and start the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Confirm that:

- the app opens without an import error;
- a single prediction can be produced;
- `sample_batch_input.csv` works in the batch tab;
- the medical-use disclaimer is visible.

## 2. Create the GitHub repository

1. Sign in to GitHub.
2. Create a new repository, for example:
   `COM763-breast-cancer-ml-pipeline`.
3. Use a public repository unless your Streamlit account and assignment rules
   support a private repository.
4. Upload the **contents** of this project folder, preserving this structure:

```text
COM763-breast-cancer-ml-pipeline/
├── .streamlit/
│   └── config.toml
├── model/
│   └── breast_cancer_pipeline.joblib
├── COM763_Task1_Breast_Cancer_ML_Pipeline.ipynb
├── app.py
├── DEPLOYMENT_GUIDE.md
├── README.md
├── requirements.txt
└── sample_batch_input.csv
```

5. Commit the files to the `main` branch.
6. Open the GitHub repository and verify that the notebook and model folder are
   visible.

## 3. Deploy on Streamlit Community Cloud

1. Open Streamlit Community Cloud and sign in with the GitHub account that owns
   the repository.
2. Choose **Create app** or **Deploy an app**.
3. Select:
   - Repository: your GitHub repository
   - Branch: `main`
   - Main file path: `app.py`
4. Open **Advanced settings**.
5. Select Python 3.13 for the closest match to the generated model environment.
   Python 3.12 should also work with the pinned scikit-learn version, but rerunning
   the notebook in the same Python version used for deployment is the safest
   reproducibility practice.
6. No secrets are required for this application.
7. Click **Deploy**.
8. Wait while Community Cloud installs `requirements.txt` and launches `app.py`.

## 4. Copy the URL for the assignment

After deployment, copy the final `.streamlit.app` URL and test it in a private
browser window. Insert that URL into the Portfolio Task 1 report.

## 5. Troubleshooting

### `ModuleNotFoundError: No module named 'joblib'`

The app now loads the persisted bundle with the Python standard library, so it
should not require a separate `joblib` install at runtime. If the cloud app
still shows an old error, commit the dependency change, then reboot the app
from **Manage app** so Community Cloud rebuilds the environment.

### Model file not found

Confirm that this exact path exists in GitHub:

```text
model/breast_cancer_pipeline.joblib
```

GitHub folder names and file names are case-sensitive on the cloud.

### Dependency or model-version error

Keep the `scikit-learn==1.8.0` pin. A joblib model should be loaded with the same
scikit-learn version used to create it.

### App remains on an error page

Open the deployed app, select **Manage app**, inspect the cloud logs, correct the
reported file or dependency error in GitHub, then reboot the app.

## 6. Recommended evidence for the technical report

Capture your own screenshots of:

- the deployed application and its public URL;
- the notebook data-audit assertions;
- the model-comparison table and chart;
- the tuning result and selected parameters;
- the threshold-analysis chart;
- the held-out metric table;
- the confusion matrix;
- ROC and precision-recall curves;
- permutation importance;
- the GitHub repository showing the `.ipynb`, `app.py`, `requirements.txt` and
  model folder.

Write the submitted report in your own words.
