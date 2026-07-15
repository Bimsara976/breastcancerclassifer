from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer ML Classifier",
    page_icon="🔬",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "breast_cancer_pipeline.joblib"


@st.cache_resource
def load_model_bundle():
    """Load the complete preprocessing-and-model pipeline once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the notebook and commit the model folder."
        )
    return joblib.load(MODEL_PATH)


def validate_batch(frame: pd.DataFrame, required_columns: list[str]) -> pd.DataFrame:
    """Validate and reorder uploaded batch data."""
    missing = sorted(set(required_columns) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(required_columns))

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if extra:
        frame = frame.drop(columns=extra)

    ordered = frame[required_columns].copy()
    ordered = ordered.apply(pd.to_numeric, errors="coerce")

    if ordered.isna().any().any():
        bad_columns = ordered.columns[ordered.isna().any()].tolist()
        raise ValueError(
            "Non-numeric or missing values were found in: "
            f"{bad_columns}"
        )

    return ordered


try:
    bundle = load_model_bundle()
except Exception as error:
    st.error("The trained model could not be loaded.")
    st.exception(error)
    st.stop()

pipeline = bundle["pipeline"]
threshold = float(bundle["threshold"])
feature_names = list(bundle["feature_names"])
feature_statistics = bundle["feature_statistics"]
example_profiles = bundle["example_profiles"]
test_metrics = bundle["test_metrics"]

st.title("Breast Cancer Diagnostic Classification")
st.caption(
    "Educational demonstration using the Wisconsin Diagnostic Breast Cancer dataset"
)

st.warning(
    "This application is not a medical device. It must not be used to diagnose, "
    "exclude, or treat cancer. Clinical decisions require qualified healthcare "
    "professionals and validated medical procedures."
)

tab_single, tab_batch, tab_info = st.tabs(
    ["Single Prediction", "Batch Prediction", "Model Information"]
)

with tab_single:
    st.subheader("Enter Cell-Nucleus Measurements")

    profile_name = st.selectbox(
        "Starting profile",
        ["Training median", "Benign example", "Malignant example"],
        help=(
            "Choose a starting profile, then modify any value. "
            "The examples are records from the training partition."
        ),
    )

    if profile_name == "Training median":
        defaults = {
            feature: float(feature_statistics[feature]["median"])
            for feature in feature_names
        }
    else:
        defaults = {
            feature: float(value)
            for feature, value in example_profiles[profile_name].items()
        }

    with st.form("single_prediction_form"):
        st.markdown(
            "The model expects all 30 numeric features. Values are constrained "
            "to the observed training range."
        )

        columns = st.columns(3)
        input_record = {}

        for index, feature in enumerate(feature_names):
            stats = feature_statistics[feature]
            minimum = float(stats["min"])
            maximum = float(stats["max"])
            default = float(defaults[feature])
            step = max((maximum - minimum) / 200.0, 0.000001)

            with columns[index % 3]:
                input_record[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    min_value=minimum,
                    max_value=maximum,
                    value=min(max(default, minimum), maximum),
                    step=step,
                    format="%.6f",
                    key=f"{profile_name}_{feature}",
                )

        submitted = st.form_submit_button(
            "Run Prediction",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        input_frame = pd.DataFrame(
            [[input_record[name] for name in feature_names]],
            columns=feature_names,
        )

        malignant_probability = float(
            pipeline.predict_proba(input_frame)[0, 1]
        )
        predicted_class = int(malignant_probability >= threshold)
        predicted_label = bundle["target_mapping"][predicted_class]

        metric_col, class_col = st.columns(2)
        with metric_col:
            st.metric(
                "Estimated malignant-class probability",
                f"{malignant_probability:.1%}",
            )
        with class_col:
            st.metric(
                "Model classification",
                predicted_label,
            )

        st.progress(
            min(max(malignant_probability, 0.0), 1.0),
            text=(
                f"Decision threshold: {threshold:.2f}. "
                "Probability values are model outputs, not clinical risk estimates."
            ),
        )

        if predicted_class == 1:
            st.error(
                "The measurements were classified as malignant-like by the model."
            )
        else:
            st.success(
                "The measurements were classified as benign-like by the model."
            )

with tab_batch:
    st.subheader("Batch CSV Prediction")
    st.write(
        "Upload a CSV containing the same 30 feature columns used in training. "
        "A ready-to-test sample is included in the repository."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            raw_batch = pd.read_csv(uploaded_file)
            batch = validate_batch(raw_batch, feature_names)

            probabilities = pipeline.predict_proba(batch)[:, 1]
            predictions = (probabilities >= threshold).astype(int)

            results = raw_batch.copy()
            results["malignant_probability"] = probabilities
            results["predicted_class"] = predictions
            results["predicted_label"] = [
                bundle["target_mapping"][int(value)]
                for value in predictions
            ]

            st.success(f"Processed {len(results)} rows.")
            st.dataframe(results, use_container_width=True)

            st.download_button(
                "Download Predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as error:
            st.error("The uploaded CSV could not be processed.")
            st.exception(error)

with tab_info:
    st.subheader("System Summary")

    st.markdown(
        f"""
        **Selected model:** {bundle['model_name']}  
        **Decision threshold:** {threshold:.2f}  
        **Dataset:** {bundle['dataset_description']}  
        **Features:** {len(feature_names)} numeric measurements  
        **Output:** benign-like or malignant-like classification
        """
    )

    metric_table = pd.DataFrame(
        {
            "Metric": [
                name.replace("_", " ").title()
                for name in test_metrics
            ],
            "Held-Out Test Value": [
                float(value) for value in test_metrics.values()
            ],
        }
    )
    metric_table["Held-Out Test Value"] = metric_table[
        "Held-Out Test Value"
    ].map(lambda value: f"{value:.4f}")

    st.dataframe(
        metric_table,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("#### Required Input Columns")
    st.dataframe(
        pd.DataFrame({"feature": feature_names}),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown(
        """
        #### Responsible-use boundary
        The dataset is small and historical, the deployment has not been clinically
        validated, and performance on other populations or measurement systems is
        unknown. Use the application only to demonstrate an end-to-end machine
        learning workflow.
        """
    )
