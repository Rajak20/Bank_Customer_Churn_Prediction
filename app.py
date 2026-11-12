import streamlit as st
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "churn_rf_model.pkl"
SCALER_PATH = BASE_DIR / "scaler (4).pkl"


# ---------------------------------------------------------
# Load trained model and scaler
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return model, scaler


try:
    model, scaler = load_artifacts()
except FileNotFoundError as error:
    st.error(
        f"Could not find {error.filename}. Keep both .pkl files in the "
        "same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Could not load the model files: {error}")
    st.stop()


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }
    .main-header {
        padding: 1.2rem 1.5rem;
        border-radius: 14px;
        background: linear-gradient(110deg, #142b4a, #2463a6);
        color: white;
        margin-bottom: 1.3rem;
    }
    .main-header h1 {
        color: white;
        margin-bottom: 0.25rem;
    }
    .main-header p {
        color: #e5eef9;
        margin-bottom: 0;
    }
    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #e6eaf0;
        padding: 15px;
        border-radius: 12px;
    }
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e6eaf0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Customer Churn Prediction</h1>
    <p>Predict customer retention using a trained Random Forest Classifier.</p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Enter the customer's details below. The application will estimate "
    "whether the customer is likely to leave the bank."
)

# Feature encoding must match the encoding used during model training:
# Geography: France=0, Germany=1, Spain=2
# Gender: Female=0, Male=1

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** Random Forest Classifier")
    st.write("**Preprocessing:** StandardScaler")
    st.write("**Input features:** 9")
    st.divider()
    st.caption(
        "The prediction is generated using your uploaded trained model "
        "and scaler."
    )


# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------
st.subheader("Customer Information")

with st.form("customer_details_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        credit_score = st.number_input(
            "Credit Score", min_value=300, max_value=900,
            value=650, step=1,
            help="Customer's credit score."
        )
        geography = st.selectbox(
            "Geography", ["France", "Germany", "Spain"]
        )
        gender = st.selectbox("Gender", ["Female", "Male"])

    with col2:
        age = st.number_input(
            "Age", min_value=18, max_value=100, value=35, step=1
        )
        tenure = st.number_input(
            "Tenure (Years)", min_value=0, max_value=10, value=5, step=1
        )
        balance = st.number_input(
            "Account Balance", min_value=0.0, value=75000.0,
            step=1000.0, format="%.2f"
        )

    with col3:
        num_products = st.selectbox(
            "Number of Bank Products", [1, 2, 3, 4], index=1
        )
        active_member = st.selectbox(
            "Is Active Member?", ["Yes", "No"]
        )
        estimated_salary = st.number_input(
            "Estimated Salary", min_value=0.0, value=100000.0,
            step=1000.0, format="%.2f"
        )

    submitted = st.form_submit_button(
        "Predict Customer Churn", type="primary", use_container_width=True
    )


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if submitted:
    geography_encoded = {"France": 0, "Germany": 1, "Spain": 2}[geography]
    gender_encoded = {"Female": 0, "Male": 1}[gender]
    active_encoded = 1 if active_member == "Yes" else 0

    # Keep this exact feature order consistent with scaler training.
    input_data = pd.DataFrame([[
        credit_score,
        geography_encoded,
        gender_encoded,
        age,
        tenure,
        balance,
        num_products,
        active_encoded,
        estimated_salary
    ]], columns=[
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "IsActiveMember",
        "EstimatedSalary"
    ])

    try:
        scaled_input = scaler.transform(input_data)
        prediction = int(model.predict(scaled_input)[0])

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(scaled_input)[0]
            class_probabilities = dict(zip(model.classes_, probabilities))
            churn_probability = float(class_probabilities.get(1, 0.0))
            stay_probability = float(class_probabilities.get(0, 0.0))
        else:
            churn_probability = None
            stay_probability = None

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ High Risk: This customer is predicted to churn.")
            st.write(
                "The model predicts that this customer is likely to leave "
                "the bank. Consider reviewing the customer's experience "
                "and retention options."
            )
        else:
            st.success("✅ Retention: This customer is predicted to stay.")
            st.write(
                "The model predicts that this customer is likely to remain "
                "with the bank."
            )

        result_col1, result_col2 = st.columns(2)
        with result_col1:
            st.metric(
                "Predicted Status",
                "Churn" if prediction == 1 else "Stay"
            )
        with result_col2:
            if churn_probability is not None:
                st.metric("Churn Probability", f"{churn_probability:.2%}")

        if churn_probability is not None:
            st.markdown("#### Prediction Probability")
            prob_col1, prob_col2 = st.columns(2)
            with prob_col1:
                st.write(f"**Churn:** {churn_probability:.2%}")
                st.progress(churn_probability)
            with prob_col2:
                st.write(f"**Stay:** {stay_probability:.2%}")
                st.progress(stay_probability)

        with st.expander("View submitted customer details"):
            st.dataframe(input_data, use_container_width=True, hide_index=True)

        st.caption(
            "Probabilities are model estimates, not guarantees of future "
            "customer behavior."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
        st.info(
            "Check that the feature encoding and feature order match the "
            "preprocessing used when training your model."
        )
else:
    st.info("Fill in the customer details and click **Predict Customer Churn**.")
