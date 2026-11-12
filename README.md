# Bank Customer Churn Prediction — Streamlit App

## Files required
Keep these three files in the same folder:
- `app.py`
- `churn_rf_model.pkl`
- `scaler (4).pkl`

## Run locally
1. Install Python 3.10–3.12.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start Streamlit:

   ```bash
   streamlit run app.py
   ```

## Model details
- Model: RandomForestClassifier
- Preprocessor: StandardScaler
- Input order: CreditScore, Geography, Gender, Age, Tenure, Balance,
  NumOfProducts, IsActiveMember, EstimatedSalary
- Encoding assumed by the app:
  - Geography: France=0, Germany=1, Spain=2
  - Gender: Female=0, Male=1

**Important:** The categorical encoding must match the encoding used when
training the model. If you used a different mapping, update the mappings
in `app.py` before deployment.
