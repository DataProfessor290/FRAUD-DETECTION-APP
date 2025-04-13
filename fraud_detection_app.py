import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
import numpy as np

# Load data
@st.cache_data
def load_data():
    # Download only if the file doesn't already exist
    if not os.path.exists("Dataset.csv"):
        url = 'https://drive.google.com/uc?id=1sn7m5d98MRTzStiOyKNfQPWvbVURBZS8'  # <-- your file ID
        gdown.download(url, "Dataset.csv", quiet=False)

    df = pd.read_csv("Dataset.csv")
    df.columns = df.columns.str.lower()
    df = df[['type', 'amount', 'oldbalanceorg', 'newbalanceorig', 'isfraud']]
    return df

# Preprocess and prepare data
def split_cols(data):
    num_cols = data.select_dtypes(include=['number']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    return num_cols, cat_cols

def prepare_data(df):
    X = df.drop('isfraud', axis=1)
    y = df['isfraud']

    sampler = RandomUnderSampler()
    X_resampled, y_resampled = sampler.fit_resample(X, y)

    num_cols, cat_cols = split_cols(X_resampled)

    num_pipe = Pipeline([('scaler', StandardScaler())])
    cat_pipe = Pipeline([('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

    processor = ColumnTransformer([
        ('num', num_pipe, num_cols),
        ('cat', cat_pipe, cat_cols)
    ])

    x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    return processor, x_train, x_test, y_train, y_test

def get_model(processor, model, x_train, y_train):
    pipe = Pipeline([
        ('processor', processor),
        ('model', model)
    ])
    return pipe.fit(x_train, y_train)

def evaluate_model(model, x_test, y_test):
    pred = model.predict(x_test)
    f1 = f1_score(y_test, pred)
    report = classification_report(y_test, pred, output_dict=True)
    matrix = confusion_matrix(y_test, pred)
    return f1, report, matrix

# Main Streamlit app
st.set_page_config(page_title="Fraud Detection", layout="wide")
st.title("💳 Fraud Detection Model Explorer")

df = load_data()
processor, x_train, x_test, y_train, y_test = prepare_data(df)

models = {
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVC": SVC(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

tabs = st.tabs(list(models.keys()))

for i, (name, clf) in enumerate(models.items()):
    with tabs[i]:
        st.subheader(f"{name} Evaluation")
        model = get_model(processor, clf, x_train, y_train)
        f1, report, matrix = evaluate_model(model, x_test, y_test)

        st.metric("F1 Score", round(f1, 4))
        st.write("**Classification Report**")
        st.dataframe(pd.DataFrame(report).transpose())
        st.write("**Confusion Matrix**")
        st.write(matrix)

        st.divider()
        st.subheader("🔍 Predict a Transaction")
        with st.form(f"{name}_form"):
            tx_type = st.selectbox("Transaction Type", options=df["type"].unique())
            amount = st.number_input("Amount", min_value=0.0, value=1000.0)
            old_balance = st.number_input("Old Balance Origin", min_value=0.0, value=5000.0)
            new_balance = st.number_input("New Balance Origin", min_value=0.0, value=4000.0)
            submit = st.form_submit_button("Predict")

        if submit:
            input_df = pd.DataFrame([{
                'type': tx_type,
                'amount': amount,
                'oldbalanceorg': old_balance,
                'newbalanceorig': new_balance
            }])
            pred = model.predict(input_df)[0]
            pred_label = "Fraudulent" if pred == 1 else "Not Fraudulent"
            st.success(f"Prediction: **{pred_label}**")

