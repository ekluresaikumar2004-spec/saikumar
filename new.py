# ============================================================
# APPENDIX A: COMPLETE PROJECT CODE (FINAL CLEAN VERSION)
# ============================================================

# =========================
# SECTION 1: IMPORTS
# =========================

from pathlib import Path
import os
import warnings
import sys
import logging
import io
import pandas as pd
import numpy as np
import time
import subprocess

# Auto-install critical packages if missing
def ensure_package(package_name):
    try:
        __import__(package_name)
    except ImportError:
        print(f"⚠️  Installing {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name, "-q"])

# Ensure critical packages
ensure_package("joblib")
ensure_package("scikit-learn")
ensure_package("openpyxl")

import joblib
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Suppress ALL warnings at the start
warnings.filterwarnings("ignore")
logging.getLogger('streamlit').setLevel(logging.ERROR)

# Redirect stderr to suppress internal warnings
sys.stderr = io.StringIO()

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
)
from sklearn.neural_network import MLPClassifier

# Set BASE_PATH early (needed for data file access)
BASE_PATH = Path(__file__).resolve().parent

# Optional ML libraries - gracefully handle if not installed
XGBClassifier = None
LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except (ImportError, ModuleNotFoundError):
    pass

try:
    from lightgbm import LGBMClassifier
except (ImportError, ModuleNotFoundError):
    pass


# =========================
# SECTION 2: DATA LOADING & PREPROCESSING
# =========================

def load_data(file_path):
    return pd.read_excel(file_path)


def preprocess_data(df):
    expense_cols = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 'Transport',
        'Eating_Out', 'Entertainment', 'Utilities', 'Healthcare',
        'Education', 'Miscellaneous'
    ]

    df['Total_Expense'] = df[expense_cols].sum(axis=1)
    df['Expense_Ratio'] = df['Total_Expense'] / df['Income']

    return df, expense_cols


# =========================
# SECTION 3: EDA
# =========================

def basic_eda(df):
    return df['Financial_Health'].value_counts(), df.corr(numeric_only=True)


# =========================
# SECTION 4: FEATURE ENGINEERING
# =========================

def create_target(df):
    def classify_health(ratio):
        if ratio < 0.5:
            return 'Minimal Spending'
        elif ratio < 0.8:
            return 'Neutral'
        else:
            return 'Over-spending'

    df['Financial_Health'] = df['Expense_Ratio'].apply(classify_health)
    df['Spending_Score'] = (1 - df['Expense_Ratio']).clip(0, 1) * 100
    return df


def prepare_features(df, expense_cols):
    features = ['Income', 'Age', 'Dependents', 'Occupation', 'City_Tier'] + expense_cols

    X = df[features].copy()
    X['Occupation'] = X['Occupation'].astype('category').cat.codes
    X['City_Tier'] = X['City_Tier'].astype('category').cat.codes

    encoder = LabelEncoder()
    y = encoder.fit_transform(df['Financial_Health'])

    return X, y, encoder


# =========================
# SECTION 5: MODEL TRAINING
# =========================

def _stratify_if_possible(y):
    y_counts = pd.Series(y).value_counts()
    return y if (y_counts >= 2).all() else None


def get_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(),
        "Gaussian NB": GaussianNB(),
        "SVM Linear": SVC(kernel='linear'),
        "SVM RBF": SVC(kernel='rbf'),
        "Random Forest": RandomForestClassifier(),
        "Extra Trees": ExtraTreesClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
        "Bagging": BaggingClassifier(),
        "MLP": MLPClassifier(max_iter=1000, early_stopping=True)
    }

    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(verbose=-1)

    return models


def train_and_evaluate(X, y, test_size):
    stratify = _stratify_if_possible(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = get_models()
    results = []

    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Time": round(time.time() - start, 3),
            "Confusion Matrix": confusion_matrix(y_test, pred),
            "Classification Report": classification_report(y_test, pred, output_dict=True)
        })

    return pd.DataFrame(results).sort_values(by="Accuracy", ascending=False), scaler


# =========================
# SECTION 6: EXPERIMENTS + TABLE
# =========================

def run_all_experiments(X, y):
    splits = {
        "80-20": 0.2,
        "70-30": 0.3,
        "60-40": 0.4,
        "75-25": 0.25
    }

    comparison_table = pd.DataFrame()

    for split_name, size in splits.items():
        results, _ = train_and_evaluate(X, y, size)

        temp = results[['Model', 'Accuracy']].copy()
        temp.rename(columns={"Accuracy": split_name}, inplace=True)

        if comparison_table.empty:
            comparison_table = temp
        else:
            comparison_table = comparison_table.merge(temp, on="Model", how="outer")

    return comparison_table


# =========================
# SECTION 7: FINAL MODEL (FORCED MLP)
# =========================

def train_final_model(X, y):
    stratify = _stratify_if_possible(y)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=stratify
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = MLPClassifier(max_iter=1000, early_stopping=True)
    model.fit(X_train, y_train)

    return model, scaler


def save_model(model, scaler, encoder, path):
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path / "model.pkl")
    joblib.dump(scaler, path / "scaler.pkl")
    joblib.dump(encoder, path / "encoder.pkl")


def ensure_artifacts(data_path=None):
    artifacts_path = BASE_PATH / "artifacts"
    model_file = artifacts_path / "model.pkl"
    scaler_file = artifacts_path / "scaler.pkl"
    encoder_file = artifacts_path / "encoder.pkl"

    # If all artifacts exist, no need to train
    if model_file.exists() and scaler_file.exists() and encoder_file.exists():
        print("✅ Artifacts found. Using existing models.")
        return

    print("⚠️  Artifacts not found. Training new models...")
    
    if data_path is None:
        # Try multiple data paths
        possible_paths = [
            Path(r"C:\Users\subha\Downloads\age_18_24_without_retired.xlsx"),
            BASE_PATH / "data" / "raw" / "age_18_24_without_retired.xlsx",
            BASE_PATH / "age_18_24_without_retired.xlsx"
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                print(f"📂 Found data file: {data_path}")
                break
        if data_path is None:
            print("❌ Error: Data file not found. Searched paths:")
            for p in possible_paths:
                print(f"   - {p}")
            return

    df = load_data(data_path)
    df, expense_cols = preprocess_data(df)
    df = create_target(df)

    X, y, encoder = prepare_features(df, expense_cols)

    comparison_table = run_all_experiments(X, y)
    print("\nMODEL COMPARISON TABLE:\n")
    print(comparison_table)

    model, scaler = train_final_model(X, y)
    save_model(model, scaler, encoder, artifacts_path)

    print("Final model trained and saved in 'artifacts' directory.")


# =========================
# SECTION 8: STREAMLIT UI
# =========================

import streamlit as st
import socket

BASE_PATH = Path(__file__).resolve().parent


def get_network_ip():
    """Get the actual network IP address of this machine"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


@st.cache_resource
def load_artifacts():
    model = joblib.load(BASE_PATH / "artifacts" / "model.pkl")
    scaler = joblib.load(BASE_PATH / "artifacts" / "scaler.pkl")
    encoder = joblib.load(BASE_PATH / "artifacts" / "encoder.pkl")
    return model, scaler, encoder


def run_app():
    st.title("Youth Financial Health Monitor")
    st.markdown("---")
    
    # Print link to terminal on startup
    if "link_printed" not in st.session_state:
        network_ip = get_network_ip()
        print("\n" + "="*60)
        print("🎯 STREAMLIT UI LINK:")
        print("📍 Local URL: http://localhost:8510")
        print(f"🌐 Network URL: http://{network_ip}:8510")
        print("="*60 + "\n")
        st.session_state.link_printed = True
    
    # Load artifacts (no data loading needed for predictions)
    try:
        model, scaler, encoder = load_artifacts()
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        st.info("Please run the script in CLI mode first to train the model: `python new.py`")
        return

    # Display model info
    st.info("📊 **Model Information**")
    model_name = type(model).__name__
    st.write(f"**Active Model:** `{model_name}`")
    st.write(f"**Encoder Classes:** {list(encoder.classes_)}")

    st.markdown("---")
    st.subheader("💰 Enter Your Financial Details")

    col1, col2 = st.columns(2)
    with col1:
        income = st.number_input("Income", value=30000.0, min_value=0.0)
        age = st.slider("Age", 18, 24, 21)
        dependents = st.number_input("Dependents", 0, 5, 0, step=1)

    with col2:
        rent = st.number_input("Rent", value=5000.0, min_value=0.0)
        groceries = st.number_input("Groceries", value=3000.0, min_value=0.0)
        transport = st.number_input("Transport", value=2000.0, min_value=0.0)

    col3, col4 = st.columns(2)
    with col3:
        eating = st.number_input("Eating Out", value=2000.0, min_value=0.0)
        entertainment = st.number_input("Entertainment", value=1500.0, min_value=0.0)
        education = st.number_input("Education", value=1000.0, min_value=0.0)

    with col4:
        misc = st.number_input("Misc", value=1000.0, min_value=0.0)
        loan = st.number_input("Loan Repayment", value=0.0, min_value=0.0)
        insurance = st.number_input("Insurance", value=1000.0, min_value=0.0)

    utilities = st.number_input("Utilities", value=1500.0, min_value=0.0)
    healthcare = st.number_input("Healthcare", value=500.0, min_value=0.0)

    st.markdown("---")

    if st.button("🔍 Analyze Financial Health", key="analyze_button"):
        # Calculate metrics
        total_expense = rent + loan + insurance + groceries + transport + eating + entertainment + utilities + healthcare + education + misc
        expense_ratio = total_expense / income if income > 0 else 0
        spending_score = (1 - expense_ratio) * 100 if expense_ratio <= 1 else 0

        # Make prediction
        features = np.array([[
            income, age, dependents, 0, 0,
            rent, loan, insurance, groceries, transport,
            eating, entertainment, utilities, healthcare,
            education, misc
        ]])

        scaled = scaler.transform(features)
        pred = model.predict(scaled)
        result = encoder.inverse_transform(pred)[0]

        # Display results
        st.markdown("---")
        st.subheader("✅ Analysis Results")
        
        col_result1, col_result2 = st.columns(2)
        
        with col_result1:
            st.metric("Financial Health Status", result)
            
        with col_result2:
            st.metric("Spending Score", f"{spending_score:.1f}%")

        # Detailed metrics
        st.markdown("### 📈 Evaluation Metrics")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Income", f"₹{income:,.0f}")
        with col_m2:
            st.metric("Total Expenses", f"₹{total_expense:,.0f}")
        with col_m3:
            st.metric("Expense Ratio", f"{expense_ratio:.2%}")
        with col_m4:
            st.metric("Savings Potential", f"₹{max(0, income - total_expense):,.0f}")

        # Breakdown
        st.markdown("### 💸 Expense Breakdown")
        expense_data = {
            "Rent": rent,
            "Loan Repayment": loan,
            "Insurance": insurance,
            "Groceries": groceries,
            "Transport": transport,
            "Eating Out": eating,
            "Entertainment": entertainment,
            "Utilities": utilities,
            "Healthcare": healthcare,
            "Education": education,
            "Miscellaneous": misc
        }
        
        expense_df = pd.DataFrame(list(expense_data.items()), columns=["Category", "Amount"])
        st.bar_chart(expense_df.set_index("Category"))

        # Recommendation link
        st.markdown("---")
        st.success("✨ Analysis Complete! Use the results above to improve your financial health.")
        
        if result == "Minimal Spending":
            st.info("🎯 **Advice:** You're doing great! Consider investing your savings for future growth.")
            st.markdown("[Learn about investments →](https://www.example.com/investments)")
        elif result == "Neutral":
            st.info("⚖️ **Advice:** Your spending is balanced. Monitor your expenses regularly.")
            st.markdown("[Financial planning tips →](https://www.example.com/planning)")
        else:  # Over-spending
            st.warning("⚠️ **Advice:** You're overspending! Consider budgeting strategies to reduce expenses.")
            st.markdown("[Budgeting guide →](https://www.example.com/budgeting)")


# =========================
# MAIN EXECUTION
# =========================

if __name__ == "__main__":
    # Check if Streamlit is running by looking for Streamlit env vars or module
    is_streamlit = False
    try:
        import streamlit as st
        if hasattr(st, "session_state"):
            is_streamlit = True
    except:
        pass
    
    # Also check env vars
    streamlit_env = (
        os.getenv("STREAMLIT_SERVER_PORT") or
        os.getenv("STREAMLIT_SERVER_HEADERS")
    )
    
    if is_streamlit or streamlit_env or "streamlit" in " ".join(os.sys.argv if hasattr(os, "sys") else []):
        run_app()
    else:
        # Try multiple data paths
        possible_paths = [
            Path(r"C:\Users\subha\Downloads\age_18_24_without_retired.xlsx"),
            BASE_PATH / "data" / "raw" / "age_18_24_without_retired.xlsx",
            BASE_PATH / "age_18_24_without_retired.xlsx"
        ]
        data_path = None
        for p in possible_paths:
            if p.exists():
                data_path = p
                break
        if data_path is None:
            print("Error: Data file not found. Searched paths:")
            for p in possible_paths:
                print(f"  - {p}")
        else:
            df = load_data(data_path)
            df, expense_cols = preprocess_data(df)
            df = create_target(df)

            X, y, encoder = prepare_features(df, expense_cols)

            comparison_table = run_all_experiments(X, y)
            print("\nMODEL COMPARISON TABLE:\n")
            print(comparison_table)

            model, scaler = train_final_model(X, y)
            save_model(model, scaler, encoder, Path("artifacts"))

            print("Final model trained and saved in 'artifacts' directory.")