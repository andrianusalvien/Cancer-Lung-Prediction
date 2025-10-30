import streamlit as st
import pandas as pd
import numpy as np
import os
import subprocess
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

# === Konfigurasi halaman ===
st.set_page_config(page_title="Lung Cancer Risk Prediction", layout="wide")

# === Unduh dataset Kaggle  ===
DATASET_SLUG = "thedevastator/cancer-patients-and-air-pollution-a-new-link"
DOWNLOAD_DIR = "data_kaggle"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

csv_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.csv"))
if not csv_files:
    st.info("Mengunduh dataset dari Kaggle")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", DATASET_SLUG,
            "--unzip", "-p", DOWNLOAD_DIR
        ], check=True)
        st.success("✅ Dataset berhasil diunduh.")
    except Exception as e:
        st.error(f"Gagal mengunduh dataset dari Kaggle: {e}")
        st.stop()

csv_files = glob.glob(os.path.join(DOWNLOAD_DIR, "*.csv"))
if not csv_files:
    st.error("❌ Tidak ditemukan file CSV setelah unzip.")
    st.stop()

DATA_FILE = csv_files[0]

@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data(DATA_FILE)
df_proc = df.copy()
if "index" in df_proc.columns:
    df_proc.drop(columns=["index"], inplace=True)
if "Patient Id" in df_proc.columns:
    df_proc.set_index("Patient Id", inplace=True)
if "Level" not in df_proc.columns:
    st.error("Kolom 'Level' tidak ditemukan di dataset.")
    st.stop()

level_map = {"Low": 0, "Medium": 1, "High": 2}
df_proc["Level_num"] = df_proc["Level"].map(level_map)

numeric_cols = df_proc.select_dtypes(include=[np.number]).columns.tolist()
if "Level_num" in numeric_cols:
    numeric_cols.remove("Level_num")

# === Sidebar Navigasi ===
st.sidebar.title("📌Navigasi")
if "page" not in st.session_state:
    st.session_state.page = "EDA"

if st.sidebar.button("🌐 EDA"):
    st.session_state.page = "EDA"
if st.sidebar.button("🤖 Model Training"):
    st.session_state.page = "Model"
if st.sidebar.button("🧪 Robustness Test"):
    st.session_state.page = "Robust"
if st.sidebar.button("🧮 Prediksi Interaktif"):
    st.session_state.page = "Prediksi"

page = st.session_state.page

# === Helper Feature Importance ===
@st.cache_data
def get_feature_importance(df, numeric_cols):
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(df[numeric_cols].fillna(0), df["Level_num"])
    importance = pd.Series(rf.feature_importances_, index=numeric_cols).sort_values(ascending=False)
    return importance

# =======================================================
# Page1
#  EDA (Exploratory Data Analysis)
# =======================================================
if page == "EDA":
    st.title("Cancer Lung Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Jumlah Data", f"{df.shape[0]:,}")
    with col2:
        st.metric("Jumlah Fitur", f"{df.shape[1]:,}")
    with col3:
        st.metric("Jumlah Kategori Risiko", df["Level"].nunique())

    st.subheader("📊 Distribusi Level Risiko")
    st.bar_chart(df_proc["Level"].value_counts())

    st.subheader("🔍 Preview Data")
    st.dataframe(df.head(), use_container_width=True)

    with st.expander("📈 Statistik Ringkasan Data"):
        st.write(df.describe().T.style.format("{:.2f}"))

    with st.expander("🧩 Korelasi Antar Fitur"):
        corr = df_proc[numeric_cols + ["Level_num"]].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, ax=ax)
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

    with st.expander("📊 Distribusi Fitur terhadap Level Risiko"):
        selected_var = st.selectbox("Pilih fitur untuk dilihat distribusinya:", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x="Level", y=selected_var, data=df_proc, order=["Low", "Medium", "High"], palette="viridis")
        plt.title(f"Distribusi {selected_var} berdasarkan Level Risiko")
        st.pyplot(fig)

    st.subheader(" Feature Importance (Random Forest)")
    importance = get_feature_importance(df_proc, numeric_cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    importance.head(15).plot(kind="barh", ax=ax, color="#2196f3")
    ax.invert_yaxis()
    plt.title("Top 15 Fitur Paling Berpengaruh")
    st.pyplot(fig)

# =======================================================
# Page2 — MODEL TRAINING
# =======================================================
elif page == "Model":
    st.title("🤖 Model Training & Evaluasi")

    importance = get_feature_importance(df_proc, numeric_cols)
    top_n = st.slider("Tampilkan berapa fitur teratas (opsional)", 3, min(20, len(numeric_cols)), 10)
    st.dataframe(importance.head(top_n).to_frame("Importance"))

    # === Default fitur awal (jika belum diset) ===
    default_feature_list = [
        'Coughing of Blood', 'Obesity', 'Passive Smoker', 'Wheezing',
        'Balanced Diet', 'Fatigue', 'Alcohol use', 'Dust Allergy',
        'Clubbing of Finger Nails', 'Snoring', 'Air Pollution', 'Smoking'
    ]

    # Hanya simpan default pertama kali
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = [
            f for f in default_feature_list if f in numeric_cols
        ]

    # Multiselect fitur
    selected_features = st.multiselect(
        "Pilih fitur untuk model:",
        options=numeric_cols,
        default=st.session_state.selected_features,
        help="Pilih fitur yang ingin digunakan untuk pelatihan model"
    )
    st.session_state.selected_features = selected_features

    if not selected_features:
        st.warning("Pilih setidaknya satu fitur.")
        st.stop()

    X = df_proc[selected_features].fillna(0)
    y = df_proc["Level_num"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    st.subheader("📊 Perbandingan Performa Model (Accuracy, Recall, F1)")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=42, use_label_encoder=False, eval_metric="mlogloss"
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        results.append((name, acc, rec, f1))

    df_results = pd.DataFrame(results, columns=["Model", "Accuracy", "Recall", "F1 Score"])
    st.dataframe(df_results.style.format({"Accuracy": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"}))

    with st.expander("📊 Tampilkan Grafik Perbandingan Model"):
        fig, ax = plt.subplots(figsize=(6, 4))
        df_results.set_index("Model")[["Recall", "F1 Score", "Accuracy"]].plot(kind="bar", ax=ax)
        plt.title("Perbandingan Model (Accuracy, Recall, F1)")
        plt.ylabel("Score")
        st.pyplot(fig)

    df_sorted = df_results.sort_values(by=["Recall", "F1 Score", "Accuracy"], ascending=[False, False, False])
    best_model_name = df_sorted.iloc[0]["Model"]

    st.success(f"✔ Model terbaik berdasarkan Test (Recall → F1 → Accuracy): **{best_model_name}**")

    best_model = models[best_model_name]
    best_model.fit(X_train, y_train)
    st.session_state["model"] = best_model
    st.session_state["train_data"] = (X_train, X_test, y_train, y_test)

# =======================================================
# Page3 — ROBUSTNESS TEST
# =======================================================
elif page == "Robust":
    st.title("🧪 Robustness Test (Ketahanan Model)")

    if "model" not in st.session_state or "selected_features" not in st.session_state:
        st.warning("⚠️ Latih model dulu di halaman Model Training untuk mengunci fitur.")
        st.stop()

    selected_features = st.session_state.selected_features
    X_train, X_test, y_train, y_test = st.session_state["train_data"]

    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    stress_level = st.selectbox("Pilih tingkat tekanan (semakin tinggi semakin ekstrem):", [0.1, 0.3, 0.5, 1.0], index=0)
    st.caption("Data test dimodifikasi untuk menilai ketahanan model terhadap kondisi ekstrem.")

    X_stress = X_test.copy()
    noise = np.random.normal(0, stress_level * 10, X_stress.shape)
    outlier_multiplier = np.where(
        np.random.rand(*X_stress.shape) < stress_level / 2,
        1 + np.random.uniform(-3, 3, size=X_stress.shape),
        1
    )
    X_stress = X_stress * outlier_multiplier + noise

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            random_state=42, use_label_encoder=False, eval_metric="mlogloss"
        ),
    }

    st.subheader("📊 Hasil Robustness Test per Model")
    robust_results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_stress = model.predict(X_stress)
        acc = accuracy_score(y_test, y_pred_stress)
        rec = recall_score(y_test, y_pred_stress, average="weighted")
        f1 = f1_score(y_test, y_pred_stress, average="weighted")
        robust_results.append((name, acc, rec, f1))

    df_robust = pd.DataFrame(robust_results, columns=["Model", "Accuracy", "Recall", "F1 Score"])
    st.dataframe(df_robust.style.format({"Accuracy": "{:.3f}", "Recall": "{:.3f}", "F1 Score": "{:.3f}"}))

    with st.expander("📊 Lihat Grafik Robustness Test"):
        fig, ax = plt.subplots(figsize=(6, 4))
        df_robust.set_index("Model")[["Recall", "F1 Score", "Accuracy"]].plot(kind="bar", ax=ax)
        plt.title("Robustness Test - Performa Model di Data Ekstrem")
        plt.ylabel("Score")
        st.pyplot(fig)

    df_sorted = df_robust.sort_values(by=["Recall", "F1 Score", "Accuracy"], ascending=[False, False, False])
    best_robust_model = df_sorted.iloc[0]["Model"]

    st.success(f"✔ Model Paling Tahan Terhadap Data Noise: **{best_robust_model}** (Recall → F1 → Accuracy)")

# =======================================================
# Page 4 — PREDIKSI INTERAKTIF
# =======================================================
elif page == "Prediksi":
    st.title("🧮 Prediksi Interaktif")

    if "model" not in st.session_state:
        st.warning("Latih model dulu di halaman Model Training.")
        st.stop()

    model = st.session_state["model"]
    features = st.session_state["selected_features"]

    with st.form("predict_form"):
        st.write("Masukkan nilai fitur:")
        input_data = {}
        for col in features:
            vals = df_proc[col].dropna()
            mn, mx = float(vals.min()), float(vals.max())
            default = float(vals.median())
            input_data[col] = st.number_input(col, mn, mx, default, format="%.3f")
        submit = st.form_submit_button("Predict")

    if submit:
        x_new = np.array([list(input_data.values())])
        pred = model.predict(x_new)[0]
        inv_map = {0: "Low", 1: "Medium", 2: "High"}
        st.success(f"**Predicted Level:** {inv_map.get(pred, pred)}")
