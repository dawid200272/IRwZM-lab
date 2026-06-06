import os
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt

DATA_PATH = "health_measurements.csv"
MODEL_FULL_PATH = "risk_model_full.joblib"
MODEL_CLEAN_PATH = "risk_model_clean.joblib"

# Dopuszczalne zakresy dla walidacji wejściowej (Etap 1)
VALID_RANGES = {
    "age": (18, 110),
    "bmi": (10.0, 60.0),
    "glucose": (40, 300),
    "systolic_bp": (70, 260),
    "diastolic_bp": (40, 150)
}

st.set_page_config(page_title="Monitor zdrowia + ML (Wariant 2)", layout="centered")
st.title("📱 Monitor zdrowia + analiza ML – Walidacja i obsługa błędów (Wariant 2)")

# -----------------------------
# Pomocnicze: inicjalizacja CSV
# -----------------------------
def ensure_data_file():
    if not os.path.exists(DATA_PATH):
        df = pd.DataFrame(columns=[
            "timestamp", "age", "bmi", "glucose", "systolic_bp", "diastolic_bp"
        ])
        df.to_csv(DATA_PATH, index=False)

def load_data():
    ensure_data_file()
    return pd.read_csv(DATA_PATH)

def append_measurement(row: dict):
    df = load_data()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

def validate_inputs(age, bmi, glucose, sbp, dbp):
    """Sprawdza czy wartości mieszczą się w zdefiniowanych zakresach."""
    errors = []
    if not (VALID_RANGES["age"][0] <= age <= VALID_RANGES["age"][1]):
        errors.append(f"Wiek powinien być między {VALID_RANGES['age'][0]} a {VALID_RANGES['age'][1]}.")
    if not (VALID_RANGES["bmi"][0] <= bmi <= VALID_RANGES["bmi"][1]):
        errors.append(f"BMI powinno być między {VALID_RANGES['bmi'][0]} a {VALID_RANGES['bmi'][1]}.")
    if not (VALID_RANGES["glucose"][0] <= glucose <= VALID_RANGES["glucose"][1]):
        errors.append(f"Glukoza powinna być między {VALID_RANGES['glucose'][0]} a {VALID_RANGES['glucose'][1]} mg/dl.")
    if not (VALID_RANGES["systolic_bp"][0] <= sbp <= VALID_RANGES["systolic_bp"][1]):
        errors.append(f"SBP powinno być między {VALID_RANGES['systolic_bp'][0]} a {VALID_RANGES['systolic_bp'][1]} mmHg.")
    if not (VALID_RANGES["diastolic_bp"][0] <= dbp <= VALID_RANGES["diastolic_bp"][1]):
        errors.append(f"DBP powinno być między {VALID_RANGES['diastolic_bp'][0]} a {VALID_RANGES['diastolic_bp'][1]} mmHg.")
    return len(errors) == 0, errors

def make_demo_label(df: pd.DataFrame) -> pd.Series:
    """
    Etykieta do celów dydaktycznych (nie jest diagnozą!):
    1 jeśli SBP>=140 lub DBP>=90, inaczej 0.
    """
    return ((df["systolic_bp"] >= 140) | (df["diastolic_bp"] >= 90)).astype(int)

def detect_outliers_iqr(df: pd.DataFrame, columns):
    """Zwraca maskę boolean (True = wiersz zawiera outlier w którejkolwiek z kolumn)."""
    outlier_mask = pd.Series(False, index=df.index)
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outlier_mask |= (df[col] < lower) | (df[col] > upper)
    return outlier_mask

def train_models(df: pd.DataFrame):
    """
    Trenuje dwa modele: na wszystkich danych oraz na danych bez outlierów.
    Zwraca słowniki z modelami i metrykami.
    """
    if len(df) < 20:
        raise ValueError("Za mało danych do trenowania (min. 20 pomiarów). Dodaj więcej wpisów.")

    y = make_demo_label(df)
    X = df[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]].copy()

    # Wykrywanie outlierów
    outlier_mask = detect_outliers_iqr(df, X.columns)
    X_clean = X[~outlier_mask]
    y_clean = y[~outlier_mask]

    # Wspólny preprocesor (potok)
    num_cols = list(X.columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="drop"
    )

    def train_and_eval(X_data, y_data, label):
        if len(X_data) < 10:  # zbyt mało do podziału test/train
            return None, None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_data, y_data, test_size=0.25, random_state=42, stratify=y_data
        )
        clf = Pipeline(steps=[
            ("pre", preprocessor),
            ("model", LogisticRegression(max_iter=2000))
        ])
        clf.fit(X_tr, y_tr)
        proba = clf.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_te, pred)),
            "roc_auc": float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else None,
            "confusion_matrix": confusion_matrix(y_te, pred).tolist(),
            "report": classification_report(y_te, pred, digits=3)
        }
        return clf, metrics

    # Trenowanie na pełnym zbiorze
    model_full, metrics_full = train_and_eval(X, y, "full")
    # Trenowanie na zbiorze bez outlierów
    model_clean, metrics_clean = train_and_eval(X_clean, y_clean, "clean")

    # Zapis modeli (jeśli istnieją)
    if model_full:
        joblib.dump({"model": model_full, "metrics": metrics_full}, MODEL_FULL_PATH)
    if model_clean:
        joblib.dump({"model": model_clean, "metrics": metrics_clean}, MODEL_CLEAN_PATH)

    return model_full, metrics_full, model_clean, metrics_clean, outlier_mask

def load_model():
    """Ładuje model wytrenowany na czystych danych (priorytet)."""
    if os.path.exists(MODEL_CLEAN_PATH):
        obj = joblib.load(MODEL_CLEAN_PATH)
        return obj["model"], obj["metrics"], "clean"
    elif os.path.exists(MODEL_FULL_PATH):
        obj = joblib.load(MODEL_FULL_PATH)
        return obj["model"], obj["metrics"], "full"
    return None, None, None

def is_input_outlier(age, bmi, glucose, sbp, dbp, df_history):
    """Sprawdza czy bieżące dane są odstające względem historycznych (IQR)."""
    if len(df_history) < 5:
        return False, "Za mało danych historycznych do oceny odstających."
    new_row = pd.DataFrame([{
        "age": age, "bmi": bmi, "glucose": glucose,
        "systolic_bp": sbp, "diastolic_bp": dbp
    }])
    cols = ["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]
    outlier_info = []
    for col in cols:
        Q1 = df_history[col].quantile(0.25)
        Q3 = df_history[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        val = new_row[col].iloc[0]
        if val < lower or val > upper:
            outlier_info.append(f"{col}: {val} (norma {lower:.1f}–{upper:.1f})")
    if outlier_info:
        return True, "Odstające wartości: " + "; ".join(outlier_info)
    return False, "Wartości mieszczą się w typowym zakresie historycznym."

# =========================
# ETAP 1: Zbieranie danych + walidacja
# =========================
st.header("Etap 1 — Zbieranie danych zdrowotnych z walidacją")

with st.form("health_form", clear_on_submit=False):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Wiek [lata]", min_value=18, max_value=110, value=40, step=1)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)
        glucose = st.number_input("Glukoza [mg/dl]", min_value=40, max_value=300, value=95, step=1)
    with col2:
        systolic_bp = st.number_input("Ciśnienie skurczowe SBP [mmHg]", min_value=70, max_value=260, value=120, step=1)
        diastolic_bp = st.number_input("Ciśnienie rozkurczowe DBP [mmHg]", min_value=40, max_value=150, value=80, step=1)

    submitted = st.form_submit_button("💾 Zapisz pomiar")

if submitted:
    is_valid, errors = validate_inputs(age, bmi, glucose, systolic_bp, diastolic_bp)
    if not is_valid:
        for err in errors:
            st.error(f"❌ Błąd walidacji: {err}")
    else:
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "age": int(age),
            "bmi": float(bmi),
            "glucose": int(glucose),
            "systolic_bp": int(systolic_bp),
            "diastolic_bp": int(diastolic_bp),
        }
        append_measurement(row)
        st.success("✅ Zapisano pomiar do pliku health_measurements.csv")

df = load_data()
st.caption(f"Liczba zapisanych pomiarów: {len(df)}")
st.dataframe(df.tail(10), use_container_width=True)

# =====================================
# ETAP 2: Analiza + wykrywanie odstających wartości
# =====================================
st.header("Etap 2 — Analiza danych i wykrywanie wartości odstających (IQR)")

if len(df) == 0:
    st.info("Dodaj co najmniej jeden pomiar, aby zobaczyć analizę.")
else:
    # Statystyki opisowe
    st.subheader("Statystyki opisowe")
    st.dataframe(df[["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]].describe().T, use_container_width=True)

    # Wykrywanie outlierów
    numeric_cols = ["age", "bmi", "glucose", "systolic_bp", "diastolic_bp"]
    outliers_mask = detect_outliers_iqr(df, numeric_cols)
    df_outliers = df[outliers_mask]
    st.subheader("🔍 Wykryte wartości odstające (metoda IQR)")
    if len(df_outliers) > 0:
        st.write(f"Znaleziono {len(df_outliers)} wierszy z odstającymi wartościami:")
        st.dataframe(df_outliers[numeric_cols], use_container_width=True)
    else:
        st.success("Brak wykrytych wartości odstających w danych historycznych.")

    # Wykres trendu z zaznaczeniem outlierów
    st.subheader("Wykres trendu (ostatnie 50 pomiarów) z oznaczeniem outlierów")
    plot_cols = st.multiselect(
        "Wybierz parametry do wykresu:",
        options=["bmi", "glucose", "systolic_bp", "diastolic_bp"],
        default=["systolic_bp", "diastolic_bp"]
    )
    if plot_cols:
        df_plot = df.copy()
        df_plot["timestamp"] = pd.to_datetime(df_plot["timestamp"], errors="coerce")
        df_plot = df_plot.dropna(subset=["timestamp"]).sort_values("timestamp").tail(50)

        fig = plt.figure(figsize=(7, 4))
        for c in plot_cols:
            plt.plot(df_plot["timestamp"], df_plot[c], label=c, alpha=0.7)

        # Zaznacz outlier na wykresie (dla każdego wybranego parametru osobno)
        for c in plot_cols:
            outlier_indices = df_plot.index[detect_outliers_iqr(df_plot[[c]], [c])]
            if len(outlier_indices):
                plt.scatter(df_plot.loc[outlier_indices, "timestamp"],
                            df_plot.loc[outlier_indices, c],
                            color='red', s=50, zorder=5, label=f'outlier {c}')

        plt.xlabel("czas")
        plt.ylabel("wartość")
        plt.xticks(rotation=30, ha="right")
        plt.legend()
        plt.tight_layout()
        st.pyplot(fig)

    # Szybka flaga progowa (demo)
    st.subheader("Szybka flaga progowa (demo)")
    df_flag = df.tail(10).copy()
    df_flag["flag_high_bp"] = ((df_flag["systolic_bp"] >= 140) | (df_flag["diastolic_bp"] >= 90)).astype(int)
    st.dataframe(df_flag[["timestamp", "systolic_bp", "diastolic_bp", "flag_high_bp"]], use_container_width=True)

# ==============================
# ETAP 3: Budowa modelu i porównanie
# ==============================
st.header("Etap 3 — Model ML: porównanie przed i po usunięciu wartości odstających")

colA, colB = st.columns([1, 2])
with colA:
    if st.button("🧠 Wytrenuj / odśwież modele"):
        try:
            model_full, met_full, model_clean, met_clean, out_mask = train_models(df)
            st.session_state["model_full_metrics"] = met_full
            st.session_state["model_clean_metrics"] = met_clean
            st.session_state["outlier_mask"] = out_mask
            st.success("Modele wytrenowane i zapisane.")
        except Exception as e:
            st.error(str(e))

with colB:
    if "model_full_metrics" in st.session_state:
        st.subheader("Metryki – model na wszystkich danych")
        met_full = st.session_state["model_full_metrics"]
        if met_full:
            st.write(f"Accuracy: **{met_full['accuracy']:.3f}**")
            if met_full["roc_auc"]:
                st.write(f"ROC AUC: **{met_full['roc_auc']:.3f}**")
            st.text("Raport:\n" + met_full["report"])
            st.write("Macierz pomyłek:", met_full["confusion_matrix"])
        else:
            st.warning("Model na pełnych danych nie został utworzony (za mało próbek).")

        st.subheader("Metryki – model na danych BEZ odstających")
        met_clean = st.session_state["model_clean_metrics"]
        if met_clean:
            st.write(f"Accuracy: **{met_clean['accuracy']:.3f}**")
            if met_clean["roc_auc"]:
                st.write(f"ROC AUC: **{met_clean['roc_auc']:.3f}**")
            st.text("Raport:\n" + met_clean["report"])
            st.write("Macierz pomyłek:", met_clean["confusion_matrix"])
        else:
            st.warning("Model na czystych danych nie został utworzony (za mało próbek po odrzuceniu outlierów).")
    else:
        st.info("Kliknij przycisk, aby wytrenować modele i zobaczyć porównanie.")

# ===================================
# ETAP 4: Predykcja z blokowaniem dla niewiarygodnych danych
# ===================================
st.header("Etap 4 — Predykcja ryzyka (integracja z walidacją i wykrywaniem outlierów)")

model_active, _, model_type = load_model()
if model_active is None:
    st.warning("Najpierw wytrenuj modele w Etapie 3.")
else:
    st.info(f"📌 Aktywny model: **{model_type}** (model wytrenowany na danych {'czystych' if model_type=='clean' else 'pełnych'})")

    # Walidacja zakresów bieżących danych
    is_valid, errors = validate_inputs(age, bmi, glucose, systolic_bp, diastolic_bp)
    if not is_valid:
        st.error("❌ Predykcja zablokowana – dane wejściowe nie spełniają poprawnych zakresów:")
        for err in errors:
            st.write(f"- {err}")
    else:
        # Dodatkowe sprawdzenie, czy bieżące wartości nie są odstające względem historii
        df_hist = load_data()
        is_outlier, msg_outlier = is_input_outlier(age, bmi, glucose, systolic_bp, diastolic_bp, df_hist)
        if is_outlier:
            st.warning(f"⚠️ Predykcja może być niemiarodajna – wykryto odstające wartości w stosunku do historii.")
            st.write(f"**Uzasadnienie:** {msg_outlier}")
            st.write("Aby wykonać predykcję, popraw dane lub usuń outlier z historii.")
            # Zgodnie z treścią wariantu: "zablokuj predykcję i pokaż uzasadnienie"
            st.stop()   # zatrzymuje dalsze wykonywanie – predykcja nie zostanie wykonana

        # Predykcja
        X_one = pd.DataFrame([{
            "age": int(age),
            "bmi": float(bmi),
            "glucose": int(glucose),
            "systolic_bp": int(systolic_bp),
            "diastolic_bp": int(diastolic_bp),
        }])
        proba = float(model_active.predict_proba(X_one)[0, 1])
        pred = int(proba >= 0.5)

        st.write(f"Prawdopodobieństwo klasy „podwyższone ryzyko (demo)” = **{proba:.3f}**")
        if pred == 1:
            st.error("🔴 Wynik: **podwyższone ryzyko (demo)** — sprawdź pomiary i rozważ konsultację medyczną.")
        else:
            st.success("🟢 Wynik: **niskie ryzyko (demo)**")
        st.caption("Uwaga: demonstracja edukacyjna – nie jest to wyrób medyczny.")

st.divider()
st.caption("Pliki lokalne: health_measurements.csv (historia), risk_model_full.joblib, risk_model_clean.joblib (modele).")
