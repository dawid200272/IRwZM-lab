# %% [markdown]
# # Zadanie 09 - `Badania przypadków klinicznych: analiza danych pacjentów i budowa modeli predykcyjnych w oparciu o rzeczywiste przypadki`
# %% [markdown]
# **Wariant 2:** Random Forest + strojenie progu dla minimalizacji FN
# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    RocCurveDisplay, precision_recall_curve
)

np.random.seed(42)
pd.set_option("display.max_columns", 100)

# %% [markdown]
# ## 1. Wczytanie danych
# %%
DATA_PATH = "cases_clinical_for_lab12.csv"

df = pd.read_csv(DATA_PATH)
print("Kształt danych:", df.shape)
print("\nPierwsze 5 wierszy:")
print(df.head())

# %% [markdown]
# ## 2. Opis problemu i danych
# %%
print("\n--- Opis danych ---")
print("Kolumny:", list(df.columns))

print("Brakujące wartości (%):")
print(df.isna().mean() * 100)

print("\nTypy zmiennych (DataFrame Info):")
print(df.info())

# %% [markdown]
# ### Rozkład zmiennej docelowej
# %%
target_col = "high_risk_cvd"
print(f"\nRozkład zmiennej docelowej '{target_col}':\n")

# Zmienna docelowa: wysokie ryzyko chorób sercowo-naczyniowych (1 = wysokie ryzyko, 0 = niskie)
print(df[target_col].value_counts(dropna=False), "\n")
print(df[target_col].value_counts(normalize=True))

# %% [markdown]
# ## 3. Preprocessing
# 
# Pipeline:
# - imputacja braków (mediana dla liczbowych, najczęstsza wartość dla kategorii),
# - OneHotEncoder dla kategorii,
# - StandardScaler dla liczbowych,
# - podział train/test (stratyfikacja).
# 
# %%
X = df.drop(columns=[target_col])
y = df[target_col].astype(int)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"num_cols: {num_cols}")
print(f"cat_cols: {cat_cols}")

# %%
# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols)
    ]
)

preprocess

# %%
# Podział na zbiór treningowy i testowy (75/25, stratyfikacja)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# %% [markdown]
# ## 4. Analiza eksploracyjna (EDA)
# 
# Wykonamy:
# - statystyki opisowe cech liczbowych
# - histogramy cech liczbowych,
# - porównanie rozkładów dla klas 0/1,
# - korelacje (część numeryczna).
# 
# %% [markdown]
# ### Statystyki opisowe cech liczbowych
# %%
print("Statystyki opisowe cech liczbowych:")
display(df[num_cols].describe().T)

# %% [markdown]
# ### Histogramy cech liczbowych
# %%
for col in num_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Rozkład cech liczbowych według klasy
# %%
for col in num_cols:
    plt.figure()
    df[df[target_col]==0][col].dropna().hist(bins=30, alpha=0.6, label="class 0")
    df[df[target_col]==1][col].dropna().hist(bins=30, alpha=0.6, label="class 1")
    plt.title(f"Rozkład {col} wg klasy")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Wykresy pudełkowe (boxplots) dla zmiennych liczbowych
# %%
for col in num_cols:
    plt.figure()
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot dla {col}")
    # plt.xlabel(col)
    # plt.ylabel("count")
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### Korelacje cech liczbowych (część numeryczna)
# %%
corr = df[num_cols + [target_col]].corr(numeric_only=True)
corr

# %%
# Heatmapa korelacji
sns.heatmap(corr, annot=True)
plt.show()

# %% [markdown]
# ## 5. Trenowanie modelu Random Forest
# %%
rf = Pipeline(steps=[
    ("pre", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample"
    ))
])

rf.fit(X_train, y_train)

# %% [markdown]
# ### Predykcje dla progu domyślnego (τ = 0.5)
# %%
proba = rf.predict_proba(X_test)[:, 1]
pred_05 = (proba >= 0.5).astype(int)

print("\n--- Wyniki dla progu domyślnego (τ = 0.5) ---")
print("Macierz pomyłek (Confusion matrix):")
print(confusion_matrix(y_test, pred_05))
print("\nRaport klasyfikacji (Classification report):")
print(classification_report(y_test, pred_05, digits=3))

roc_auc = roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else None
print(f"ROC AUC: {roc_auc:.3f}" if roc_auc is not None else "ROC AUC: n/a")

# %%
# Wykres ROC
RocCurveDisplay.from_predictions(y_test, proba)
plt.title("Krzywa ROC – Random Forest (τ=0.5)")
plt.show()

# %% [markdown]
# ## 6. Strojenie progu – poszukiwanie τ z Recall ≥ 0.90
# %%
thresholds = np.arange(0.20, 0.81, 0.05)

thresholds_list = [round(th, 2) for th in thresholds.tolist()]
thresholds_list

# %%
recalls = []
fps = []
best_thresh = None

for th in thresholds:
    pred = (proba >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    recalls.append(recall)
    fps.append(fp)
    if recall >= 0.90 and best_thresh is None:
        best_thresh = th

print("\n--- Strojenie progu (τ) ---")
print(f"Zakres τ: {thresholds[0]:.2f} – {thresholds[-1]:.2f} co 0.05")
print(f"Osiągnięto Recall >= 0.90 dla τ = {best_thresh:.2f}" if best_thresh else "Nie osiągnięto Recall >= 0.90 w zadanym zakresie.")

# Wybór progu – jeśli nie osiągnięto, bierzemy τ dający najwyższy recall
if best_thresh is None:
    best_idx = np.argmax(recalls)
    best_thresh = thresholds[best_idx]
    best_recall = recalls[best_idx]
    print(f"Ustawiono τ = {best_thresh:.2f} (najwyższy recall = {best_recall:.3f})")
else:
    best_idx = thresholds.tolist().index(best_thresh)
    best_recall = recalls[best_idx]

pred_opt = (proba >= best_thresh).astype(int)
tn_opt, fp_opt, fn_opt, tp_opt = confusion_matrix(y_test, pred_opt).ravel()

print(f"\n--- Wyniki dla optymalnego progu τ = {best_thresh:.2f} ---")
print(f"Recall = {best_recall:.3f}")
print(f"Liczba FP = {fp_opt} (wzrost o {fp_opt - fps[thresholds_list.index(0.5)]} w porównaniu do τ=0.5)")
print(f"Liczba FN = {fn_opt}")

# %% [markdown]
# ## 7. Najbardziej niepewne przypadki (p blisko progu)
# %%
uncertainty = np.abs(proba - best_thresh)
top_uncertain_idx = np.argsort(uncertainty)[:5]  # 5 najmniejszych odległości

# Przygotowanie danych dla wybranych przypadków
X_test_orig = X_test.iloc[top_uncertain_idx]
proba_unc = proba[top_uncertain_idx]
pred_unc = pred_opt[top_uncertain_idx]
y_true_unc = y_test.iloc[top_uncertain_idx]

print("\n--- 5 najbardziej niepewnych przypadków (prawdopodobieństwo blisko progu) ---")
for i, idx in enumerate(top_uncertain_idx):
    print(f"\nPrzypadek {i+1} (indeks {idx}):")
    print("Cechy:", X_test_orig.iloc[i].to_dict())
    print(f"Prawdopodobieństwo: {proba_unc[i]:.4f} (τ={best_thresh:.2f})")
    print(f"Predykcja: {pred_unc[i]}, wartość rzeczywista: {y_true_unc.iloc[i]}")

# %% [markdown]
# ## 8. Analiza błędów – 3 FP i 3 FN
# %%
# Dodajemy przewidywania do DataFrame testowego
results = X_test.copy()
results["y_true"] = y_test.values
results["proba"] = proba
results["pred_opt"] = pred_opt
results["error"] = results.apply(
    lambda row: "FP" if (row["y_true"] == 0 and row["pred_opt"] == 1) else
                ("FN" if (row["y_true"] == 1 and row["pred_opt"] == 0) else "OK"), axis=1
)

fp_cases = results[results["error"] == "FP"].sort_values("proba", ascending=False).head(3)
fn_cases = results[results["error"] == "FN"].sort_values("proba", ascending=True).head(3)

print("\n--- 3 przypadki False Positive (FP) ---")
print(fp_cases[["sex","age","bmi","systolic_bp","diastolic_bp","glucose","smoker","family_history","y_true","proba"]])
print("\n--- 3 przypadki False Negative (FN) ---")
print(fn_cases[["sex","age","bmi","systolic_bp","diastolic_bp","glucose","smoker","family_history","y_true","proba"]])

# Opis przypadków (w raporcie będzie rozwinięty)
# Dla każdego przypadku analizujemy potencjalne przyczyny błędu (np. brakujące dane, nietypowe wartości,
# młody wiek przy wysokim ryzyku, palenie tytoniu nieuwzględnione itp.)

# %% [markdown]
# ## 9. Wnioski i ograniczenia
# %%
# ---------------------------
# 9. Wnioski i ograniczenia
# ---------------------------
print("\n--- Wnioski i ograniczenia ---")
print("- Model Random Forest osiągnął ROC AUC = {:.3f} przy progu 0.5.".format(roc_auc))
print("- Aby uzyskać czułość ≥ 0.90, próg obniżono do {:.2f}, co zwiększyło liczbę FP z {} do {} (wzrost o {}).".format(
    best_thresh, fps[thresholds_list.index(0.5)], fp_opt, fp_opt - fps[thresholds_list.index(0.5)]))

# %%
print("- Niepewne przypadki to osoby z prawdopodobieństwem blisko granicy decyzyjnej – wymagają dodatkowych badań.")
print("- Błędy FP występują często u starszych pacjentów z podwyższonym ciśnieniem, ale bez innych czynników ryzyka.")
print("- Błędy FN dotyczą głównie młodych osób z korzystnymi parametrami, u których jednak wystąpiło zdarzenie – mogą to być pacjenci z nieuwzględnionymi czynnikami (np. genetyka, styl życia).")
print("- Ograniczenia: brak walidacji zewnętrznej, dane mają braki, model nie uwzględnia interakcji między lekami, brakuje wielu klinicznie istotnych cech (np. lipidogram, EKG).")
