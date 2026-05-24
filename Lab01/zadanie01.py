# %% [markdown]
# # Zadanie 01 - `Praca z danymi medycznymi: import, przetwarzanie i analiza zbiorów danych pacjentów`
# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

pd.set_option("display.max_columns", 100)
# %% [markdown]
# ## Etap 1: Import danych
# %%
DATA_PATH = "dataset/Smoker_Epigenetic_df.csv"

def load_data(path: str) -> pd.DataFrame:
    _, ext = os.path.splitext(path.lower())
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        raise ValueError("Obsługiwane są tylko pliki .csv oraz .xlsx/.xls")
    return df

df = load_data(DATA_PATH)

# Podstawowe informacje
print("Kształt zbioru:", df.shape)

print("\nPierwsze 5 wierszy:")
print(df.head())

print("\nInformacje o danych:")
df.info()

print("\nTypy danych")
print(df.dtypes)
# %% [markdown]
# ## Etap 2: Wstępne przetwarzanie danych
# %%
# ------------------------------
# 2. Uzupełnienie brakujących danych i badanie zależności
# ------------------------------

# Sprawdzenie brakujących wartości
print("\nBrakujące wartości w każdej kolumnie:")
print(df.isnull().sum())

# W zbiorze brakuje wartości w kolumnach CpG dla ostatnich 62 wierszy.
# Należy uzupełnić je medianą z kolumn (dla każdej CpG osobno).
# Wybranie kolumn z danymi metylacji (zaczynające się od 'cg')
methyl_cols = [col for col in df.columns if col.startswith('cg')]

# Kopiowanie danych, aby nie modyfikować oryginału
df_imputed = df.copy()

# Imputacja medianą dla kolumn metylacji
imputer = SimpleImputer(strategy='median')
df_imputed[methyl_cols] = imputer.fit_transform(df[methyl_cols])

# Sprawdzenie, czy braki zostały uzupełnione
print("\nBrakujące wartości po imputacji medianą:")
print(df_imputed[methyl_cols].isnull().sum().sum())  # powinno być 0
# %% [markdown]
# ## Etap 3: Analiza i wizualizacja
# %%
# Statystyki opisowe
print("\nStatystyki opisowe (wybrane kolumny liczbowe):")
print(df.describe())
# display(df.describe())
# %%

# Histogramy wybranych zmiennych
numeric_candidates = ["Age"] + methyl_cols
numeric_cols = [c for c in numeric_candidates if c in df.columns]

for col in numeric_cols:
    plt.figure()
    df[col].hist(bins=30)
    plt.title(f"Histogram: {col}")
    plt.xlabel(col)
    plt.ylabel("Liczność")
    plt.show()
# %%

# Wybór kolumn metylacji (CpG) do analizy zależności
sample_cpg = methyl_cols

# Wizualizacja zależności między wiekiem a wybranymi wartościami CpG, z podziałem na palaczy/niepalących
plt.figure(figsize=(12, 4))
for i, cpg in enumerate(sample_cpg[:3]):
    if cpg in df_imputed.columns:
        plt.subplot(1, 3, i+1)
        sns.scatterplot(data=df_imputed, x='Age', y=cpg, hue='Smoking Status', alpha=0.6)
        plt.title(f'{cpg} vs wiek')
plt.tight_layout()
plt.show()

# %%
# Wybór zestawów kolumn do wizualizacji (boxplot, macierz korelacji)
corr_cols = []
col = []

for i in range(len(methyl_cols)):
    if i % 5 == 0:
        col = ["Age"]
    # print(i % 5)
    col.append(methyl_cols[i])
    if i % 5 == 4:
        corr_cols.append(col)

# %%
# Boxplot (wiek)
plt.figure()
sns.boxplot(data=df["Age"], orient="h")
plt.title("Wykres pudełkowy (wiek)")
plt.show()
# %%
# Boxplot i macierz korelacji dla każdego zestawu kolumn
for cols in corr_cols:

    # Boxplot
    if cols:
        boxplot_cols = cols.copy()
        boxplot_cols.remove("Age")

        plt.figure()
        sns.boxplot(data=df[boxplot_cols], orient="h")
        plt.title("Wykres pudełkowy (wybrane wartości metylacji)")
        plt.show()

    # Macierz korelacji
    if len(cols) >= 2:
        corr = df[cols].corr(numeric_only=True)
        plt.figure()
        sns.heatmap(corr, annot=True, fmt=".2f", square=True, cmap="coolwarm")
        plt.title("Macierz korelacji (wiek i wybrane wartości metylacji)")
        plt.show()

# %%

# Zależność: wiek vs. wartości metylacji
for col in methyl_cols:
    if {"Age", col}.issubset(df.columns):
        plt.figure()
        sns.scatterplot(data=df, x="Age", y=col)
        plt.title(f"Wiek vs. {col}")
        plt.show()

# %%
# ------------------------------
# 3. Wizualizacja częstości występowania kategorii w populacji
# ------------------------------

# Rozkład statusu palenia
plt.figure(figsize=(10, 4))
df['Smoking Status'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Liczebność palaczy vs niepalących')
plt.ylabel('Liczba osób')
plt.xticks(rotation=0)
plt.show()
# %%
# Normalizacja wartości w kolumnie "Gender"
display(df["Gender"].value_counts())

df["Gender"] = df["Gender"].str.lower()

display(df["Gender"].value_counts())

# %%

# Rozkład płci w zależności od palenia
pd.crosstab(df['Gender'], df['Smoking Status']).plot(kind='bar', stacked=True)
plt.title('Płeć a status palenia')
plt.xlabel('Płeć')
plt.ylabel('Liczba osób')
plt.legend(title='Status palenia')
plt.show()

# Rozkład wieku w grupach
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x='Age', hue='Smoking Status', kde=True, bins=20)
plt.title('Rozkład wieku w zależności od statusu palenia')
plt.xlabel('Wiek')
plt.ylabel('Liczba osób')
plt.show()
# %% [markdown]
# ## Etap 4: Przygotowanie do modelowania
# %%
# ------------------------------
# 4. Budowa prostego modelu predykcyjnego (regresja logistyczna)
# ------------------------------

# Przygotowanie cech i celu
X = df_imputed[methyl_cols].copy()  # wszystkie CpG jako cechy
y = (df_imputed['Smoking Status'] == 'current').astype(int)  # 1 = palący, 0 = niepalący

# Podział na zbiór treningowy i testowy (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model regresji logistycznej (bez normalizacji)
model_raw = LogisticRegression(max_iter=1000, random_state=42)
model_raw.fit(X_train, y_train)
y_pred_raw = model_raw.predict(X_test)
y_proba_raw = model_raw.predict_proba(X_test)[:, 1]

print("\n--- Wyniki modelu bez normalizacji ---")
print(f"Dokładność (accuracy): {accuracy_score(y_test, y_pred_raw):.3f}")
print(f"AUC ROC: {roc_auc_score(y_test, y_proba_raw):.3f}")
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred_raw, target_names=['Niepalący', 'Palący']))

# Macierz pomyłek
cm = confusion_matrix(y_test, y_pred_raw)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Niepalący', 'Palący'], yticklabels=['Niepalący', 'Palący'])
plt.title('Macierz pomyłek (regresja logistyczna)')
plt.xlabel('Przewidziane')
plt.ylabel('Rzeczywiste')
plt.show()
# %% [markdown]
# ## Etap 5: Porównanie skuteczności metod normalizacji
# %%
# ------------------------------
# 5. Porównanie dwóch metod normalizacji danych
# ------------------------------

# StandardScaler vs MinMaxScaler w potoku z imputacją
# Użycie walidacji krzyżowej 5-krotnej, aby ocenić stabilność.

# Definicja potoków
def make_pipeline(scaler):
    return Pipeline([
        ('scaler', scaler),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])

pipeline_standard = make_pipeline(StandardScaler())
pipeline_minmax = make_pipeline(MinMaxScaler())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_standard = cross_val_score(pipeline_standard, X, y, cv=cv, scoring='accuracy')
scores_minmax = cross_val_score(pipeline_minmax, X, y, cv=cv, scoring='accuracy')

print("\n--- Porównanie normalizacji (5-krotna walidacja krzyżowa) ---")
print(f"StandardScaler: średnia dokładność = {scores_standard.mean():.4f} (+/- {scores_standard.std():.4f})")
print(f"MinMaxScaler:   średnia dokładność = {scores_minmax.mean():.4f} (+/- {scores_minmax.std():.4f})")

# Test statystyczny (t-test dla prób zależnych)
from scipy.stats import ttest_rel
t_stat, p_val = ttest_rel(scores_standard, scores_minmax)
print(f"\nTest t-Studenta dla par: t = {t_stat:.4f}, p = {p_val:.4f}")
if p_val < 0.05:
    print("Różnica między metodami normalizacji jest statystycznie istotna.")
else:
    print("Brak statystycznie istotnej różnicy między metodami normalizacji.")

# Porównanie AUC
scores_standard_auc = cross_val_score(pipeline_standard, X, y, cv=cv, scoring='roc_auc')
scores_minmax_auc = cross_val_score(pipeline_minmax, X, y, cv=cv, scoring='roc_auc')
print(f"\nAUC ROC: StandardScaler = {scores_standard_auc.mean():.4f}, MinMaxScaler = {scores_minmax_auc.mean():.4f}")