# %% [markdown]
# # Zadanie 06 - `Implementacja systemów ekspertowych: regułowe systemy wspomagania diagnozy medycznej`
# %%
import pandas as pd
import numpy as np

# ------------------------------
# 1. Wczytanie danych
# ------------------------------
df = pd.read_csv('pacjenci_demo_system_ekspertowy.csv')
print("Dane wczytane. Liczba pacjentów:", len(df))

# %% [markdown]
# ## Etap 1: Klasyczne reguły (logika binarna)
# %% [markdown]
# ### Reprezentacja reguł
# %%
# reguły zdefiniowane zgodnie z przykładowym kodem
rules = [
    {
        "if": lambda p: p["systolic_bp"] >= 140 and p["diastolic_bp"] >= 90,
        "then": "Hypertension",
        "weight": 0.9
    },
    {
        "if": lambda p: 120 <= p["systolic_bp"] < 140 or 80 <= p["diastolic_bp"] < 90,
        "then": "Borderline_hypertension",
        "weight": 0.9
    }
]

# %% [markdown]
# ### Mechanizm wnioskowania
# %%
def expert_system(patient, rules):
    conclusions = {}

    for rule in rules:
        if rule["if"](patient):
            diagnosis = rule["then"]
            weight = rule["weight"]
            conclusions[diagnosis] = max(
                conclusions.get(diagnosis, 0), weight
            )

    return conclusions

# %%
print("dane dla pierwszego pacjenta:")
print(df.iloc[0])

# %% [markdown]
# ### Przykład użycia
# %%
dataset_results = []

for _, patient in df.iterrows():
    print("Patient ID:", patient["patient_id"])

    patient_results = expert_system(patient, rules)

    for diagnosis, confidence in patient_results.items():
        print(f"{diagnosis}: confidence={confidence}")

    dataset_results.append(patient_results)

# %% [markdown]
# ### Reprezentacja reguł (wersja alternatywna)
# %%
def classic_hypertension(patient):
    """Zwraca (czy_nadciśnienie, czy_graniczny)"""
    sbp = patient['systolic_bp']
    dbp = patient['diastolic_bp']
    # Kryterium nadciśnienia wg klasycznej reguły
    hypertension = (sbp >= 140) and (dbp >= 90)
    # Przypadek graniczny: wartości podwyższone, ale nie spełniające obu progów
    borderline = (120 <= sbp < 140 or 80 <= dbp < 90) and not hypertension
    return hypertension, borderline

# Zastosowanie do każdego wiersza
df['HT_classic'], df['borderline_classic'] = zip(*df.apply(classic_hypertension, axis=1))

# Podgląd pacjentów granicznych
borderline_patients = df[df['borderline_classic'] == True]
print("\nLiczba pacjentów granicznych (klasycznie):", len(borderline_patients))
print(borderline_patients[['patient_id', 'systolic_bp', 'diastolic_bp']])

# %% [markdown]
# ## Etap 2: Reguły rozmyte (Fuzzy Mamdani)
# %% [markdown]
# ### Reprezentacja reguł
# %%
# Funkcje przynależności (trapezowe i trójkątne)
def trapmf(x, a, b, c, d):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    # rising edge
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    # top
    idx = (b <= x) & (x <= c)
    y[idx] = 1.0
    # falling edge
    idx = (c < x) & (x < d)
    y[idx] = (d - x[idx]) / (d - c)
    return np.clip(y, 0, 1)

def trimf(x, a, b, c):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=float)
    # rising edge
    idx = (a < x) & (x < b)
    y[idx] = (x[idx] - a) / (b - a)
    # peak
    y[x == b] = 1.0
    # falling edge
    idx = (b < x) & (x < c)
    y[idx] = (c - x[idx]) / (c - b)
    return np.clip(y, 0, 1)

# %% [markdown]
# ### Zbiory rozmyte dla wejść i wyjść
# %%
# Zbiory rozmyte dla SBP
def sbp_low(x):    return trapmf([x], 80, 80, 110, 125)[0]
def sbp_border(x): return trimf([x], 120, 135, 150)[0]
def sbp_high(x):   return trapmf([x], 135, 145, 200, 200)[0]

# Zbiory rozmyte dla DBP
def dbp_low(x):    return trapmf([x], 40, 40, 75, 85)[0]
def dbp_border(x): return trimf([x], 80, 88, 96)[0]
def dbp_high(x):   return trapmf([x], 85, 90, 120, 120)[0]

# Uniwersum dla wyniku (ryzyko 0-100)
RISK = np.linspace(0, 100, 101)
risk_low    = trimf(RISK, 0, 20, 40)
risk_medium = trimf(RISK, 30, 50, 70)
risk_high   = trimf(RISK, 60, 80, 100)

# %% [markdown]
# ### Silnik wnioskowania rozmytego (Mamdani)
# %%
# Silnik wnioskowania rozmytego
def fuzzy_hypertension_risk(patient_data: pd.Series):
    sbp = patient_data['systolic_bp']
    dbp = patient_data['diastolic_bp']

    # Stopnie przynależności wejść
    mu = {
        'sbp_low': sbp_low(sbp),
        'sbp_border': sbp_border(sbp),
        'sbp_high': sbp_high(sbp),
        'dbp_low': dbp_low(dbp),
        'dbp_border': dbp_border(dbp),
        'dbp_high': dbp_high(dbp)
    }

    # Reguły
    # R1: SBP wysokie LUB DBP wysokie -> ryzyko wysokie
    r1 = max(mu['sbp_high'], mu['dbp_high'])
    # R2: SBP graniczne I DBP graniczne -> ryzyko średnie
    r2 = min(mu['sbp_border'], mu['dbp_border'])
    # R3: SBP niskie I DBP niskie -> ryzyko niskie
    r3 = min(mu['sbp_low'], mu['dbp_low'])

    # Implikacja (przycinanie zbiorów wyjściowych)
    out_high   = np.minimum(r1, risk_high)
    out_medium = np.minimum(r2, risk_medium)
    out_low    = np.minimum(r3, risk_low)

    # Agregacja (max)
    aggregated = np.maximum.reduce([out_low, out_medium, out_high])

    # Defuzyfikacja (centroid)
    if aggregated.sum() == 0:
        crisp = 0.0
    else:
        crisp = (RISK * aggregated).sum() / aggregated.sum()

    return crisp, mu, {'r1': r1, 'r2': r2, 'r3': r3}

# %% [markdown]
# ### Użycie wnioskowania rozmytego na zbiorze pacjentów
# %%
# Obliczenie ryzyka rozmytego dla każdego pacjenta
risks = []
for _, row in df.iterrows():
    crisp, _, _ = fuzzy_hypertension_risk(row)
    risks.append(round(crisp, 2))
df['fuzzy_risk'] = risks

# %% [markdown]
# ### Mapowanie wartości ryzyka na etykietę
# %%
# Mapowanie wyniku rozmytego na etykietę
def risk_label(crisp):
    if crisp < 35:
        return 'niskie'
    elif crisp < 65:
        return 'średnie'
    else:
        return 'wysokie'

df['fuzzy_label'] = df['fuzzy_risk'].apply(risk_label)

# %%
df

# %%
borderline_patients

# %% [markdown]
# ## Etap 3: Wyjaśnialność (dlaczego ryzyko nie zostało zaklasyfikowane jako wysokie?)
# %% [markdown]
# ### Wybór pacjenta granicznego (nie spełnia klasycznych progów)
# %%
idx = 0
hypertension_classic = True
hypertension_fuzzy = True

example_patient: pd.Series

while hypertension_classic or hypertension_fuzzy:
    example_patient = borderline_patients.iloc[idx]
    print("\n=== Przykład pacjenta granicznego ===")
    print(f"ID: {example_patient['patient_id']}, SBP={example_patient['systolic_bp']}, DBP={example_patient['diastolic_bp']}")

    # Ponowne zastosowanie reguł rozmytych z zapisem szczegółów
    crisp, mu, rules_strength = fuzzy_hypertension_risk(example_patient)
    print(f"Rozmyte ryzyko: {crisp:.2f}/100 -> {risk_label(crisp)}")

    hypertension_classic = example_patient["HT_classic"] == True
    hypertension_fuzzy = risk_label(crisp) == "wysokie"

    print("Czy ryzkyko pacjenta zostało zaklasyfikowane jako wysokie?")
    print(f"Metoda klasyczna: {hypertension_classic}")
    print(f"Metoda rozmyta: {hypertension_fuzzy}")

    if hypertension_classic or hypertension_fuzzy:
        idx += 1
        example_patient = borderline_patients.iloc[idx]

# %% [markdown]
# ### Wyjaśnialny system wnioskowania (explainable expert system)
# %%
def explainable_fuzzy_hypertension_risk(patient_data, explain_top_k=10):
    sbp = float(patient_data["systolic_bp"])
    dbp = float(patient_data["diastolic_bp"])

    # memberships
    mu = {
        "SBP_low": sbp_low(sbp),
        "SBP_border": sbp_border(sbp),
        "SBP_high": sbp_high(sbp),
        "DBP_low": dbp_low(dbp),
        "DBP_border": dbp_border(dbp),
        "DBP_high": dbp_high(dbp),
    }

    # helper ops
    AND = min
    OR = max

    # --------- Define rules with human-readable templates ----------
    rules = [
        {
            "id": "R1",
            "text": "Jeśli SBP jest WYSOKIE LUB DBP jest WYSOKIE, to ryzyko jest WYSOKIE.",
            "strength": OR(mu["SBP_high"], mu["DBP_high"]),
            "conclusion": "risk_high",
            "out_set": risk_high,
            "why": {
                "SBP_high": mu["SBP_high"],
                "DBP_high": mu["DBP_high"],
                "operator": "OR"
            }
        },
        {
            "id": "R2",
            "text": "Jeśli SBP jest GRANICZNE I DBP jest GRANICZNE, to ryzyko jest ŚREDNIE.",
            "strength": AND(mu["SBP_border"], mu["DBP_border"]),
            "conclusion": "risk_medium",
            "out_set": risk_medium,
            "why": {
                "SBP_border": mu["SBP_border"],
                "DBP_border": mu["DBP_border"],
                "operator": "AND"
            }
        },
        {
            "id": "R3",
            "text": "Jeśli SBP jest NISKIE I DBP jest NISKIE, to ryzyko jest NISKIE.",
            "strength": AND(mu["SBP_low"], mu["DBP_low"]),
            "conclusion": "risk_low",
            "out_set": risk_low,
            "why": {
                "SBP_low": mu["SBP_low"],
                "DBP_low": mu["DBP_low"],
                "operator": "AND"
            }
        },
    ]

    # --------- Implication (clipping) + aggregation ----------
    clipped = []
    for r in rules:
        clipped_set = np.minimum(r["strength"], r["out_set"])
        clipped.append(clipped_set)
        r["clipped_area"] = float(clipped_set.sum())  # proxy "how much it contributed"

    aggregated = np.maximum.reduce(clipped) if clipped else np.zeros_like(RISK)

    # Defuzzification (centroid)
    if aggregated.sum() == 0:
        crisp = 0.0
    else:
        crisp = float((RISK * aggregated).sum() / aggregated.sum())

    # Additional interpretable metrics:
    # how much each rule contributed in area terms (normalized)
    total_area = sum(r["clipped_area"] for r in rules) + 1e-12
    for r in rules:
        r["contribution_pct"] = 100.0 * r["clipped_area"] / total_area

    # Sort rules by strength, then by contribution
    rules_sorted = sorted(rules, key=lambda r: (r["strength"], r["clipped_area"]), reverse=True)[:explain_top_k]

    explanation = {
        "patient": {"systolic_bp": sbp, "diastolic_bp": dbp},
        "memberships": mu,
        "rules_fired": rules_sorted,
        "risk_crisp": crisp,
        "risk_label": risk_label(crisp),
    }
    return explanation

# %%
def print_explanation(exp, decimals=3):
    print("=== Explainable Fuzzy Expert System (HT Risk) ===")
    p = exp["patient"]
    print(f"Pacjent: SBP={p['systolic_bp']:.0f}, DBP={p['diastolic_bp']:.0f}")
    print(f"Wynik ryzyka: {exp['risk_crisp']:.2f} / 100  ->  etykieta: {exp['risk_label']}")
    print("\nStopnie przynależności (0..1):")
    for k, v in exp["memberships"].items():
        print(f"  - {k}: {v:.{decimals}f}")

    print("\nNajbardziej aktywne reguły (dlaczego?):")
    for r in exp["rules_fired"]:
        print(f"\n[{r['id']}] {r['text']}")
        print(f"  Siła reguły (strength): {r['strength']:.{decimals}f}")
        print(f"  Udział w agregacji (proxy): {r['contribution_pct']:.1f}%")
        # show WHY details
        why = r["why"]
        if r["id"] == "R1":
            print(f"  Ponieważ: SBP_high={why['SBP_high']:.{decimals}f}, DBP_high={why['DBP_high']:.{decimals}f} (OR)")
        elif r["id"] == "R2":
            print(f"  Ponieważ: SBP_border={why['SBP_border']:.{decimals}f}, DBP_border={why['DBP_border']:.{decimals}f} (AND)")
        elif r["id"] == "R3":
            print(f"  Ponieważ: SBP_low={why['SBP_low']:.{decimals}f}, DBP_low={why['DBP_low']:.{decimals}f} (AND)")

# %% [markdown]
# ### Użycie na przykładzie + wyjaśnienie ("dlaczego?")
# %%
if not example_patient.empty:
    print("\n=== Przykład pacjenta granicznego ===")
    print(f"ID: {example_patient['patient_id']}, SBP={example_patient['systolic_bp']}, DBP={example_patient['diastolic_bp']}")

    # Zastosowanie reguł rozmytych z wnioskowaniem
    exp = explainable_fuzzy_hypertension_risk(example_patient)

    # Pokazanie wnioskowania
    print_explanation(exp)

# %% [markdown]
# ### Pokazanie wyjaśnienia (dlaczego ryzyko nie wysokie?)
# %%
def show_not_high_risk_explanation(patient_data: pd.Series):
    # Ponowne zastosowanie reguł rozmytych z zapisem szczegółów
    crisp, mu, rules_strength = fuzzy_hypertension_risk(patient_data)
    print(f"Rozmyte ryzyko: {crisp:.2f}/100 -> {risk_label(crisp)}")

    # Wyjaśnienie "dlaczego ryzyko nie wysokie"
    print("\n--- Wyjaśnienie ---")
    print("Stopnie przynależności:")
    for k, v in mu.items():
        print(f"  {k}: {v:.3f}")
    print("Siły aktywacji reguł:")
    for k, v in rules_strength.items():
        print(f"  {k}: {v:.3f}")

    if rules_strength['r1'] < 0.5:
        print("-> Reguła prowadząca do wysokiego ryzyka (R1) jest słabo aktywowana.")
    if mu['sbp_high'] < 0.5 and mu['dbp_high'] < 0.5:
        print("-> Ani SBP, ani DBP nie są w wystarczającym stopniu wysokie.")
    if rules_strength['r2'] > rules_strength['r1']:
        print("-> Dominuje reguła ryzyka średniego (R2).")

    # Dodatkowo: porównanie z klasyczną decyzją
    print(f"\nKlasyczna diagnoza: nadciśnienie = {patient_data['HT_classic']}, graniczny = {patient_data['borderline_classic']}")

# %%
if not example_patient.empty:
    print("\n=== Wyjaśnienie dla przykładowego pacjenta granicznego dlaczego ryzyko nie wysokie ===")
    print(f"ID: {example_patient['patient_id']}, SBP={example_patient['systolic_bp']}, DBP={example_patient['diastolic_bp']}")
    show_not_high_risk_explanation(example_patient)

# %% [markdown]
# ### Prezentacja wyników dla wszystkich pacjentów granicznych
# %%
print("\n=== Pacjenci graniczni (klasycznie) i ich ryzyko rozmyte - wyjaśnienie ===")
for _, patient in borderline_patients.iterrows():
    print(f"Patient ID: {patient['patient_id']}")
    exp = explainable_fuzzy_hypertension_risk(patient)
    print_explanation(exp)
    print("=" * 50, "\n")
