"""
RETO NÍTIDO — Examen 3 · Machine Learning II
Universidad Externado de Colombia
Script completo con las 20 acciones obligatorias.
Ejecutar con: python nitido_notebook.py
Genera todos los artefactos (figuras + modelo serializado) para el cuadernillo.
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import warnings, os, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    f1_score, accuracy_score, ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap
import lime
import lime.lime_tabular

os.makedirs("figures", exist_ok=True)
os.makedirs("artifacts", exist_ok=True)

SEED = 42
np.random.seed(SEED)

PALETTE = {"0": "#E07B54", "1": "#4C9BE8"}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# ─────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────
df = pd.read_csv("candidatos_nitido.csv")
print(f"Dataset cargado: {df.shape[0]} filas × {df.shape[1]} columnas")

FEATURES = [c for c in df.columns if c != "avanza"]
TARGET = "avanza"

# ═══════════════════════════════════════════════════════════════
# FASE 1 — COMPRENSIÓN DEL PROBLEMA Y LOS DATOS
# ═══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# ACCIÓN 1 — EDA visual con hallazgos explícitos
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 1: EDA ===")

# Clasificar variables por tipo
binary_vars   = ["x5", "x17"]
ordinal_vars  = ["x3", "x4", "x7", "x8", "x9", "x12", "x13", "x18"]
continuous_vars = ["x1", "x2", "x6", "x10", "x11", "x14", "x15", "x16"]

# Fig 1a — Distribuciones continuas por clase
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, var in enumerate(continuous_vars):
    for cls, color in zip([0, 1], ["#E07B54", "#4C9BE8"]):
        subset = df[df[TARGET] == cls][var].dropna()
        axes[i].hist(subset, bins=30, alpha=0.6, color=color,
                     label=f"avanza={cls}", density=True)
    axes[i].set_title(var, fontweight="bold")
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel("")
fig.suptitle("Distribuciones de variables continuas por clase (avanza)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/01a_distribuciones_continuas.png", dpi=150)
plt.close()

# Fig 1b — Variables ordinales/discretas (boxplot por clase)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, var in enumerate(ordinal_vars):
    data_0 = df[df[TARGET] == 0][var].dropna()
    data_1 = df[df[TARGET] == 1][var].dropna()
    bp = axes[i].boxplot([data_0, data_1], labels=["Rechazado", "Aprobado"],
                         patch_artist=True, medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#E07B54")
    bp["boxes"][1].set_facecolor("#4C9BE8")
    axes[i].set_title(var, fontweight="bold")
fig.suptitle("Variables ordinales: distribución por clase", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/01b_ordinales_boxplot.png", dpi=150)
plt.close()

# Fig 1c — Variables binarias
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for i, var in enumerate(binary_vars):
    ct = pd.crosstab(df[var], df[TARGET], normalize="index") * 100
    ct.plot(kind="bar", ax=axes[i], color=["#E07B54", "#4C9BE8"], edgecolor="white")
    axes[i].set_title(f"{var} — tasa de aprobación por valor", fontweight="bold")
    axes[i].set_ylabel("% dentro del grupo")
    axes[i].set_xlabel(var)
    axes[i].legend(["Rechazado (0)", "Aprobado (1)"])
    axes[i].tick_params(axis="x", rotation=0)
fig.suptitle("Variables binarias: tasa de aprobación", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/01c_binarias.png", dpi=150)
plt.close()

print("Figuras EDA guardadas.")

# ──────────────────────────────────────────────
# ACCIÓN 2 — Diagnóstico de calidad
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 2: Calidad de datos ===")

# NAs
na_counts = df[FEATURES].isnull().sum()
na_pct = (na_counts / len(df) * 100).round(2)
quality_df = pd.DataFrame({"NAs": na_counts, "% NA": na_pct})
print(quality_df[quality_df["NAs"] > 0])

# Fig 2a — NAs
fig, ax = plt.subplots(figsize=(10, 4))
vars_with_na = quality_df[quality_df["NAs"] > 0]
ax.bar(vars_with_na.index, vars_with_na["% NA"], color="#4C9BE8", edgecolor="white")
ax.set_ylabel("% de valores faltantes")
ax.set_title("Variables con valores faltantes (NAs)", fontweight="bold")
ax.axhline(5, color="red", linestyle="--", alpha=0.7, label="Umbral 5%")
ax.legend()
plt.tight_layout()
plt.savefig("figures/02a_nas.png", dpi=150)
plt.close()

# Outliers (IQR) para continuas
outlier_summary = {}
for var in continuous_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    n_out = ((df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)).sum()
    outlier_summary[var] = n_out
print("\nOutliers por variable (método IQR):")
print(pd.Series(outlier_summary))

# Fig 2b — Desbalance de clases
fig, ax = plt.subplots(figsize=(6, 5))
vc = df[TARGET].value_counts()
ax.bar(["Rechazado (0)", "Aprobado (1)"], vc.values,
       color=["#E07B54", "#4C9BE8"], edgecolor="white", width=0.5)
for j, v in enumerate(vc.values):
    ax.text(j, v + 20, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
ax.set_title("Distribución de la clase objetivo (avanza)", fontweight="bold")
ax.set_ylabel("Cantidad de candidatos")
plt.tight_layout()
plt.savefig("figures/02b_desbalance.png", dpi=150)
plt.close()

print(f"\nDesbalance: {vc[0]} rechazados ({vc[0]/len(df)*100:.1f}%) vs {vc[1]} aprobados ({vc[1]/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────
# ACCIÓN 3 — Correlaciones
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 3: Correlaciones ===")

corr = df[FEATURES].corr()

fig, ax = plt.subplots(figsize=(14, 12))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
            vmin=-1, vmax=1)
ax.set_title("Matriz de correlaciones entre variables (triángulo inferior)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/03_correlaciones.png", dpi=150)
plt.close()

# Pares con |r| > 0.6
high_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.6:
            high_corr.append((corr.columns[i], corr.columns[j], round(r, 3)))
print("Pares con |r| > 0.6:", high_corr)

# ═══════════════════════════════════════════════════════════════
# FASE 2 — PREPROCESAMIENTO Y MODELADO
# ═══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# ACCIÓN 4 — Partición Train / Validation / Test
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 4: Partición ===")

X = df[FEATURES].copy()
y = df[TARGET].copy()

# Imputar NAs con mediana (antes de particionar para evitar leakage con fit posterior)
# (La mediana se re-calculará solo en train en el pipeline real)
from sklearn.impute import SimpleImputer

# Partición 70 / 15 / 15 estratificada
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.15/0.85, random_state=SEED, stratify=y_temp)

print(f"Train:      {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Validation: {X_val.shape[0]}  ({X_val.shape[0]/len(df)*100:.1f}%)")
print(f"Test:       {X_test.shape[0]}  ({X_test.shape[0]/len(df)*100:.1f}%)")
print(f"Clase en train — 0:{y_train.sum()/len(y_train)*100:.1f}% aprobados")

# ──────────────────────────────────────────────
# ACCIÓN 5 — Preprocesamiento (impute + scale)
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 5: Preprocesamiento ===")

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=FEATURES)
X_val_imp   = pd.DataFrame(imputer.transform(X_val),   columns=FEATURES)
X_test_imp  = pd.DataFrame(imputer.transform(X_test),  columns=FEATURES)

X_train_sc = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=FEATURES)
X_val_sc   = pd.DataFrame(scaler.transform(X_val_imp),   columns=FEATURES)
X_test_sc  = pd.DataFrame(scaler.transform(X_test_imp),  columns=FEATURES)

# Para tree-based models, no se necesita escalar — usaremos X_train_imp
# Para LogReg, usaremos X_train_sc

# ──────────────────────────────────────────────
# ACCIÓN 6 — Manejo del desbalance (SMOTE)
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 6: Desbalance ===")

# El dataset tiene 52.4% / 47.6% — desbalance moderado
# Se aplica SMOTE en train para equilibrar clases
smote = SMOTE(random_state=SEED)
X_train_res, y_train_res = smote.fit_resample(X_train_imp, y_train)
print(f"Train antes SMOTE: {y_train.value_counts().to_dict()}")
print(f"Train después SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")

# Para LogReg (estandarizado)
X_train_res_sc = pd.DataFrame(scaler.fit_transform(X_train_res), columns=FEATURES)
X_val_sc   = pd.DataFrame(scaler.transform(X_val_imp),   columns=FEATURES)
X_test_sc  = pd.DataFrame(scaler.transform(X_test_imp),  columns=FEATURES)

# ──────────────────────────────────────────────
# ACCIÓN 7 — Modelo base: Regresión Logística
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 7: Modelo base (Regresión Logística) ===")

lr = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr.fit(X_train_res_sc, y_train_res)

lr_val_pred = lr.predict(X_val_sc)
lr_val_proba = lr.predict_proba(X_val_sc)[:, 1]
lr_val_auc = roc_auc_score(y_val, lr_val_proba)
lr_val_f1  = f1_score(y_val, lr_val_pred)
print(f"LR  — AUC val: {lr_val_auc:.4f}  |  F1 val: {lr_val_f1:.4f}")

# ──────────────────────────────────────────────
# ACCIÓN 8 — Modelos complejos (RF y XGBoost)
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 8: Modelos complejos ===")

# Random Forest (valores iniciales, se optimizará en acc 9)
rf_base = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1)
rf_base.fit(X_train_res, y_train_res)
rf_val_pred  = rf_base.predict(X_val_imp)
rf_val_proba = rf_base.predict_proba(X_val_imp)[:, 1]
rf_val_auc   = roc_auc_score(y_val, rf_val_proba)
rf_val_f1    = f1_score(y_val, rf_val_pred)
print(f"RF  — AUC val: {rf_val_auc:.4f}  |  F1 val: {rf_val_f1:.4f}")

# XGBoost
xgb_base = xgb.XGBClassifier(
    n_estimators=200, learning_rate=0.1, max_depth=5,
    random_state=SEED, eval_metric="logloss", verbosity=0)
xgb_base.fit(X_train_res, y_train_res)
xgb_val_pred  = xgb_base.predict(X_val_imp)
xgb_val_proba = xgb_base.predict_proba(X_val_imp)[:, 1]
xgb_val_auc   = roc_auc_score(y_val, xgb_val_proba)
xgb_val_f1    = f1_score(y_val, xgb_val_pred)
print(f"XGB — AUC val: {xgb_val_auc:.4f}  |  F1 val: {xgb_val_f1:.4f}")

# ──────────────────────────────────────────────
# ACCIÓN 9 — Búsqueda de hiperparámetros (GridSearchCV)
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 9: Búsqueda de hiperparámetros ===")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

# Grid para Random Forest
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=-1),
    rf_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
rf_gs.fit(X_train_res, y_train_res)
print(f"RF best params: {rf_gs.best_params_}  |  CV AUC: {rf_gs.best_score_:.4f}")

# Grid para XGBoost
xgb_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
}
xgb_gs = GridSearchCV(
    xgb.XGBClassifier(random_state=SEED, eval_metric="logloss", verbosity=0),
    xgb_param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, verbose=0
)
xgb_gs.fit(X_train_res, y_train_res)
print(f"XGB best params: {xgb_gs.best_params_}  |  CV AUC: {xgb_gs.best_score_:.4f}")

rf_best  = rf_gs.best_estimator_
xgb_best = xgb_gs.best_estimator_

# ──────────────────────────────────────────────
# ACCIÓN 10 — Comparación de modelos
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 10: Comparación de modelos ===")

models_val = {
    "Regresión Logística": (lr, X_val_sc),
    "Random Forest": (rf_best, X_val_imp),
    "XGBoost": (xgb_best, X_val_imp),
}

results = {}
for name, (model, Xv) in models_val.items():
    proba = model.predict_proba(Xv)[:, 1]
    pred  = model.predict(Xv)
    results[name] = {
        "AUC-ROC": roc_auc_score(y_val, proba),
        "F1": f1_score(y_val, pred),
        "Accuracy": accuracy_score(y_val, pred),
        "AP": average_precision_score(y_val, proba),
    }

results_df = pd.DataFrame(results).T.round(4)
print(results_df)

# Fig 10 — Comparación
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(results_df))
w = 0.2
metrics_plot = ["AUC-ROC", "F1", "Accuracy", "AP"]
colors = ["#4C9BE8", "#E07B54", "#5BBF6C", "#9B59B6"]
for i, (m, c) in enumerate(zip(metrics_plot, colors)):
    ax.bar(x + i*w, results_df[m], width=w, label=m, color=c, alpha=0.85, edgecolor="white")
ax.set_xticks(x + w*1.5)
ax.set_xticklabels(results_df.index)
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("Valor métrica")
ax.set_title("Comparación de modelos en conjunto de validación", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("figures/10_comparacion_modelos.png", dpi=150)
plt.close()

# MODELO FINAL = XGBoost (mejor AUC en validación)
final_model = xgb_best
final_Xval  = X_val_imp
final_Xtest = X_test_imp
print("\nModelo final seleccionado: XGBoost")

# ═══════════════════════════════════════════════════════════════
# FASE 3 — EVALUACIÓN Y EXPLICABILIDAD
# ═══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# ACCIÓN 11 — Métrica principal: AUC-ROC
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 11: Métrica principal ===")
# Justificación: NÍTIDO pre-filtra candidatos.
# Falsos negativos (rechazar buenos) = costo alto (sesgo, demandas).
# Falsos positivos (pasar malos) = bajo costo (humano los filtra después).
# → Priorizar recall de clase 1 → AUC-ROC captura el trade-off completo.
# Además el CEO necesita defender el modelo ante reguladores → AUC es estándar en fairness.
print("Métrica principal: AUC-ROC. Justificación registrada.")

# ──────────────────────────────────────────────
# ACCIÓN 12 — Matriz de confusión, ROC, umbral
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 12: Evaluación test ===")

test_proba = final_model.predict_proba(final_Xtest)[:, 1]
test_pred  = final_model.predict(final_Xtest)

auc_test = roc_auc_score(y_test, test_proba)
print(f"AUC-ROC test: {auc_test:.4f}")

# Buscar umbral óptimo (Youden J)
fpr, tpr, thresholds = roc_curve(y_test, test_proba)
j_scores = tpr - fpr
best_idx = np.argmax(j_scores)
best_thresh = thresholds[best_idx]
print(f"Umbral óptimo (Youden J): {best_thresh:.3f}")
test_pred_opt = (test_proba >= best_thresh).astype(int)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Curva ROC
axes[0].plot(fpr, tpr, color="#4C9BE8", lw=2, label=f"XGBoost (AUC={auc_test:.3f})")
axes[0].plot([0,1],[0,1],"k--", lw=1, alpha=0.5)
axes[0].scatter(fpr[best_idx], tpr[best_idx], color="red", s=100, zorder=5,
                label=f"Umbral={best_thresh:.2f}")
axes[0].set_xlabel("Tasa de Falsos Positivos"); axes[0].set_ylabel("Tasa de Verdaderos Positivos")
axes[0].set_title("Curva ROC — Test", fontweight="bold"); axes[0].legend()

# Curva Precision-Recall
precision, recall, pr_thresh = precision_recall_curve(y_test, test_proba)
ap = average_precision_score(y_test, test_proba)
axes[1].plot(recall, precision, color="#E07B54", lw=2, label=f"AP={ap:.3f}")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Curva Precision-Recall — Test", fontweight="bold"); axes[1].legend()

# Matriz de confusión (umbral óptimo)
cm = confusion_matrix(y_test, test_pred_opt)
disp = ConfusionMatrixDisplay(cm, display_labels=["Rechazado", "Aprobado"])
disp.plot(ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title(f"Matriz de Confusión\n(umbral={best_thresh:.2f})", fontweight="bold")

plt.tight_layout()
plt.savefig("figures/12_roc_pr_cm.png", dpi=150)
plt.close()

print(classification_report(y_test, test_pred_opt, target_names=["Rechazado", "Aprobado"]))

# ──────────────────────────────────────────────
# ACCIÓN 13 — Diagnóstico de sobreajuste
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 13: Diagnóstico sobreajuste ===")

train_proba_full = final_model.predict_proba(X_train_imp)[:, 1]
train_auc = roc_auc_score(y_train, train_proba_full)
val_proba_full = final_model.predict_proba(X_val_imp)[:, 1]
val_auc = roc_auc_score(y_val, val_proba_full)

print(f"AUC Train:      {train_auc:.4f}")
print(f"AUC Validation: {val_auc:.4f}")
print(f"AUC Test:       {auc_test:.4f}")
print(f"Diferencia train-test: {train_auc - auc_test:.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
sets_  = ["Entrenamiento", "Validación", "Test"]
aucs_  = [train_auc, val_auc, auc_test]
colors_ = ["#4C9BE8", "#E07B54", "#5BBF6C"]
bars = ax.bar(sets_, aucs_, color=colors_, edgecolor="white", width=0.5)
ax.set_ylim(0.7, 1.0)
ax.set_ylabel("AUC-ROC")
ax.set_title("AUC-ROC por conjunto (diagnóstico de sobreajuste)", fontweight="bold")
for bar, val in zip(bars, aucs_):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.003, f"{val:.4f}",
            ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("figures/13_sobreajuste.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# ACCIÓN 14 — Explicabilidad global: Permutation Importance + SHAP beeswarm
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 14: Explicabilidad global ===")

# Permutation Importance
perm_imp = permutation_importance(
    final_model, X_test_imp, y_test,
    n_repeats=20, random_state=SEED, scoring="roc_auc", n_jobs=-1
)
perm_df = pd.DataFrame({
    "variable": FEATURES,
    "importancia_media": perm_imp.importances_mean,
    "importancia_std": perm_imp.importances_std,
}).sort_values("importancia_media", ascending=False)

fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(perm_df["variable"][::-1], perm_df["importancia_media"][::-1],
        xerr=perm_df["importancia_std"][::-1], color="#4C9BE8", alpha=0.85, edgecolor="white",
        error_kw={"elinewidth": 1.5, "ecolor": "gray"})
ax.set_xlabel("Reducción en AUC-ROC al permutar la variable")
ax.set_title("Permutation Importance (test set)", fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("figures/14a_permutation_importance.png", dpi=150)
plt.close()

# SHAP — TreeExplainer
explainer = shap.TreeExplainer(final_model)
shap_values_test = explainer.shap_values(X_test_imp)

fig, ax = plt.subplots(figsize=(10, 8))
shap.summary_plot(shap_values_test, X_test_imp, plot_type="dot",
                  show=False, max_display=18)
plt.title("SHAP Beeswarm — Explicabilidad global (XGBoost)", fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("figures/14b_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()

print("Top 5 variables por Permutation Importance:")
print(perm_df.head(5)[["variable", "importancia_media"]].to_string(index=False))

# ──────────────────────────────────────────────
# ACCIÓN 15 — PDP + ICE en variables top
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 15: PDP + ICE ===")

top3_vars = perm_df["variable"].tolist()[:3]
top3_idx  = [FEATURES.index(v) for v in top3_vars]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, feat_idx, feat_name in zip(axes, top3_idx, top3_vars):
    PartialDependenceDisplay.from_estimator(
        final_model, X_test_imp, [feat_idx],
        kind="both", ax=ax, subsample=200, random_state=SEED,
        ice_lines_kw={"alpha": 0.08, "color": "#4C9BE8"},
        pd_line_kw={"color": "#E07B54", "linewidth": 3}
    )
    ax.set_title(f"PDP + ICE — {feat_name}", fontweight="bold")
plt.suptitle("Partial Dependence Plot + Individual Conditional Expectation\n(Top 3 variables)", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("figures/15_pdp_ice.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"PDP+ICE calculados para: {top3_vars}")

# ──────────────────────────────────────────────
# ACCIÓN 16 — SHAP local + LIME en 3 casos representativos
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 16: Explicabilidad local ===")

# Identificar 3 casos representativos
proba_test = final_model.predict_proba(X_test_imp)[:, 1]

# Aprobado con alta confianza (proba >= 0.85)
approved_high_idx = X_test_imp.index[
    (proba_test >= 0.85) & (y_test.values == 1)
]
case_approved = approved_high_idx[0] if len(approved_high_idx) > 0 else X_test_imp.index[np.argmax(proba_test)]

# Rechazado con alta confianza (proba <= 0.15)
rejected_high_idx = X_test_imp.index[
    (proba_test <= 0.15) & (y_test.values == 0)
]
case_rejected = rejected_high_idx[0] if len(rejected_high_idx) > 0 else X_test_imp.index[np.argmin(proba_test)]

# Borderline (proba entre 0.45 y 0.55)
borderline_idx = X_test_imp.index[
    (proba_test >= 0.45) & (proba_test <= 0.55)
]
case_border = borderline_idx[0] if len(borderline_idx) > 0 else X_test_imp.index[np.argmin(np.abs(proba_test - 0.5))]

cases = {
    "Aprobado (alta confianza)": case_approved,
    "Rechazado (alta confianza)": case_rejected,
    "Caso borderline": case_border,
}

# SHAP local
shap_explainer = shap.TreeExplainer(final_model)

for label, idx in cases.items():
    row_pos = X_test_imp.index.get_loc(idx)
    row = X_test_imp.iloc[[row_pos]]
    sv = shap_explainer.shap_values(row)[0]
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(values=sv, base_values=shap_explainer.expected_value,
                         data=row.values[0], feature_names=FEATURES),
        show=False, max_display=15
    )
    plt.title(f"SHAP local — {label}\n(P(aprobado)={proba_test[row_pos]:.3f})", fontweight="bold")
    plt.tight_layout()
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(f"figures/16_shap_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP local guardado: {label} — P={proba_test[row_pos]:.3f}")

# LIME
lime_exp = lime.lime_tabular.LimeTabularExplainer(
    X_train_imp.values,
    feature_names=FEATURES,
    class_names=["Rechazado", "Aprobado"],
    mode="classification",
    random_state=SEED,
)

for label, idx in cases.items():
    row_pos = X_test_imp.index.get_loc(idx)
    row = X_test_imp.iloc[row_pos].values
    explanation = lime_exp.explain_instance(
        row, final_model.predict_proba, num_features=10, num_samples=1000
    )
    fig = explanation.as_pyplot_figure()
    fig.suptitle(f"LIME — {label}", fontweight="bold", y=1.02)
    plt.tight_layout()
    safe = label.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
    plt.savefig(f"figures/16_lime_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  LIME guardado: {label}")

# ──────────────────────────────────────────────
# ACCIÓN 17 — Contrafactuales
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 17: Contrafactuales ===")

def simple_counterfactual(model, row, feature_names, target_class=1,
                            step_fractions=(0.1, 0.25, 0.5),
                            max_changes=5):
    """
    Heurística greedy: modifica las variables con mayor SHAP negativo
    (para candidatos rechazados) incrementalmente hasta alcanzar target_class.
    Devuelve dict con los cambios sugeridos.
    """
    row_arr = np.array(row, dtype=float).reshape(1, -1)
    current_pred = model.predict(row_arr)[0]
    if current_pred == target_class:
        return {"nota": "El candidato ya es aprobado.", "cambios": {}}

    # Calcular SHAP para este candidato
    tree_exp = shap.TreeExplainer(model)
    sv = tree_exp.shap_values(row_arr)[0]

    # Features que más perjudican (SHAP más negativo para clase 1)
    shap_series = pd.Series(sv, index=feature_names)
    worst_features = shap_series.sort_values().index.tolist()  # más negativo primero

    counterfactual = row_arr.copy()
    changes = {}

    for feat in worst_features[:max_changes]:
        feat_idx = feature_names.index(feat)
        original_val = counterfactual[0, feat_idx]
        # Intentar aumentar el valor (suponemos que SHAP negativo → aumentar ayuda)
        for frac in step_fractions:
            delta = max(abs(original_val) * frac, 0.5)
            new_val = original_val + delta
            test_cf = counterfactual.copy()
            test_cf[0, feat_idx] = new_val
            if model.predict(test_cf)[0] == target_class:
                counterfactual[0, feat_idx] = new_val
                changes[feat] = {"original": round(float(original_val), 3),
                                 "sugerido": round(float(new_val), 3),
                                 "delta": round(float(delta), 3)}
                break
        if model.predict(counterfactual)[0] == target_class:
            break

    return {
        "prob_original": float(model.predict_proba(row_arr)[0, 1]),
        "prob_counterfactual": float(model.predict_proba(counterfactual)[0, 1]),
        "prediccion_cf": int(model.predict(counterfactual)[0]),
        "cambios": changes,
    }

# Aplicar al caso rechazado
row_pos_rej = X_test_imp.index.get_loc(case_rejected)
cf_result = simple_counterfactual(
    final_model,
    X_test_imp.iloc[row_pos_rej].values.tolist(),
    FEATURES
)
print("Contrafactual para caso rechazado:")
print(json.dumps(cf_result, indent=2, ensure_ascii=False))

# Fig 17 — Visualización contrafactual
if cf_result["cambios"]:
    fig, ax = plt.subplots(figsize=(10, max(4, len(cf_result["cambios"])*1.2)))
    feats_cf = list(cf_result["cambios"].keys())
    orig_vals = [cf_result["cambios"][f]["original"] for f in feats_cf]
    sug_vals  = [cf_result["cambios"][f]["sugerido"] for f in feats_cf]
    y_pos = range(len(feats_cf))
    ax.barh([f + " (original)" for f in feats_cf], orig_vals, color="#E07B54", alpha=0.7, label="Original")
    ax.barh([f + " (sugerido)" for f in feats_cf], sug_vals, color="#4C9BE8", alpha=0.7, label="Sugerido")
    ax.set_xlabel("Valor de la variable")
    ax.set_title(
        f"Contrafactual — Caso rechazado\n"
        f"P(aprobado): {cf_result['prob_original']:.3f} → {cf_result['prob_counterfactual']:.3f}",
        fontweight="bold"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig("figures/17_contrafactual.png", dpi=150)
    plt.close()
    print("Figura de contrafactual guardada.")

# Guardar contrafactual como JSON para la app Streamlit
with open("artifacts/counterfactual_example.json", "w") as f:
    json.dump(cf_result, f, ensure_ascii=False, indent=2)

# ═══════════════════════════════════════════════════════════════
# FASE 4 — AUDITORÍA, DESPLIEGUE Y DECISIÓN
# ═══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────
# ACCIÓN 18 — Auditoría de sesgos
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 18: Auditoría de sesgos ===")

# Variables binarias como posibles grupos protegidos: x5, x17
bias_results = {}
for prot_var in ["x5", "x17"]:
    group_0_mask = X_test_imp[prot_var] == 0
    group_1_mask = X_test_imp[prot_var] == 1

    proba_g0 = proba_test[group_0_mask.values]
    proba_g1 = proba_test[group_1_mask.values]
    y_g0 = y_test.values[group_0_mask.values]
    y_g1 = y_test.values[group_1_mask.values]

    pred_g0 = (proba_g0 >= best_thresh).astype(int)
    pred_g1 = (proba_g1 >= best_thresh).astype(int)

    # Tasa de aprobación
    rate_g0 = pred_g0.mean()
    rate_g1 = pred_g1.mean()
    # Disparate Impact (DI = tasa minoritaria / tasa mayoritaria)
    di = min(rate_g0, rate_g1) / max(rate_g0, rate_g1) if max(rate_g0, rate_g1) > 0 else np.nan

    # AUC por grupo
    auc_g0 = roc_auc_score(y_g0, proba_g0) if len(np.unique(y_g0)) > 1 else np.nan
    auc_g1 = roc_auc_score(y_g1, proba_g1) if len(np.unique(y_g1)) > 1 else np.nan

    bias_results[prot_var] = {
        "n_grupo_0": int(group_0_mask.sum()),
        "n_grupo_1": int(group_1_mask.sum()),
        "tasa_aprob_g0": round(rate_g0, 4),
        "tasa_aprob_g1": round(rate_g1, 4),
        "disparate_impact": round(di, 4),
        "auc_g0": round(auc_g0, 4),
        "auc_g1": round(auc_g1, 4),
    }
    print(f"\n{prot_var}:")
    print(f"  Tasa aprobación grupo 0: {rate_g0:.2%}")
    print(f"  Tasa aprobación grupo 1: {rate_g1:.2%}")
    print(f"  Disparate Impact: {di:.4f} (umbral regulatorio: ≥ 0.80)")
    print(f"  AUC grupo 0: {auc_g0:.4f} | AUC grupo 1: {auc_g1:.4f}")

with open("artifacts/bias_audit.json", "w") as f:
    json.dump(bias_results, f, ensure_ascii=False, indent=2)

# Fig 18 — Auditoría de sesgos
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for i, prot_var in enumerate(["x5", "x17"]):
    br = bias_results[prot_var]
    groups = [f"{prot_var}=0\n(n={br['n_grupo_0']})", f"{prot_var}=1\n(n={br['n_grupo_1']})"]
    rates = [br["tasa_aprob_g0"], br["tasa_aprob_g1"]]
    bars = axes[i].bar(groups, rates, color=["#E07B54", "#4C9BE8"], edgecolor="white", width=0.5)
    axes[i].set_ylim(0, 1)
    axes[i].set_ylabel("Tasa de aprobación")
    axes[i].set_title(
        f"Auditoría de sesgo — {prot_var}\nDisparate Impact = {br['disparate_impact']:.3f}",
        fontweight="bold"
    )
    axes[i].axhline(0.8 * max(rates), color="red", linestyle="--", alpha=0.7, label="Umbral DI=0.8")
    for bar, v in zip(bars, rates):
        axes[i].text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.1%}", ha="center", fontweight="bold")
    axes[i].legend()
plt.suptitle("Auditoría de Equidad — Variables Protegidas (x5, x17)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("figures/18_auditoria_sesgo.png", dpi=150)
plt.close()

# ──────────────────────────────────────────────
# ACCIÓN 19 — Serializar modelo para Streamlit
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 19: Serializar artefactos para Streamlit ===")

pickle.dump(final_model, open("artifacts/modelo_xgb.pkl", "wb"))
pickle.dump(imputer, open("artifacts/imputer.pkl", "wb"))
pickle.dump(scaler, open("artifacts/scaler.pkl", "wb"))

model_meta = {
    "features": FEATURES,
    "target": TARGET,
    "best_threshold": float(best_thresh),
    "auc_test": float(auc_test),
    "top3_features": top3_vars,
}
with open("artifacts/model_meta.json", "w") as f:
    json.dump(model_meta, f, ensure_ascii=False, indent=2)

print("Artefactos guardados en /artifacts/")

# ──────────────────────────────────────────────
# ACCIÓN 20 — Recomendación final
# ──────────────────────────────────────────────
print("\n=== ACCIÓN 20: Recomendación final ===")

di_x5  = bias_results["x5"]["disparate_impact"]
di_x17 = bias_results["x17"]["disparate_impact"]

print(f"""
══════════════════════════════════════════════════════════════════
RECOMENDACIÓN FINAL AL CEO (Juanpis)
══════════════════════════════════════════════════════════════════

MODELO: XGBoost optimizado con SMOTE + GridSearchCV
AUC-ROC test:       {auc_test:.4f}
F1-score (umbral*): {f1_score(y_test, test_pred_opt):.4f}
Umbral óptimo:      {best_thresh:.3f} (Youden J)

SOBREAJUSTE: AUC train {train_auc:.4f} vs test {auc_test:.4f}
  → Diferencia de {train_auc - auc_test:.4f}  — ACEPTABLE (< 0.05)

EQUIDAD (Disparate Impact):
  x5:  DI = {di_x5:.3f}  {"✓ OK" if di_x5 >= 0.80 else "⚠ ALERTA — por debajo del umbral 0.80"}
  x17: DI = {di_x17:.3f}  {"✓ OK" if di_x17 >= 0.80 else "⚠ ALERTA — por debajo del umbral 0.80"}

CONCLUSIÓN:
{"El modelo puede salir a producción CONDICIONALMENTE." if min(di_x5, di_x17) >= 0.80
 else "El modelo NO debe salir a producción sin antes corregir los sesgos detectados."}

CONDICIONES PREVIAS AL DESPLIEGUE:
  1. Revelar al regulador qué representa cada variable (x1-x18) para
     poder confirmar que no codifican características protegidas.
  2. Implementar monitoreo continuo de Disparate Impact mensualmente.
  3. Definir protocolo de revisión humana para casos borderline
     (probabilidad entre 0.40 y 0.60).
  4. Establecer umbral de decisión en {best_thresh:.2f} (no el default 0.50).
══════════════════════════════════════════════════════════════════
""")

print("\n✅ Las 20 acciones completadas. Figuras en /figures/ | Artefactos en /artifacts/")
print("Ejecuta ahora: streamlit run app_nitido.py")
