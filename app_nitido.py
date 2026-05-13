"""
App NÍTIDO — Sistema de Pre-filtrado de Candidatos
Universidad Externado de Colombia | Reto Examen 3
Ejecutar con: streamlit run app_nitido.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pickle, json, warnings
warnings.filterwarnings("ignore")

import shap
from sklearn.impute import SimpleImputer

# ── Configuración de página ──────────────────────────────────────
st.set_page_config(
    page_title="NÍTIDO — Evaluador de Candidatos",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS personalizado ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a8b2d8; font-size: 1.1rem; margin: 0.5rem 0 0; }
    .metric-card {
        background: #f8f9fa; border-radius: 10px; padding: 1.5rem;
        border-left: 4px solid #0f3460; margin: 0.5rem 0;
    }
    .approved  { background: #e8f5e9; border-left: 4px solid #4CAF50; border-radius: 10px; padding: 1.5rem; }
    .rejected  { background: #ffebee; border-left: 4px solid #f44336; border-radius: 10px; padding: 1.5rem; }
    .warning   { background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
    .info-box  { background: #e3f2fd; border-left: 4px solid #2196F3; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Carga de artefactos ──────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model   = pickle.load(open("artifacts/modelo_xgb.pkl", "rb"))
    imputer = pickle.load(open("artifacts/imputer.pkl", "rb"))
    meta    = json.load(open("artifacts/model_meta.json"))
    return model, imputer, meta

try:
    model, imputer, meta = load_artifacts()
    FEATURES       = meta["features"]
    BEST_THRESHOLD = meta["best_threshold"]
    AUC_TEST       = meta["auc_test"]
    TOP3           = meta["top3_features"]
    artifacts_ok   = True
except Exception as e:
    st.error(f"⚠️ No se encontraron los artefactos del modelo. Ejecuta primero `nitido_notebook.py`.\n\nError: {e}")
    artifacts_ok = False
    st.stop()

# ── Header ───────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🎯 NÍTIDO</h1>
    <p>Sistema de Pre-filtrado Inteligente de Candidatos · Powered by XGBoost + SHAP</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Inputs del candidato ──────────────────────────────
st.sidebar.header("📋 Datos del Candidato")
st.sidebar.markdown("Ingresa los valores de las variables del candidato:")

# Rangos observados en el dataset para guiar al usuario
VAR_RANGES = {
    "x1":  (0.0,  25.0,  4.9,  0.1),
    "x2":  (7.0,  100.0, 64.7, 1.0),
    "x3":  (1,    5,     3,    1),
    "x4":  (0,    8,     4,    1),
    "x5":  (0,    1,     0,    1),
    "x6":  (22,   65,    43,   1),
    "x7":  (1,    6,     3,    1),
    "x8":  (1,    10,    5,    1),
    "x9":  (0,    3,     1,    1),
    "x10": (3.0,  5.0,   4.0,  0.01),
    "x11": (0.1,  100.0, 42.0, 0.1),
    "x12": (0,    11,    5,    1),
    "x13": (0,    3,     1,    1),
    "x14": (0.0,  24.0,  5.0,  0.1),
    "x15": (1.0,  50.0,  18.0, 0.5),
    "x16": (0.0,  100.0, 50.0, 1.0),
    "x17": (0,    1,     0,    1),
    "x18": (0.0,  10.0,  4.0,  1.0),
}

candidate_values = {}
with st.sidebar:
    cols_vars = [FEATURES[i:i+9] for i in range(0, 18, 9)]
    for var in FEATURES:
        mn, mx, dv, step = VAR_RANGES[var]
        is_int = isinstance(mn, int) and isinstance(mx, int)
        if is_int:
            v = st.slider(var, int(mn), int(mx), int(dv), step=int(step))
        else:
            v = st.slider(var, float(mn), float(mx), float(dv), step=float(step))
        candidate_values[var] = v

    st.markdown("---")
    predict_btn = st.button("🚀 Evaluar Candidato", type="primary", use_container_width=True)

# ── Función de predicción ────────────────────────────────────────
def predict_candidate(values_dict):
    row = pd.DataFrame([values_dict])
    row_imp = pd.DataFrame(imputer.transform(row), columns=FEATURES)
    proba = model.predict_proba(row_imp)[0, 1]
    pred  = int(proba >= BEST_THRESHOLD)
    return proba, pred, row_imp

def compute_shap(row_imp):
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row_imp)
    return sv[0], explainer.expected_value

def greedy_counterfactual(row_imp, max_changes=5):
    """Heurística greedy para generar contrafactual accionable."""
    row_arr = row_imp.values.copy()
    current_proba = model.predict_proba(row_arr)[0, 1]
    if current_proba >= BEST_THRESHOLD:
        return None

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(row_imp)[0]
    shap_series = pd.Series(sv, index=FEATURES)
    worst_features = shap_series.sort_values().index.tolist()

    counterfactual = row_arr.copy()
    changes = {}

    for feat in worst_features[:max_changes]:
        feat_idx = FEATURES.index(feat)
        original_val = counterfactual[0, feat_idx]
        for frac in [0.1, 0.25, 0.5, 1.0]:
            delta = max(abs(original_val) * frac, 0.5)
            new_val = original_val + delta
            test_cf = counterfactual.copy()
            test_cf[0, feat_idx] = new_val
            new_proba = model.predict_proba(test_cf)[0, 1]
            if new_proba >= BEST_THRESHOLD:
                counterfactual[0, feat_idx] = new_val
                changes[feat] = {
                    "original": round(float(original_val), 3),
                    "sugerido": round(float(new_val), 3),
                    "delta": round(float(delta), 3),
                    "new_proba": round(float(new_proba), 3)
                }
                break
        if model.predict_proba(counterfactual)[0, 1] >= BEST_THRESHOLD:
            break

    final_proba = model.predict_proba(counterfactual)[0, 1]
    return {
        "prob_original": round(float(current_proba), 4),
        "prob_counterfactual": round(float(final_proba), 4),
        "aprobado": final_proba >= BEST_THRESHOLD,
        "cambios": changes,
    }

# ── Contenido principal ──────────────────────────────────────────
if not predict_btn:
    # Estado inicial — mostrar métricas del modelo
    st.markdown("### 📊 Métricas del Modelo en Producción")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AUC-ROC (test)", f"{AUC_TEST:.4f}", "XGBoost")
    c2.metric("Umbral decisión", f"{BEST_THRESHOLD:.3f}", "Youden J")
    c3.metric("Candidatos históricos", "5,000", "dataset NÍTIDO")
    c4.metric("Variables", "18", "anonimizadas")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>ℹ️ Cómo usar esta aplicación</b><br>
    1. Ajusta los valores del candidato en el panel lateral izquierdo.<br>
    2. Haz clic en <b>Evaluar Candidato</b>.<br>
    3. Verás la predicción, la explicación SHAP y (si es rechazado) un contrafactual accionable.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🏆 Variables más influyentes del modelo")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Top variables por Permutation Importance**")
        for i, v in enumerate(TOP3, 1):
            st.markdown(f"**{i}.** `{v}`")
    with col_b:
        st.markdown("""
        <div class="warning">
        <b>⚠️ Aviso regulatorio</b><br>
        Las variables están anonimizadas. Antes de producción, NÍTIDO debe revelar 
        al Ministerio qué representa cada variable para confirmar que no codifican 
        características protegidas (género, etnia, edad, etc.).
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Predicción ─────────────────────────────────────────────
    with st.spinner("Evaluando candidato..."):
        proba, pred, row_imp = predict_candidate(candidate_values)
        shap_vals, expected_val = compute_shap(row_imp)

    st.markdown("## 📋 Resultado de la Evaluación")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if pred == 1:
            st.markdown(f"""
            <div class="approved">
                <h2 style="color:#2e7d32; margin:0">✅ CANDIDATO APROBADO</h2>
                <p style="color:#388e3c; font-size:1.1rem; margin:0.5rem 0 0">
                Este candidato avanza a la etapa de entrevistas.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="rejected">
                <h2 style="color:#c62828; margin:0">❌ CANDIDATO RECHAZADO</h2>
                <p style="color:#d32f2f; font-size:1.1rem; margin:0.5rem 0 0">
                Este candidato no cumple el umbral para avanzar.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.metric("P(Aprobado)", f"{proba:.1%}")
    with col3:
        st.metric("Umbral", f"{BEST_THRESHOLD:.3f}")

    # Barra de probabilidad
    st.markdown("---")
    fig_bar, ax_bar = plt.subplots(figsize=(10, 1.2))
    ax_bar.barh(["Probabilidad"], [proba], color="#4CAF50" if pred == 1 else "#f44336",
                height=0.5, alpha=0.85)
    ax_bar.barh(["Probabilidad"], [1 - proba], left=[proba],
                color="#e0e0e0", height=0.5, alpha=0.5)
    ax_bar.axvline(BEST_THRESHOLD, color="black", linestyle="--", linewidth=2,
                   label=f"Umbral ({BEST_THRESHOLD:.2f})")
    ax_bar.set_xlim(0, 1)
    ax_bar.set_xlabel("Probabilidad de aprobación")
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.set_title("Probabilidad de aprobación del candidato", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig_bar, use_container_width=True)
    plt.close()

    # ── SHAP local ─────────────────────────────────────────────
    st.markdown("## 🔍 Explicación de la Decisión (SHAP)")
    st.markdown("El gráfico muestra qué variables empujaron la predicción hacia arriba (rojo) o hacia abajo (azul) respecto al promedio del modelo.")

    fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
    shap_explanation = shap.Explanation(
        values=shap_vals,
        base_values=expected_val,
        data=row_imp.values[0],
        feature_names=FEATURES,
    )
    shap.waterfall_plot(shap_explanation, show=False, max_display=15)
    plt.title("SHAP — Contribución de variables para este candidato", fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig_shap, use_container_width=True)
    plt.close()

    # Top factores en texto
    shap_df = pd.DataFrame({"variable": FEATURES, "shap": shap_vals})
    top_pos = shap_df.nlargest(3, "shap")
    top_neg = shap_df.nsmallest(3, "shap")

    c_pos, c_neg = st.columns(2)
    with c_pos:
        st.markdown("**🟢 Factores a favor:**")
        for _, r in top_pos.iterrows():
            st.markdown(f"- `{r['variable']}` = {candidate_values[r['variable']]} → +{r['shap']:.3f}")
    with c_neg:
        st.markdown("**🔴 Factores en contra:**")
        for _, r in top_neg.iterrows():
            st.markdown(f"- `{r['variable']}` = {candidate_values[r['variable']]} → {r['shap']:.3f}")

    # ── Contrafactual (solo si rechazado) ──────────────────────
    if pred == 0:
        st.markdown("---")
        st.markdown("## 💡 ¿Qué tendría que cambiar para ser aprobado?")
        st.markdown("A continuación se muestra el cambio mínimo en las variables del candidato que cambiaría la decisión del modelo.")

        with st.spinner("Calculando contrafactual..."):
            cf = greedy_counterfactual(row_imp)

        if cf and cf["cambios"]:
            if cf["aprobado"]:
                st.success(f"✅ Con los siguientes cambios, la probabilidad pasaría de **{cf['prob_original']:.1%}** a **{cf['prob_counterfactual']:.1%}** (umbral: {BEST_THRESHOLD:.2f})")
            else:
                st.warning(f"⚠️ Se encontraron cambios parciales. La probabilidad mejora de **{cf['prob_original']:.1%}** a **{cf['prob_counterfactual']:.1%}**, pero no supera el umbral. Se necesitan cambios más significativos.")

            # Tabla de cambios
            cf_data = []
            for feat, vals in cf["cambios"].items():
                cf_data.append({
                    "Variable": feat,
                    "Valor actual": vals["original"],
                    "Valor sugerido": vals["sugerido"],
                    "Cambio": f"+{vals['delta']:.3f}",
                })
            cf_df = pd.DataFrame(cf_data)
            st.dataframe(cf_df, use_container_width=True, hide_index=True)

            # Gráfico de barras comparativo
            if len(cf_data) > 0:
                fig_cf, ax_cf = plt.subplots(figsize=(10, max(3, len(cf_data) * 1.5)))
                feats_cf = [d["Variable"] for d in cf_data]
                orig_cf  = [d["Valor actual"] for d in cf_data]
                sug_cf   = [d["Valor sugerido"] for d in cf_data]
                x_cf     = np.arange(len(feats_cf))
                w_cf     = 0.35
                ax_cf.bar(x_cf - w_cf/2, orig_cf, w_cf, label="Actual", color="#f44336", alpha=0.8, edgecolor="white")
                ax_cf.bar(x_cf + w_cf/2, sug_cf,  w_cf, label="Sugerido", color="#4CAF50", alpha=0.8, edgecolor="white")
                ax_cf.set_xticks(x_cf); ax_cf.set_xticklabels(feats_cf, fontweight="bold")
                ax_cf.set_ylabel("Valor de la variable")
                ax_cf.set_title("Contrafactual — Cambios sugeridos para aprobación", fontweight="bold")
                ax_cf.legend()
                plt.tight_layout()
                st.pyplot(fig_cf, use_container_width=True)
                plt.close()
        else:
            st.info("No se encontraron cambios simples que reviertan la decisión. El rechazo es robusto en este candidato.")

    # ── Aviso regulatorio ──────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div class="warning">
    <b>⚠️ Aviso importante — Uso responsable</b><br>
    Este sistema es un <b>apoyo a la decisión</b>, no un reemplazo del criterio humano.
    Toda predicción de rechazo debe ser revisada por un reclutador humano antes de comunicarse al candidato.
    El modelo tiene un AUC-ROC de {:.4f}, lo que implica que aproximadamente el {}% de los casos pueden 
    estar mal clasificados. La decisión final siempre recae en una persona.
    </div>
    """.format(AUC_TEST, round((1 - 0.73) * 100)), unsafe_allow_html=True)
