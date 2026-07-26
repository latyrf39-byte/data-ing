
import streamlit as st
import pandas as pd
import joblib

# ======================
# CONFIG
# ======================

st.set_page_config(
    page_title="Customer Churn Prediction Platform",
    page_icon="🎯",
    layout="wide"
)

# ======================
# CSS
# ======================

st.markdown("""
<div style="
background: linear-gradient(90deg,#0B3C5D,#1F618D);
padding:20px;
border-radius:15px;
color:white;
margin-top:20px;
margin-bottom:20px;
">
<h1 style="margin:0;">
🎯 Customer Churn Prediction Platform
</h1>
<p style="margin:0;">
MLOps • Machine Learning • Customer Retention Analytics
</p>
</div>
""", unsafe_allow_html=True)

# ======================
# MODEL
# ======================

model = joblib.load("logistic_churn_model.pkl")

# ======================
# HEADER
# ======================

st.markdown(
"""
<div class='title'>
🎯 Customer Churn Prediction Platform
</div>
""",
unsafe_allow_html=True
)

st.markdown(
"""
<div class='subtitle'>
Plateforme de scoring prédictif permettant d'identifier les clients à risque de résiliation.
</div>
""",
unsafe_allow_html=True
)

st.divider()

# ======================
# KPI
# ======================

k1, k2, k3, k4 = st.columns(4)

k1.metric("📞 Clients analysés", "7 043")
k2.metric("🤖 Modèle", "Logistic")
k3.metric("🎯 Accuracy", "74.3%")
k4.metric("📈 ROC-AUC", "83.8%")

st.divider()

# ======================
# SIDEBAR
# ======================

st.sidebar.header("⚙️ Paramètres Client")

tenure = st.sidebar.slider(
    "📅 Ancienneté (mois)",
    0,
    72,
    12
)

monthly = st.sidebar.slider(
    "💰 Charges mensuelles",
    0.0,
    120.0,
    50.0
)

total = st.sidebar.slider(
    "💳 Charges totales",
    0.0,
    10000.0,
    500.0
)

# ======================
# LAYOUT
# ======================

col1, col2 = st.columns([2,1])

with col1:

    st.subheader("📋 Informations du client")

    st.info(f"""
    Ancienneté : {tenure} mois

    Charges mensuelles : {monthly:.2f} $

    Charges totales : {total:.2f} $
    """)

with col2:

    st.subheader("📊 Aperçu")

    st.metric("Ancienneté", f"{tenure} mois")
    st.metric("Mensuel", f"{monthly:.2f}$")
    st.metric("Total", f"{total:.2f}$")

# ======================
# PREDICTION
# ======================

if st.button(
    "🚀 Lancer l'analyse",
    use_container_width=True
):

    X = pd.DataFrame(
        [[0]*len(model.feature_names_in_)],
        columns=model.feature_names_in_
    )

    if "tenure" in X.columns:
        X["tenure"] = tenure

    if "MonthlyCharges" in X.columns:
        X["MonthlyCharges"] = monthly

    if "TotalCharges" in X.columns:
        X["TotalCharges"] = total

    prediction = model.predict(X)[0]

    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[0][1]
    else:
        probability = 0.5

    st.divider()

    st.subheader("🎯 Score de Risque")

    st.progress(float(probability))

    st.metric(
        "Probabilité de Churn",
        f"{probability*100:.1f}%"
    )

    if prediction == 1:

        st.markdown(
        f"""
        <div class='risk-high'>
        <h2>🔴 RISQUE ÉLEVÉ</h2>

        Probabilité de churn :
        <b>{probability*100:.1f}%</b>

        <br><br>

        Recommandations :

        <ul>
        <li>Contacter immédiatement le client</li>
        <li>Proposer une offre promotionnelle</li>
        <li>Mettre en place un plan de fidélisation</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
        )

    else:

        st.markdown(
        f"""
        <div class='risk-low'>
        <h2>🟢 CLIENT FIDÈLE</h2>

        Risque estimé :
        <b>{probability*100:.1f}%</b>

        <br><br>

        Recommandations :

        <ul>
        <li>Maintenir la qualité de service</li>
        <li>Continuer les actions de fidélisation</li>
        <li>Suivi standard</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
        )

st.divider()

st.caption(
"Projet MLOps - Telco Customer Churn | MLflow • Streamlit • FastAPI • SHAP"
)
