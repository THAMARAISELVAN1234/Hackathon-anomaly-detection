import streamlit as st
import pandas as pd
import numpy as np
import snowflake.connector
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from pyod.models.copod import COPOD
import plotly.express as px
import openai

# PAGE CONFIG
st.set_page_config(page_title="Anomaly Detection", layout="wide")
st.title("🔍 Anomaly Detection with AI Explanation")

# API CONFIG
openai.api_key = st.secrets["OPENAI_API_KEY"]

# SNOWFLAKE CONFIG
snowflake_config = {
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "account": st.secrets["SNOWFLAKE_ACCOUNT"],
}

@st.cache_data
def load_data():
    conn = snowflake.connector.connect(
        user=snowflake_config["user"],
        password=snowflake_config["password"],
        account=snowflake_config["account"],
        warehouse="COMPUTE_WH",
        database="INSURANCE_HACKATHON",
        schema="INSURANCE_HACKATHON_SCHEMA"
    )

    query = """
        SELECT 
            p.POLICY_ID, p.CUSTOMER_ID, p.POLICY_TYPE, p.REGION, p.PREMIUM_AMOUNT,
            pt.AMOUNT AS PREMIUM_TXN_AMOUNT,
            c.CLAIM_ID, c.CLAIM_TYPE, c.CLAIM_STATUS,
            ct.CLAIM_AMOUNT
        FROM POLICY p
        LEFT JOIN PREMIUM_TRANSACTION pt ON p.POLICY_ID = pt.POLICY_ID
        LEFT JOIN CLAIM c ON p.POLICY_ID = c.POLICY_ID
        LEFT JOIN CLAIM_TRANSACTION ct ON c.CLAIM_ID = ct.CLAIM_ID
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def explain_anomaly(row):
    """Send anomaly details to OpenAI for explanation."""
    prompt = f"""
    You are an insurance fraud analyst. The following record seems anomalous:
    {row.to_dict()}.
    Explain in simple terms why this might be unusual.
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert insurance analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Explanation error: {e}"

# ==============================
# FETCH DATA BUTTON
# ==============================
if st.button("📥 Fetch Data from Snowflake"):
    df = load_data()
    
    if df.empty:
        st.warning("No data found in Snowflake.")
    else:
        st.subheader("Sample Data")
        st.dataframe(df.head())

        # ==============================
        # PREPROCESSING
        # ==============================
        st.subheader("Running Anomaly Detection...")
        df_clean = df.dropna()

        numeric_features = ["PREMIUM_AMOUNT", "PREMIUM_TXN_AMOUNT", "CLAIM_AMOUNT"]
        categorical_features = ["POLICY_TYPE", "REGION", "CLAIM_TYPE", "CLAIM_STATUS"]

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(), categorical_features)
            ]
        )

        X = preprocessor.fit_transform(df_clean)

        # ==============================
        # ANOMALY DETECTION
        # ==============================
        model = COPOD()
        model.fit(X)
        df_clean["Anomaly_Score"] = model.decision_scores_
        df_clean["Anomaly_Flag"] = model.predict(X)

        anomalies = df_clean[df_clean["Anomaly_Flag"] == 1]
        
        st.success(f"✅ Anomalies Detected: {len(anomalies)}")
        st.dataframe(anomalies)

        # ==============================
        # VISUALIZATIONS
        # ==============================
        st.subheader("📊 Anomaly Visualizations")

        fig1 = px.pie(anomalies, names="POLICY_TYPE", title="Anomalies by Policy Type")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(anomalies, x="REGION", title="Anomalies by Region", color="REGION")
        st.plotly_chart(fig2, use_container_width=True)

        # ==============================
        # AI EXPLANATION
        # ==============================
        st.subheader("🧠 AI Explanations for Anomalies")
        for i, row in anomalies.head(5).iterrows():
            st.markdown(f"**Policy ID:** {row['POLICY_ID']} | **Claim ID:** {row['CLAIM_ID']}")
            explanation = explain_anomaly(row)
            st.write(explanation)
            st.markdown("---")

        # ==============================
        # DOWNLOAD OPTION
        # ==============================
        st.subheader("⬇ Download Results")
        csv = anomalies.to_csv(index=False)
        st.download_button(
            label="Download Anomalies as CSV",
            data=csv,
            file_name="anomalies.csv",
            mime="text/csv"
        )