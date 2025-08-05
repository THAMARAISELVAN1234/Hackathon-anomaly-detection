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
st.set_page_config(page_title="Anomaly Validator", layout="wide")
st.title("Anomaly Detection using AI")


# API CONFIG
openai.api_key = st.secrets["OPENAI_API_KEY"]


# SNOWFLAKE CONFIG
snowflake_config = {
    "user": st.secrets["SNOWFLAKE_USER"],
    "password": st.secrets["SNOWFLAKE_PASSWORD"],
    "account": st.secrets["SNOWFLAKE_ACCOUNT"]
}

@st.cache_data
def load_training_data():
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
            p.POLICY_ID,
            p.CUSTOMER_ID,
            p.POLICY_TYPE,
            p.REGION,
            p.PREMIUM_AMOUNT,
            pt.TXN_ID AS PREMIUM_TXN_ID,
            pt.AMOUNT AS PREMIUM_TXN_AMOUNT,
            pt.TXN_DATE AS PREMIUM_TXN_DATE,
            c.CLAIM_ID,
            c.CLAIM_TYPE,
            c.CLAIM_STATUS,
            ct.CLAIM_TXN_ID,
            ct.CLAIM_AMOUNT
            FROM POLICY p
            LEFT JOIN PREMIUM_TRANSACTION pt ON p.POLICY_ID = pt.POLICY_ID
            LEFT JOIN CLAIM c ON p.POLICY_ID = c.POLICY_ID
            LEFT JOIN CLAIM_TRANSACTION ct ON c.CLAIM_ID = ct.CLAIM_ID;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def explain_anomaly(row):
    """Generate explanation using OpenAI GPT."""
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

# UI
st.subheader("Step 1: Load Historical Training Data from Snowflake")
if st.button("Training Model"):
    df_train = load_training_data()
    if df_train.empty:
        st.warning("No data found in Snowflake.")
    else:
        st.success("Training Data Loaded")
        #st.dataframe(df_train.head())

        st.session_state["df_train"] = df_train

st.subheader("Step 2: Upload New Validation Data (CSV)")
csv_file = st.file_uploader("Upload validation dataset", type=["csv"])

if csv_file and "df_train" in st.session_state:
    df_train = st.session_state["df_train"]
    df_test = pd.read_csv(csv_file)
    st.success("Validation Data Loaded")
    st.dataframe(df_test.head())

    
    # PREPROCESSING
    numeric_features = ["PREMIUM_AMOUNT", "PREMIUM_TXN_AMOUNT", "CLAIM_AMOUNT"]
    categorical_features = ["POLICY_TYPE", "REGION", "CLAIM_TYPE", "CLAIM_STATUS"]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)
    
    # TRAIN MODEL
    model = COPOD()
    model.fit(X_train)

    
    # VALIDATE NEW DATA
    df_test["Anomaly_Score"] = model.decision_function(X_test)
    df_test["Anomaly_Flag"] = model.predict(X_test)
    anomalies = df_test[df_test["Anomaly_Flag"] == 1]

    st.subheader("Detected Anomalies")
    st.dataframe(anomalies)

    
    # VISUALIZATIONS
    '''st.subheader("Anomaly Visualization")
    if not anomalies.empty:
        fig1 = px.pie(anomalies, names="POLICY_TYPE", title="Anomalies by Policy Type")
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(anomalies, x="REGION", title="Anomalies by Region", color="REGION")
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No anomalies detected.")'''


    # AI EXPLANATION
    if st.toggle("Explain anomalies with AI"):
        for i, row in anomalies.head(5).iterrows():
            st.markdown(f"**Policy ID:** {row['POLICY_ID']} | **Claim ID:** {row.get('CLAIM_ID','N/A')}")
            explanation = explain_anomaly(row)
            st.write(explanation)
            st.markdown("---")

    
    # DOWNLOAD OPTION
    st.download_button(
        "Download Anomalies as CSV",
        anomalies.to_csv(index=False),
        file_name="validation_anomalies.csv",
        mime="text/csv"
    )