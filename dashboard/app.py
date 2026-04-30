import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Smart Factory AI Dashboard", layout="wide")

# Load model
model = joblib.load("models/factory_predictor.pkl")
# Title
st.markdown("""
<h1 style='text-align: center; color: #0EA5E9;'>
Smart Factory Predictive Maintenance Dashboard
</h1>
<p style='text-align: center; font-size:18px;'>
AI-Powered Machine Failure Detection | Risk Monitoring | Maintenance Intelligence
</p>
""", unsafe_allow_html=True)

# File upload
st.sidebar.header("Upload Machine Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(df.head())

    # Encode Type
    type_mapping = {'L': 0, 'M': 1, 'H': 2}
    if 'Type' in df.columns:
        df['Type'] = df['Type'].map(type_mapping)

    # Required columns / Features for prediction
    features = [
        'Type',
        'Air temperature [K]',
        'Process temperature [K]',
        'Rotational speed [rpm]',
        'Torque [Nm]',
        'Tool wear [min]'
    ]

    # Prediction probability
    probabilities = model.predict_proba(df[features])[:, 1]

    # Threshold 0.65
    predictions = (probabilities > 0.65).astype(int)

    df['Failure Risk %'] = probabilities * 100
    df['Prediction'] = predictions

    # Show results
    st.subheader("Prediction Results")
    st.dataframe(df)

    # KPI
    total = len(df)
    failures = df['Prediction'].sum()
    safe = total - failures

    col1, col2, col3, col4 = st.columns(4)

    risk_rate = (failures / total) * 100

    col1.metric("Total Machines", total)
    col2.metric("Predicted Failures", failures)
    col3.metric("Safe Machines", safe)
    col4.metric("Failure Risk Rate (%)", f"{risk_rate:.2f}%")


    def risk_category(prob):
        if prob > 80:
            return "Critical"
        elif prob > 65:
            return "Warning"
        else:
            return "Safe"

    df['Risk Category'] = df['Failure Risk %'].apply(risk_category)


    st.subheader("Machine Risk Analysis")
    display_cols = [
        'Failure Risk %',
        'Risk Category',
        'Prediction'
        ]

    st.dataframe(df[display_cols])

    critical_count = len(df[df['Risk Category'] == "Critical"])
    warning_count = len(df[df['Risk Category'] == "Warning"])
    safe_count = len(df[df['Risk Category'] == "Safe"])

    st.subheader("Risk Category Summary")

    r1, r2, r3 = st.columns(3)

    r1.metric("Critical Machines", critical_count)
    r2.metric("Warning Machines", warning_count)
    r3.metric("Safe Machines", safe_count)

    import plotly.express as px

    fig = px.histogram(
        df,
        x='Failure Risk %',
        nbins=20,
        title="Failure Risk Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

    critical_machines = df[df['Risk Category'] == "Critical"]

    if not critical_machines.empty:
        st.error(f"Critical Alert: {len(critical_machines)} machines need immediate attention!")
        st.subheader("Critical Machine Details")
        st.dataframe(
            critical_machines[
                [
                    'Failure Risk %',
                    'Torque [Nm]',
                    'Rotational speed [rpm]',
                    'Tool wear [min]'
                ]
            ]
        )

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download Full Prediction Report",
        data=csv,
        file_name="smart_factory_predictions.csv",
        mime="text/csv"
    )


    st.markdown("""
    ---
    ### Smart Factory AI System  
    Built for Predictive Maintenance, Downtime Reduction, and Machine Intelligence
    """)