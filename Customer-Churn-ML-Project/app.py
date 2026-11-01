import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom styling
st.markdown("""
    <style>
    /* Main Background & Font Styling */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Title Card Styling */
    .title-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .title-card h1 {
        color: white !important;
        margin-bottom: 5px;
        font-weight: 700;
    }
    .title-card p {
        color: #e0e6ed;
        margin: 0;
        font-size: 1.05rem;
    }

    /* Container Box Styling */
    .css-1r6slb0, .stForm {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Button Styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 12px 20px;
        border-radius: 8px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #1e40af 100%);
        box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
        transform: translateY(-1px);
    }
    
    /* Metric Card Custom Style */
    .metric-card {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #2563eb;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 1. Load trained artifacts safely
@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, scaler, model_columns

try:
    model, scaler, model_columns = load_artifacts()
except Exception as e:
    st.error(f"⚠️ Error loading model files: {e}")
    st.info("Make sure `model.pkl`, `scaler.pkl`, and `model_columns.pkl` are in the project root directory.")
    st.stop()

# Header Section
st.markdown("""
    <div class="title-card">
        <h1>🔮 Telecom Customer Churn Intelligence</h1>
        <p>Predict customer retention risk in real-time with machine learning insights.</p>
    </div>
""", unsafe_allow_html=True)

# 2. Form Input Section using 3 Tabbed/Columns Design
st.subheader("📋 Enter Customer Details")

with st.container():
    col1, col2, col3 = st.columns([1, 1, 1], gap="medium")

    with col1:
        st.markdown("### 👤 Personal Info")
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No (0)", "Yes (1)"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12, help="Number of months customer has stayed with company")

    with col2:
        st.markdown("### 📡 Services Subscribed")
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet_service = st.selectbox("Internet Service Provider", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    with col3:
        st.markdown("### 💳 Account & Billing")
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        
        # Calculate dynamic TotalCharges estimated value if tenure changes
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=1.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(tenure * monthly_charges), step=10.0)

st.markdown("---")

# 3. Prediction & Output
st.markdown("<br>", unsafe_allow_html=True)
if st.button("🚀 Analyze Churn Risk"):
    # Convert inputs to raw data format
    raw_data = {
        'SeniorCitizen': 1 if "1" in senior_citizen else 0,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'is_male': 1 if gender == "Male" else 0
    }

    input_df = pd.DataFrame([raw_data])

    # Convert categoricals to dummies
    input_df = pd.get_dummies(input_df)

    # Reindex against training model columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale numeric features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Model inference
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    churn_prob = probabilities[1]
    stay_prob = probabilities[0]

    st.subheader("📊 Analytical Summary")
    
    res_col1, res_col2 = st.columns([1, 1], gap="large")

    with res_col1:
        if prediction == 1 or prediction == 'Yes':
            st.error("🚨 **High Risk of Churn**")
            st.metric(label="Churn Probability", value=f"{churn_prob:.1%}", delta=f"+{churn_prob:.1%} Risk", delta_color="inverse")
            st.progress(float(churn_prob))
            st.warning("**Recommendation:** Offer a contract extension discount or low-cost upgrade to improve retention.")
        else:
            st.success("✅ **Low Risk / Loyal Customer**")
            st.metric(label="Retention Probability", value=f"{stay_prob:.1%}", delta=f"{stay_prob:.1%} Safe")
            st.progress(float(stay_prob))
            st.info("**Recommendation:** Good status. Consider cross-selling premium streaming add-ons.")

    with res_col2:
        st.markdown("#### Key Customer Overview")
        st.write(f"• **Contract:** {contract}")
        st.write(f"• **Monthly Bill:** ${monthly_charges:.2f}")
        st.write(f"• **Total Revenue:** ${total_charges:.2f}")
        st.write(f"• **Tenure Duration:** {tenure} Months")