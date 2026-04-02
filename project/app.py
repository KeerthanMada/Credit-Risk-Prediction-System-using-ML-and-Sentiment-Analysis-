import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib
from textblob import TextBlob

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="🏦",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD MODEL & ENCODERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load('extra_trees_credit_model.pkl')
    encoders = {
        'Sex'             : joblib.load('Sex_encoder.pkl'),
        'Housing'         : joblib.load('Housing_encoder.pkl'),
        'Saving accounts' : joblib.load('Saving accounts_encoder.pkl'),
        'Checking account': joblib.load('Checking account_encoder.pkl'),
        'Purpose'         : joblib.load('Purpose_encoder.pkl'),
    }
    return model, encoders

model, encoders = load_artifacts()

# ─────────────────────────────────────────────
# HELPER – safely extract SHAP values for class 1
# ─────────────────────────────────────────────
def get_shap_class1(sv):
    if isinstance(sv, list):
        return np.array(sv[1])       # binary: [class0, class1]
    arr = np.array(sv)
    if arr.ndim == 3:
        return arr[:, :, 1]          # (samples, features, classes)
    return arr                       # already (samples, features)

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🏦 Credit Risk Prediction App")
st.markdown(
    "Fill in the applicant's details below. The model will predict whether "
    "their credit risk is **Good** or **Bad**, along with confidence scores "
    "and a full SHAP explanation."
)
st.divider()

# ─────────────────────────────────────────────
# INPUT FORM
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Personal Details")
    age     = st.number_input("Age", min_value=18, max_value=80, value=30)
    sex     = st.selectbox("Sex", ["male", "female"])
    job     = st.selectbox(
        "Job Category",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "0 – Unskilled & non-resident",
            1: "1 – Unskilled & resident",
            2: "2 – Skilled",
            3: "3 – Highly skilled"
        }[x],
        index=2
    )
    housing = st.selectbox("Housing", ["own", "rent", "free"])

with col2:
    st.subheader("💳 Financial Details")
    saving_accounts  = st.selectbox("Saving Accounts",  ["little", "moderate", "quite rich", "rich"])
    checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
    credit_amount    = st.number_input("Credit Amount (€)", min_value=0, value=1000, step=100)
    duration         = st.number_input("Loan Duration (months)", min_value=1, max_value=72, value=12)
    purpose          = st.selectbox(
        "Purpose",
        ["car", "furniture/equipment", "radio/TV",
         "education", "business", "vacation/others",
         "repairs", "domestic appliances"]
    )

# ── Loan Remarks ─────────────────────────────────
st.subheader("📝 Loan Officer Remarks  *(for Sentiment Analysis)*")
remarks = st.text_area(
    "Enter any notes about the applicant",
    value="Applicant has stable employment and consistent savings history.",
    height=100,
    help="TextBlob analyses this text and extracts a sentiment score used as a model feature."
)

# Live sentiment preview
polarity     = TextBlob(remarks).sentiment.polarity
subjectivity = TextBlob(remarks).sentiment.subjectivity
label        = "🟢 Positive" if polarity > 0 else ("🔴 Negative" if polarity < 0 else "⚪ Neutral")

s1, s2, s3 = st.columns(3)
s1.metric("Polarity",     f"{polarity:.3f}",     help="-1 = very negative, +1 = very positive")
s2.metric("Subjectivity", f"{subjectivity:.3f}", help="0 = objective, 1 = subjective")
s3.metric("Sentiment",    label)

st.divider()

# ─────────────────────────────────────────────
# PREDICT BUTTON
# ─────────────────────────────────────────────
if st.button("🔍 Predict Credit Risk", use_container_width=True, type="primary"):

    # Build input row
    feature_order = [
        'Age', 'Sex', 'Job', 'Housing',
        'Saving accounts', 'Checking account',
        'Credit amount', 'Duration', 'Purpose',
        'Sentiment_Polarity', 'Sentiment_Subjectivity'
    ]

    input_dict = {
        'Age'                    : age,
        'Sex'                    : encoders['Sex'].transform([sex])[0],
        'Job'                    : job,
        'Housing'                : encoders['Housing'].transform([housing])[0],
        'Saving accounts'        : encoders['Saving accounts'].transform([saving_accounts])[0],
        'Checking account'       : encoders['Checking account'].transform([checking_account])[0],
        'Credit amount'          : credit_amount,
        'Duration'               : duration,
        'Purpose'                : encoders['Purpose'].transform([purpose])[0],
        'Sentiment_Polarity'     : polarity,
        'Sentiment_Subjectivity' : subjectivity,
    }

    input_df = pd.DataFrame([input_dict])[feature_order]

    # Predict
    prediction = model.predict(input_df)[0]
    proba      = model.predict_proba(input_df)[0]
    prob_good  = proba[1]
    prob_bad   = proba[0]

    # ── Result ────────────────────────────────────
    st.subheader("📊 Prediction Result")
    r1, r2, r3 = st.columns(3)
    if prediction == 1:
        r1.success("✅ **GOOD Credit Risk**")
    else:
        r1.error("❌ **BAD Credit Risk**")
    r2.metric("Confidence (Good)", f"{prob_good * 100:.1f}%")
    r3.metric("Confidence (Bad)",  f"{prob_bad  * 100:.1f}%")

    # Probability bar
    fig, ax = plt.subplots(figsize=(7, 1.2))
    ax.barh([0], [prob_good], color='#2ecc71', label=f'Good  {prob_good*100:.1f}%')
    ax.barh([0], [prob_bad],  left=[prob_good], color='#e74c3c', label=f'Bad  {prob_bad*100:.1f}%')
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title('Probability Split', fontsize=10, fontweight='bold')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()

    # ── SHAP Explanation ──────────────────────────
    st.subheader("🧠 Why did the model decide this?  *(SHAP Explanation)*")

    with st.spinner("Computing SHAP values..."):
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)
        sv_class1   = get_shap_class1(shap_values)
        single_shap = np.array(sv_class1[0]).flatten()  # guaranteed 1D

    # SHAP waterfall bar chart (fully manual — no shap plot functions)
    shap_df  = pd.DataFrame({'Feature': feature_order, 'SHAP Value': single_shap})
    shap_df  = shap_df.reindex(shap_df['SHAP Value'].abs().sort_values().index)
    clrs     = ['#e74c3c' if v < 0 else '#2ecc71' for v in shap_df['SHAP Value']]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.barh(shap_df['Feature'], shap_df['SHAP Value'], color=clrs, edgecolor='white')
    ax2.axvline(0, color='black', linewidth=0.8)
    ax2.set_title(
        'Feature Contributions (SHAP)\n🟢 pushes toward GOOD  |  🔴 pushes toward BAD',
        fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel('SHAP Value')
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Plain-English summary
    top_good    = shap_df[shap_df['SHAP Value'] > 0].sort_values('SHAP Value', ascending=False).head(2)
    top_bad     = shap_df[shap_df['SHAP Value'] < 0].sort_values('SHAP Value').head(2)
    lines = []
    if not top_good.empty:
        lines.append("✅ Factors supporting **Good** outcome: " +
                      ", ".join(f"**{f}**" for f in top_good['Feature']))
    if not top_bad.empty:
        lines.append("⚠️ Factors raising concern: " +
                      ", ".join(f"**{f}**" for f in top_bad['Feature']))
    if lines:
        st.info("\n\n".join(lines))

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.caption(
    "Minor Project · Credit Risk Modeling with ML · "
    "Extra Trees Classifier + Sentiment Analysis + SHAP Explainability"
)