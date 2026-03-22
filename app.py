import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from clv import calculate_clv, calculate_revenue_at_risk, calculate_retention_roi, get_clv_tier
from retention import generate_retention_strategies, get_churn_risk_label

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

st.set_page_config(
    page_title="ChurnIQ — Telco Intelligence",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

.stApp { background: #f5f2ed; }
           
.top-bar {
    background: #1a1a2e;
    border-radius: 20px;
    padding: 1.8rem 2.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.5rem;
    margin-top: 1rem;
    
    /* add these two lines */
    opacity: 0;
    animation: fadeIn 0.4s ease 0.1s forwards;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-8px); }
    to   { opacity: 1; transform: translateY(0); }
}
/*----------------------------*/

/*------------------------------*/
          
.top-bar h1 {
    font-size: 2rem;
    font-weight: 800;
    color: #f5f2ed;
    margin: 0;
    letter-spacing: -0.03em;
}

.top-bar .subtitle {
    color: #8892b0;
    font-size: 0.85rem;
    font-family: 'DM Mono', monospace;
    margin-top: 0.3rem;
}

.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    border: 1px solid #e8e4dd;
}

.card-dark {
    background: #1a1a2e;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 50px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.stat-number {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}

.stat-label {
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.3rem;
}

.strategy-card {
    border-left: 4px solid;
    padding: 0.9rem 1.2rem;
    border-radius: 0 12px 12px 0;
    margin-bottom: 0.8rem;
    background: #fafaf8;
}

.strategy-title {
    font-weight: 700;
    font-size: 0.95rem;
    margin-bottom: 0.2rem;
}

.strategy-detail {
    font-size: 0.82rem;
    color: #555;
    margin-bottom: 0.3rem;
}

.strategy-impact {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #22c55e;
}

.tab-header {
    font-size: 0.75rem;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    color: #16161d;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #1a1a2e;
}

.tier-badge {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.05em;
}

[data-testid="stSidebar"] {
    background: #1a1a2e !important;
}

[data-testid="stSidebar"] label {
    color: #8892b0 !important;
    font-size: 0.8rem !important;
    font-family: 'DM Mono', monospace !important;
}

[data-testid="stSidebar"] .sidebar-section-title {
    color: #f5f2ed;
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 0.8rem 0 0.3rem 0;
    border-top: 1px solid #2d2d4e;
    margin-top: 0.5rem;
}

.stTabs [data-baseweb="tab-list"] {
    background: #f5f2ed;
    border-radius: 12px;
    gap: 4px;
    padding: 4px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    color: #1a1a2e ;   /* change this to 1a1a2e */
}
/*this below .stTabs id for the hover */
.stTabs [data-baseweb="tab"]:hover {
    color: #e11d48 !important;  /* reddish color on hover */
    background: transparent !important;
}
.stTabs [aria-selected="true"]:hover {
    background: #1a1a2e !important;
    color: #f5f2ed !important;
}
            
.stTabs [aria-selected="true"] {
    background: #1a1a2e !important;
    color: #f5f2ed !important;
}

.stButton > button {
    background: #1a1a2e !important;
    color: #f5f2ed !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.2s ease !important;
}


/* Target the specific attribute for the deploy button */
            
[data-testid="stAppDeployButton"] {
    display: none !important;
}

/* Also target the header's right-side container to ensure no gap is left */
.stAppDeployButton, .st-emotion-cache-15ec6hl {
    display: none !important;
    width: 0px !important;
    height: 0px !important;
    overflow: hidden !important;
}


.stButton > button:hover {
    background: #2d2d5e !important;
    transform: translateY(-1px) !important;
}
            
/*new css line------*/
.stMarkdown p {
    color: #1a1a2e !important;
    opacity: 0.8; /* Optional: makes it look slightly softer/professional */
}
/*new css end here---*/            

.whatif-change {
    font-family: 'DM Mono', monospace;
    font-size: 0.70rem;
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin-bottom: 0.5rem;
    border-left: 3px solid;
    color: #4a4a5e !important;
}

/* 1. Force the collapse/expand button to be 100% opaque at all times */
[data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    visibility: visible !important;
}

/* 2. Target the SVG inside so it's not faint */
[data-testid="stSidebarCollapseButton"] svg {
    fill: #8892b0 !important; /* Muted grey-blue */
    opacity: 1 !important;
}

/* 3. The "Glow" effect: when the cursor crosses it, it gets bright white and bigger */
[data-testid="stSidebarCollapseButton"]:hover svg {
    fill: #ffffff !important;
    filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.9));
    transform: scale(1.2);
    transition: all 0.2s ease-in-out;
}

/* 4. This is the "Magic Fix" for the right-side cursor issue: 
   It stops the container from fading out when the mouse leaves the sidebar */
div[data-testid="stSidebar"] + section + div [data-testid="stSidebarCollapseButton"],
header + div [data-testid="stSidebarCollapseButton"] {
    opacity: 1 !important;
    display: flex !important;
}

/* 5. Fixes the 'Expand' button (the one that appears when sidebar is closed) */
button[kind="headerNoPadding"] {
    opacity: 1 !important;
    visibility: visible !important;
}

/*this code the top bar make dark colour even in light theme*/       
[data-testid="stHeader"] {
    background-color: #1a1a2e !important;
}

[data-testid="stHeader"] button {
    color: #f5f2ed !important;
}

[data-testid="stHeader"] svg {
    fill: #f5f2ed !important;
    color: #f5f2ed !important;
}
            
/*foter code css---------------------------------------------*/
[data-testid="stSidebar"] {
    height: calc(100vh - 50px) !important;
    overflow-y: auto !important;
}
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #1a1a2e;
    padding: 0.65rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    z-index: 9999;
    border-top: 1px solid #2d2d4e;
}

.footer-left {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.footer-brand {
    color: #f5f2ed;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.footer-sep {
    color: #2d2d4e;
    font-size: 1rem;
}

.footer-tagline {
    color: #8892b0;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
}

.footer-right {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}

.footer-link {
    color: #8892b0;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    text-decoration: none;
    transition: color 0.2s ease;
    cursor: pointer;
}

.footer-link:hover {
    color: #f5f2ed;
}

.footer-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #22c55e;
    display: inline-block;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* Push page content up so footer doesn't overlap bottom elements */
.main .block-container {
    padding-bottom: 4rem !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    model = load(os.path.join(MODELS_DIR, "best_model.pkl"))
    preprocessor = load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    feature_names = load(os.path.join(MODELS_DIR, "feature_names.pkl"))
    column_info = load(os.path.join(MODELS_DIR, "column_info.pkl"))
    comparison = load(os.path.join(MODELS_DIR, "model_comparison.pkl"))
    raw_columns = load(os.path.join(MODELS_DIR, "raw_columns.pkl"))
    meta = load(os.path.join(MODELS_DIR, "meta.pkl"))
    return model, preprocessor, feature_names, column_info, comparison, raw_columns, meta


def build_sidebar(column_info, raw_columns):
    categorical_cols = column_info["categorical_cols"]
    numeric_cols = column_info["numeric_cols"]
    binary_cols = column_info["binary_cols"]
    cat_options = raw_columns["categorical_options"]
    num_ranges = raw_columns["numeric_ranges"]

    user_inputs = {}

    st.sidebar.markdown("""
    <div style='padding: -1rem 0 0.5rem;'>
        <span style='color: #f5f2ed; font-size: 1.8rem; font-weight: 980;'>🔭 ChurnIQ</span>
        <p style='color: #8892b0; font-size: 0.75rem; font-family: DM Mono, monospace; margin-top: -1.3rem;'>Customer Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("<div class='sidebar-section-title'>Demographics</div>", unsafe_allow_html=True)
    demo_cols = ["gender", "Partner", "Dependents"]
    for col in demo_cols:
        if col in categorical_cols:
            options = cat_options.get(col, [])
            user_inputs[col] = st.sidebar.selectbox(col, options=options, key=f"sb_{col}")

    user_inputs["SeniorCitizen"] = st.sidebar.selectbox("SeniorCitizen", options=[0, 1], key="sb_SeniorCitizen")

    st.sidebar.markdown("<div class='sidebar-section-title'>Services</div>", unsafe_allow_html=True)
    service_cols = ["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
                    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    for col in service_cols:
        if col in categorical_cols:
            options = cat_options.get(col, [])
            user_inputs[col] = st.sidebar.selectbox(col, options=options, key=f"sb_{col}")

    st.sidebar.markdown("<div class='sidebar-section-title'>Billing</div>", unsafe_allow_html=True)
    billing_cols = ["Contract", "PaperlessBilling", "PaymentMethod"]
    for col in billing_cols:
        if col in categorical_cols:
            options = cat_options.get(col, [])
            user_inputs[col] = st.sidebar.selectbox(col, options=options, key=f"sb_{col}")

    st.sidebar.markdown("<div class='sidebar-section-title'>Charges & Tenure</div>", unsafe_allow_html=True)
    for col in numeric_cols:
        r = num_ranges[col]
        step = 1.0 if col == "tenure" else 0.5
        user_inputs[col] = st.sidebar.slider(
            col,
            min_value=float(r["min"]),
            max_value=float(r["max"]),
            value=float(r["mean"]),
            step=step,
            key=f"sb_{col}"
        )

    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.sidebar.button("🔮 Analyse Customer", use_container_width=True)
    return user_inputs, predict_clicked


def encode_inputs(user_inputs, preprocessor, column_info):
    categorical_cols = column_info["categorical_cols"]
    numeric_cols = column_info["numeric_cols"]
    binary_cols = column_info["binary_cols"]
    all_cols = categorical_cols + numeric_cols + binary_cols
    row = {col: user_inputs.get(col, 0) for col in all_cols}
    input_df = pd.DataFrame([row])
    return preprocessor.transform(input_df)


def predict(model, encoded_input, threshold):
    prob = model.predict_proba(encoded_input)[0][1]
    pred = int(prob >= threshold)
    return pred, prob


def compute_shap(model, encoded_input):
    if not SHAP_AVAILABLE:
        return None, None
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded_input)
        if isinstance(shap_values, list):
            return explainer, shap_values[1]
        return explainer, shap_values
    except Exception:
        return None, None


def render_shap_chart(shap_vals, feature_names, top_n=12):
    if shap_vals is None:
        st.info("SHAP not available. Install with: pip install shap")
        return

    shap_series = pd.Series(shap_vals[0], index=feature_names)
    top_shap = shap_series.abs().nlargest(top_n)
    top_shap_vals = shap_series[top_shap.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafaf8")

    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in top_shap_vals.values]
    bars = ax.barh(range(len(top_shap_vals)), top_shap_vals.values[::-1], color=colors[::-1], height=0.6)
    ax.set_yticks(range(len(top_shap_vals)))
    ax.set_yticklabels([f.replace("_", " ")[:22] for f in top_shap_vals.index[::-1]],
                        fontsize=8, fontfamily="monospace")
    ax.axvline(0, color="#1a1a2e", linewidth=1.2)
    ax.set_xlabel("SHAP Value (impact on churn probability)", fontsize=8)
    ax.set_title("Why this prediction?", fontsize=11, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_model_comparison(comparison):
    st.markdown("<div class='tab-header'>Model Performance Comparison</div>", unsafe_allow_html=True)
    rows = []
    for model_name, metrics in comparison.items():
        rows.append({
            "Model": model_name,
            "Test AUC": f"{metrics['test_auc']:.4f}",
            "Test F1": f"{metrics['test_f1']:.4f}",
            "CV AUC (5-fold)": f"{metrics['cv_auc']:.4f}",
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#fafaf8")
    model_names = [r["Model"] for r in rows]
    auc_vals = [float(r["Test AUC"]) for r in rows]
    f1_vals = [float(r["Test F1"]) for r in rows]
    x = np.arange(len(model_names))
    ax.bar(x - 0.2, auc_vals, width=0.35, label="Test AUC", color="#1a1a2e")
    ax.bar(x + 0.2, f1_vals, width=0.35, label="Test F1", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    ax.set_title("AUC & F1 by Model", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


def render_whatif(user_inputs, model, preprocessor, column_info, threshold, base_prob):
    st.markdown("<div class='tab-header'>What-If Scenario Simulator</div>", unsafe_allow_html=True)
    st.markdown("See how changing one attribute affects churn probability.")

    scenarios = [
        ("Switch to 2-year contract", {"Contract": "Two year"}),
        ("Switch to 1-year contract", {"Contract": "One year"}),
        ("Add Tech Support", {"TechSupport": "Yes"}),
        ("Add Online Security", {"OnlineSecurity": "Yes"}),
        ("Switch to auto-pay", {"PaymentMethod": "Bank transfer (automatic)"}),
        ("+12 months tenure", {"tenure": min(user_inputs.get("tenure", 0) + 12, 72)}),
        ("Reduce monthly charge by $20", {"MonthlyCharges": max(18.0, user_inputs.get("MonthlyCharges", 65) - 20)}),
    ]

    results = []
    for label, overrides in scenarios:
        modified = {**user_inputs, **overrides}
        enc = encode_inputs(modified, preprocessor, column_info)
        _, prob = predict(model, enc, threshold)
        delta = prob - base_prob
        results.append((label, prob, delta))

    results.sort(key=lambda x: x[1])

    for label, prob, delta in results:
        direction = "▼" if delta < 0 else "▲"
        color = "#22c55e" if delta < 0 else "#ef4444"
        bg = "#f0fdf4" if delta < 0 else "#fef2f2"
        st.markdown(f"""
        <div class='whatif-change' style='background:{bg}; border-color:{color};'>
            <strong>{label}</strong>
            <span style='float:right; color:{color}; font-weight:700;'>{direction} {abs(delta):.1%} → {prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)


def render_batch_prediction(model, preprocessor, column_info, threshold):
    st.markdown("<div class='tab-header'>Batch Prediction — Upload CSV</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#1a1a2e; font-size:0.85rem; margin-bottom:0.3rem;'>Upload customer CSV (same columns as training data, without Churn column)</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")
    # uploaded = st.file_uploader("Upload customer CSV (same columns as training data, without Churn column)", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            if "customerID" in batch_df.columns:
                customer_ids = batch_df["customerID"].tolist()
                batch_df = batch_df.drop(columns=["customerID"])
            else:
                customer_ids = [f"C{i:04d}" for i in range(len(batch_df))]

            if "Churn" in batch_df.columns:
                batch_df = batch_df.drop(columns=["Churn"])

            batch_df["TotalCharges"] = pd.to_numeric(batch_df["TotalCharges"], errors="coerce")
            batch_df.dropna(inplace=True)

            categorical_cols = column_info["categorical_cols"]
            numeric_cols = column_info["numeric_cols"]
            binary_cols = column_info["binary_cols"]
            all_cols = categorical_cols + numeric_cols + binary_cols
            batch_df = batch_df[[col for col in all_cols if col in batch_df.columns]]

            X_batch = preprocessor.transform(batch_df)
            probs = model.predict_proba(X_batch)[:, 1]
            preds = (probs >= threshold).astype(int)

            results_df = pd.DataFrame({
                "CustomerID": customer_ids[:len(probs)],
                "ChurnProbability": np.round(probs, 4),
                "Prediction": ["Churn" if p == 1 else "Stay" for p in preds],
                "RiskLevel": [get_churn_risk_label(p)[0] for p in probs],
            }).sort_values("ChurnProbability", ascending=False)

            # col1, col2, col3 = st.columns(3)
            # with col1:
            #     st.metric("Total Customers", len(results_df))
            # with col2:
            #     st.metric("Predicted Churners", int(preds.sum()))
            # with col3:
            #     st.metric("Avg Churn Risk", f"{probs.mean():.1%}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-number' style='color:#1a1a2e;'>{len(results_df)}</div>
                    <div class='stat-label'>Total Customers</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-number' style='color:#ef4444;'>{int(preds.sum())}</div>
                    <div class='stat-label'>Predicted Churners</div>
                </div>""", unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class='card' style='text-align:center;'>
                    <div class='stat-number' style='color:#f59e0b;'>{probs.mean():.1%}</div>
                    <div class='stat-label'>Avg Churn Risk</div>
                </div>""", unsafe_allow_html=True)

            st.dataframe(results_df, use_container_width=True, hide_index=True)

            csv_output = results_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_output, "churn_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error processing file: {e}")


def main():
    #footer code---------------------------------------- bcz of the load time if it is in the last it load 2sec later
    st.markdown("""
    <div class='footer' id='churniq-footer'>
        <div class='footer-left'>
            <span class='footer-brand'>🔭 ChurnIQ</span>
            <span class='footer-sep'>|</span>
            <span class='footer-tagline'>Telco Customer Intelligence Platform</span>
        </div>
        <div class='footer-right'>
            <span class='footer-dot'></span>
            <span class='footer-link'>Model Active</span>
            <span class='footer-sep' style='color:#2d2d4e;'>·</span>
            <span class='footer-link'>v1.0.0</span>
            <span class='footer-sep' style='color:#2d2d4e;'>·</span>
            <span class='footer-link'>© 2026 ChurnIQ</span>
        </div>
    </div>

    <script>
    (function() {
        function syncFooter() {
            var footer = document.getElementById('churniq-footer');
            if (!footer) return;
            
            var sidebar = document.querySelector('[data-testid="stSidebar"]');
            if (!sidebar) { footer.style.left = '0px'; return; }
            
            var expanded = sidebar.getAttribute('aria-expanded');
            if (expanded === 'false') {
                footer.style.left = '0px';
            } else {
                var w = sidebar.offsetWidth;
                footer.style.left = (w > 0 ? w : 300) + 'px';
            }
        }

        syncFooter();
        setInterval(syncFooter, 200);

        var sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            new MutationObserver(syncFooter).observe(sidebar, {
                attributes: true,
                attributeFilter: ['aria-expanded', 'style', 'class']
            });
        }
    })();
    </script>
    """, unsafe_allow_html=True)
    #footer code------------------------------------end here----------------------------------
    st.markdown("""
    <div class='top-bar'>
        <div>
            <h1>🔭 ChurnIQ</h1>
            <div class='subtitle'>Telco Customer Intelligence · Random Forest + SHAP + CLV + What-If</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        model, preprocessor, feature_names, column_info, comparison, raw_columns, meta = load_artifacts()
        threshold = meta["threshold"]
        best_model_name = meta["best_model_name"]
    except FileNotFoundError:
        st.error("⚠️ Model artifacts not found. Run `python src/train.py` first.")
        st.stop()

    user_inputs, predict_clicked = build_sidebar(column_info, raw_columns)

    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Prediction & SHAP", "💰 CLV & Revenue Risk", "🔄 What-If Simulator", "📊 Model Comparison & Batch"])

    if predict_clicked:
        encoded_input = encode_inputs(user_inputs, preprocessor, column_info)
        prediction, churn_prob = predict(model, encoded_input, threshold)
        no_churn_prob = 1 - churn_prob

        _, shap_vals = compute_shap(model, encoded_input)
        shap_top = []
        if shap_vals is not None:
            shap_series = pd.Series(shap_vals[0], index=feature_names)
            shap_top = shap_series.abs().nlargest(5).index.tolist()

        risk_label, risk_color = get_churn_risk_label(churn_prob)
        strategies = generate_retention_strategies(user_inputs, churn_prob, shap_top)

        with tab1:
            left, right = st.columns([1, 1.2], gap="large")

            with left:
                st.markdown("<div class='tab-header'>Prediction Result</div>", unsafe_allow_html=True)

                result_bg = "#fff0f0" if prediction == 1 else "#f0fff4"
                result_border = "#ef4444" if prediction == 1 else "#22c55e"
                result_icon = "⚠️" if prediction == 1 else "✅"
                result_text = "LIKELY TO CHURN" if prediction == 1 else "LIKELY TO STAY"
                result_color = "#ef4444" if prediction == 1 else "#22c55e"

                st.markdown(f"""
                <div style='background:{result_bg}; border:2px solid {result_border}; border-radius:16px; padding:1.8rem; text-align:center; margin-bottom:1rem;'>
                    <div style='font-size:2.5rem;'>{result_icon}</div>
                    <div style='font-size:1.6rem; font-weight:800; color:{result_color}; letter-spacing:-0.03em;'>{result_text}</div>
                    <div style='font-family:DM Mono,monospace; font-size:0.85rem; color:#555; margin-top:0.4rem;'>Confidence: {max(churn_prob, no_churn_prob):.1%}</div>
                    <span class='risk-badge' style='background:{risk_color}22; color:{risk_color}; margin-top:0.6rem;'>{risk_label} Risk</span>
                </div>
                """, unsafe_allow_html=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"""
                    <div class='card' style='text-align:center;'>
                        <div class='stat-number' style='color:#ef4444;'>{churn_prob:.1%}</div>
                        <div class='stat-label'>Churn Probability</div>
                    </div>""", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"""
                    <div class='card' style='text-align:center;'>
                        <div class='stat-number' style='color:#22c55e;'>{no_churn_prob:.1%}</div>
                        <div class='stat-label'>Retention Probability</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("**Churn Risk Gauge**")
                st.progress(float(churn_prob))
                st.markdown(f"<div style='font-family:DM Mono,monospace; font-size:0.75rem; color:#888;'>Model: {best_model_name} · Threshold: {threshold:.2f}</div>", unsafe_allow_html=True)

                st.markdown("<div class='tab-header' style='margin-top:1.5rem;'>Retention Strategies</div>", unsafe_allow_html=True)
                for s in strategies:
                    st.markdown(f"""
                    <div class='strategy-card' style='border-color:{s["color"]};'>
                        <div class='strategy-title'>{s["icon"]} {s["action"]} <span class='risk-badge' style='background:{s["color"]}22; color:{s["color"]}; font-size:0.65rem;'>{s["priority"]}</span></div>
                        <div class='strategy-detail'>{s["detail"]}</div>
                        <div class='strategy-impact'>→ {s["impact"]}</div>
                    </div>
                    """, unsafe_allow_html=True)

            with right:
                st.markdown("<div class='tab-header'>SHAP Explanation — Why This Prediction?</div>", unsafe_allow_html=True)
                render_shap_chart(shap_vals, feature_names)

                st.markdown("<div class='tab-header' style='margin-top:1rem;'>Feature Impact Table</div>", unsafe_allow_html=True)
                if shap_vals is not None:
                    shap_df = pd.DataFrame({
                        "Feature": feature_names,
                        "SHAP Value": np.round(shap_vals[0], 4),
                        "Direction": ["↑ Increases Churn" if v > 0 else "↓ Reduces Churn" for v in shap_vals[0]]
                    }).sort_values("SHAP Value", key=abs, ascending=False).head(12)
                    st.dataframe(shap_df, use_container_width=True, hide_index=True, height=320)

        with tab2:
            monthly = user_inputs.get("MonthlyCharges", 65.0)
            tenure = user_inputs.get("tenure", 12)
            contract = user_inputs.get("Contract", "Month-to-month")

            clv = calculate_clv(monthly, tenure, churn_prob, contract)
            revenue_at_risk = calculate_revenue_at_risk(monthly, churn_prob)
            roi_data = calculate_retention_roi(revenue_at_risk)
            tier_name, tier_color = get_clv_tier(clv)

            st.markdown("<div class='tab-header'>Customer Lifetime Value Analysis</div>", unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            metrics = [
                (f"${clv:,.0f}", "Est. CLV", "#1a1a2e"),
                (f"${revenue_at_risk:,.0f}", "Revenue at Risk", "#ef4444"),
                (f"${roi_data['net_benefit']:,.0f}", "Net Retention Benefit", "#22c55e" if roi_data["net_benefit"] > 0 else "#ef4444"),
                (f"{roi_data['roi_percent']:.0f}%", "Retention ROI", "#f59e0b"),
            ]
            for col, (val, label, color) in zip([c1, c2, c3, c4], metrics):
                with col:
                    st.markdown(f"""
                    <div class='card' style='text-align:center;'>
                        <div class='stat-number' style='color:{color};'>{val}</div>
                        <div class='stat-label'>{label}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class='card'>
                <div style='display:flex; align-items:center; gap:1rem;'>
                    <div>
                        <span style='font-size:0.85rem; color:#888;'>Customer Tier</span><br>
                        <span class='tier-badge' style='background:{tier_color}22; color:{tier_color}; font-size:1rem; padding:0.4rem 1.2rem; margin-top:0.3rem;'>
                            {tier_name}
                        </span>
                    </div>
                    <div style='flex:1; padding-left:1rem; border-left:2px solid #e8e4dd;'>
                        <div style='font-size:0.85rem; color:#555;'>
                            Based on estimated CLV of <strong>${clv:,.0f}</strong>.
                            {"✅ <strong>Worth retaining</strong> — retention ROI is positive." if roi_data["worthwhile"] else "⚠️ <strong>Borderline case</strong> — consider cost-effective interventions only."}
                        </div>
                        <div style='font-family:DM Mono,monospace; font-size:0.75rem; color:#888; margin-top:0.5rem;'>
                            Revenue at risk: ${revenue_at_risk:,.0f} · Retention cost: ${roi_data["retention_cost"]:,.0f} · Net benefit: ${roi_data["net_benefit"]:,.0f}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div class='tab-header' style='margin-top:1rem;'>CLV Breakdown</div>", unsafe_allow_html=True)
            fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
            fig.patch.set_facecolor("#ffffff")

            labels = ["Estimated CLV", "Revenue at Risk", "Retention Cost"]
            values = [clv, revenue_at_risk, roi_data["retention_cost"]]
            colors_bar = ["#1a1a2e", "#ef4444", "#f59e0b"]
            axes[0].bar(labels, values, color=colors_bar, width=0.5)
            axes[0].set_facecolor("#fafaf8")
            axes[0].spines[["top", "right"]].set_visible(False)
            axes[0].set_title("Financial Overview ($)", fontweight="bold", fontsize=10)
            for i, v in enumerate(values):
                axes[0].text(i, v + 10, f"${v:,.0f}", ha="center", fontsize=8, fontfamily="monospace")

            wedge_data = [max(0, revenue_at_risk - roi_data["retention_cost"]), roi_data["retention_cost"]]
            wedge_labels = ["Net Benefit", "Retention Cost"]
            axes[1].pie(wedge_data, labels=wedge_labels, colors=["#22c55e", "#f59e0b"],
                        autopct="%1.0f%%", startangle=90, textprops={"fontsize": 9})
            axes[1].set_title("Cost vs Benefit", fontweight="bold", fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with tab3:
            render_whatif(user_inputs, model, preprocessor, column_info, threshold, churn_prob)

        with tab4:
            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                render_model_comparison(comparison)
            with col_right:
                render_batch_prediction(model, preprocessor, column_info, threshold)

    else:
        with tab1:
            st.markdown("""
            <div style='background:#ffffff; border:2px dashed #e8e4dd; border-radius:20px; padding:4rem; text-align:center; margin-top:1rem;'>
                <div style='font-size:3.5rem;'>🔭</div>
                <div style='font-size:1.3rem; font-weight:800; color:#1a1a2e; margin-top:1rem; letter-spacing:-0.02em;'>Configure a customer in the sidebar</div>
                <div style='color:#8892b0; font-family:DM Mono,monospace; font-size:0.8rem; margin-top:0.5rem;'>
                    Fill in attributes → Click "Analyse Customer" → Get SHAP explanations, CLV, strategies & what-if scenarios
                </div>
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                render_model_comparison(comparison)
            with col_right:
                st.markdown("<div class='tab-header'>Batch Prediction</div>", unsafe_allow_html=True)
                st.info("Run a single prediction first to unlock batch upload, or use the uploader below directly.")
                render_batch_prediction(model, preprocessor, column_info, threshold)
    

if __name__ == "__main__":
    main()