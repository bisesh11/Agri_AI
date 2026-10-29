"""AgriAI - Enterprise Agronomic Decision Support System.

Production-grade AI Agricultural Assistant combining Multinomial Naive Bayes
classification with live online information retrieval from authentic agricultural sources.
Designed with a modern, high-contrast, professional mobile-first SaaS aesthetic.
"""

import os
import json
import streamlit as st
import pandas as pd
from PIL import Image

from chatbot import answer_question
from weather_service import get_live_weather
from market_service import get_mandi_prices, MANDI_PRICE_REGISTRY, format_mandi_price_markdown

# Page Configuration: mobile-first, clean slate layout
st.set_page_config(
    page_title="AgriAI - Agronomic Intelligence System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enterprise Modern UI Theme (High-End SaaS Slate & Emerald Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Global CSS Variables & Resets */
    :root {
        --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
        --bg-main: #f8fafc;
        --surface: #ffffff;
        --surface-subtle: #f1f5f9;
        --border: #e2e8f0;
        --border-focus: #94a3b8;
        --text-primary: #0f172a;
        --text-secondary: #475569;
        --text-muted: #64748b;
        --emerald-primary: #059669;
        --emerald-dark: #065f46;
        --emerald-light: #ecfdf5;
        --blue-primary: #0284c7;
        --blue-light: #f0f9ff;
    }

    html, body, [class*="css"] {
        font-family: var(--font-sans) !important;
    }

    .stApp {
        background-color: var(--bg-main);
        color: var(--text-primary);
    }

    /* Container Spacing for Clean Responsive Layout */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 4rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: 980px !important;
        margin: 0 auto;
    }

    @media (min-width: 768px) {
        .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            padding-top: 1.8rem !important;
        }
    }

    /* Executive Top Application Bar */
    .top-app-bar {
        background: #0f172a;
        color: #ffffff;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 14px -2px rgba(15, 23, 42, 0.12);
        display: flex;
        flex-direction: column;
        gap: 6px;
    }

    .app-brand-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 8px;
    }

    .brand-title-wrap {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .brand-logo-badge {
        background: #10b981;
        color: #0f172a;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
    }

    .top-app-bar h1 {
        color: #ffffff !important;
        font-size: clamp(1.2rem, 4vw, 1.65rem) !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }

    .top-app-bar p {
        color: #94a3b8 !important;
        font-size: clamp(0.80rem, 2.8vw, 0.90rem) !important;
        margin: 0 !important;
        line-height: 1.45;
    }

    .system-status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 20px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.2px;
    }

    .status-dot {
        width: 6px;
        height: 6px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 6px #10b981;
    }

    /* Minimalist Meta Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 10px;
    }

    .meta-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .badge-topic-node {
        background-color: #f1f5f9;
        color: #0f172a;
        border: 1px solid #cbd5e1;
    }

    .badge-conf-node {
        background-color: #f0f9ff;
        color: #0369a1;
        border: 1px solid #bae6fd;
    }

    .badge-source-node {
        background-color: #f0fdf4;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }

    /* Modern SaaS Segmented Tabs */
    div[data-baseweb="tab-list"] {
        background-color: #f1f5f9;
        padding: 4px;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        gap: 4px;
        margin-bottom: 16px;
        overflow-x: auto;
    }

    div[data-baseweb="tab"] {
        padding: 6px 14px !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        color: #475569 !important;
        background-color: transparent !important;
        border: none !important;
        transition: all 0.15s ease-in-out;
    }

    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08) !important;
    }

    /* Refined Query Suggestion Cards */
    .query-suggestion-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }

    .query-suggestion-box h3 {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 6px 0;
        letter-spacing: -0.01em;
    }

    .query-suggestion-box p {
        font-size: 0.82rem;
        color: #64748b;
        margin: 0 0 12px 0;
    }

    /* Minimalist Touch Buttons */
    div.stButton > button {
        min-height: 42px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.84rem;
        color: #1e293b;
        background-color: #ffffff;
        border: 1px solid #cbd5e1;
        transition: all 0.15s ease;
        padding: 8px 12px;
        text-align: left;
    }

    div.stButton > button:hover {
        background-color: #f8fafc;
        border-color: #059669;
        color: #059669;
    }

    /* Sleek KPI Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 14px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        color: #64748b !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        font-size: clamp(1.15rem, 3.5vw, 1.45rem) !important;
        font-weight: 700 !important;
        color: #0f172a !important;
    }

    /* Chat Messages - Executive High-Contrast Cards */
    .stChatMessage {
        border-radius: 10px !important;
        padding: 14px 16px !important;
        margin-bottom: 12px !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #f1f5f9 !important;
        border: 1px solid #e2e8f0 !important;
    }

    div[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background-color: #ffffff !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
    }

    /* Sources Accordion Polish */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #334155 !important;
        background-color: #f8fafc !important;
        border-radius: 6px !important;
    }

    /* Chat Input Bar */
    .stChatInput textarea {
        font-size: 15px !important;
        border-radius: 8px !important;
        border: 1px solid #cbd5e1 !important;
    }
    .stChatInput textarea:focus {
        border-color: #059669 !important;
        box-shadow: 0 0 0 2px rgba(5, 150, 105, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_question" not in st.session_state:
    st.session_state.input_question = ""

# Support URL test cases for validation and screenshots
query_case = st.query_params.get("test_case")
if query_case and len(st.session_state.messages) == 0:
    if query_case == "crop":
        sample_q = "What fertilizer practices are recommended for rice cultivation?"
        st.session_state.messages.append({"role": "user", "content": sample_q})
        res = answer_question(sample_q)
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "topic": res["predicted_topic"],
            "confidence": res["confidence"],
            "status": res["search_status"],
            "sources": res.get("sources", [])
        })
    elif query_case == "disease":
        sample_q = "Why are my rice plants turning yellow with brown spots?"
        st.session_state.messages.append({"role": "user", "content": sample_q})
        res = answer_question(sample_q)
        st.session_state.messages.append({
            "role": "assistant",
            "content": res["answer"],
            "topic": res["predicted_topic"],
            "confidence": res["confidence"],
            "status": res["search_status"],
            "sources": res.get("sources", [])
        })


def load_cached_metrics():
    """Loads evaluation metrics computed during evaluate_model.py."""
    metrics_path = os.path.join("results", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar: Minimalist Control Panel
with st.sidebar:
    st.markdown("### System Telemetry & Controls")
    st.caption("College Mini / Final-Year Technical Project")
    st.markdown("---")
    
    st.markdown("**Subsystem Diagnostics:**")
    st.markdown("""
    - Classifier: `MultinomialNB (v1.0)`
    - Retrieval: `Live Web Index (DDGS)`
    - Weather: `Open-Meteo REST API`
    - Mandi Registry: `Agmarknet Standards`
    """)
    
    st.markdown("---")
    st.markdown("**Inference Settings:**")
    answer_mode = st.selectbox(
        "Synthesis Engine",
        options=["auto", "extractive", "llm"],
        index=0,
        help="Default 'auto' selects local extractive evidence synthesis unless an LLM API key is detected."
    )
    
    confidence_thresh = st.slider(
        "Topic Decision Threshold",
        min_value=0.05,
        max_value=0.50,
        value=0.12,
        step=0.01,
        help="Posterior probability cutoff for flagging classification uncertainty."
    )
    
    st.markdown("---")
    st.markdown("**Accredited Repositories:**")
    st.markdown("""
    - [ICAR Central Directorate](https://icar.gov.in/)
    - [TNAU Agritech Portal](https://www.agritech.tnau.ac.in/)
    - [IMD Agricultural Meteorology](https://imdagrimet.gov.in/)
    - [Agmarknet Market Information](https://agmarknet.gov.in/)
    """)
    st.caption("Advisory Disclaimer: Recommendations are grounded in indexed agronomic advisories. Field validation with regional Krishi Vigyan Kendras (KVK) is advised.")


# Executive Top Application Bar
st.markdown("""
<div class="top-app-bar">
    <div class="app-brand-row">
        <div class="brand-title-wrap">
            <span class="brand-logo-badge">AGRI-AI</span>
            <h1>Agronomic Intelligence System</h1>
        </div>
        <div class="system-status-indicator">
            <span class="status-dot"></span>
            <span>SYSTEM ACTIVE</span>
        </div>
    </div>
    <p>Multinomial Naive Bayes Question Classification & Grounded Online Evidence Retrieval</p>
</div>
""", unsafe_allow_html=True)

# Segmented Navigation Tabs
tab_chat, tab_weather, tab_market, tab_analytics = st.tabs([
    "Consultation Desk",
    "Meteorological Advisory",
    "Wholesale Mandi Rates",
    "Classifier Telemetry"
])

# -------------------------------------------------------------
# TAB 1: Consultation Desk (Chat)
# -------------------------------------------------------------
with tab_chat:
    sample_queries = [
        ("Rice Yellowing & Spotting", "Why are my rice plants turning yellow with brown spots?"),
        ("Tomato Fruit Borer Protocol", "How can I control fruit borer and pests in tomato plants?"),
        ("Wheat NPK Fertilizer Dosage", "What is the recommended NPK fertilizer practice for wheat cultivation?"),
        ("Soil Organic Carbon Regeneration", "How can I improve soil organic carbon naturally without chemical fertilizer?"),
        ("District Agromet Weather Advisory", "What weather conditions and rain forecast should farmers prepare for in Coimbatore?"),
        ("Onion Mandi Spot Rates", "What is the current wholesale market price of onions in Lasalgaon mandi?")
    ]

    # Clean Prompts Grid (When Conversation Empty)
    if len(st.session_state.messages) == 0:
        st.markdown("""
        <div class="query-suggestion-box">
            <h3>Agronomic Consultation</h3>
            <p>Query pest identification, disease management, fertilization schedules, local weather advisories, or commodity rates. Select a standard benchmark query or enter your question below:</p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        for idx, (label, q_text) in enumerate(sample_queries):
            target_col = c1 if idx % 2 == 0 else c2
            with target_col:
                if st.button(f"{label} →\n\"{q_text[:34]}...\"", key=f"start_btn_{idx}", use_container_width=True):
                    st.session_state.pending_question = q_text
                    st.rerun()
    else:
        with st.expander("Benchmark Query Templates", expanded=False):
            c1, c2 = st.columns(2)
            for idx, (label, q_text) in enumerate(sample_queries):
                target_col = c1 if idx % 2 == 0 else c2
                with target_col:
                    if st.button(f"{label}", key=f"quick_btn_{idx}", use_container_width=True):
                        st.session_state.pending_question = q_text
                        st.rerun()

    # Message Stream
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                topic_label = msg.get("topic", "General").replace("_", " ").title()
                conf = msg.get("confidence", 0.0) * 100
                status = msg.get("status", "ONLINE_SUCCESS")
                
                st.markdown(
                    f'<div class="badge-container">'
                    f'<span class="meta-badge badge-topic-node">CLASSIFICATION: {topic_label.upper()}</span>'
                    f'<span class="meta-badge badge-conf-node">POSTERIOR CONF: {conf:.1f}%</span>'
                    f'<span class="meta-badge badge-source-node">RETRIEVAL: {status}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(msg["content"])
                
                sources = msg.get("sources", [])
                if sources:
                    with st.expander(f"Verified Evidence Records ({len(sources)})", expanded=False):
                        for idx, src in enumerate(sources, start=1):
                            title = src.get("title", f"Document {idx}")
                            url = src.get("url", "#")
                            domain = src.get("source", "Accredited Domain")
                            pub_date = src.get("published_date", "Recent")
                            snippet = src.get("snippet", "No excerpt provided.")
                            is_static = src.get("is_static_fallback", False)
                            
                            st.markdown(f"**[{idx}] [{title}]({url})**")
                            fallback_note = " *(Official Reference Archive)*" if is_static else ""
                            st.caption(f"Domain: `{domain}` | Published: `{pub_date}`{fallback_note}")
                            st.info(f"Evidence Excerpt: {snippet}")
                            st.markdown("---")

    # Input Bar
    user_input = st.chat_input("Enter agronomic query regarding crops, nutrition, diseases, weather, or mandi prices...")
    
    if hasattr(st.session_state, "pending_question") and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = None

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.write(user_input)
            
        with st.chat_message("assistant"):
            with st.spinner("Classifying intent with Naive Bayes & retrieving institutional evidence..."):
                result = answer_question(
                    question=user_input,
                    user_mode=answer_mode,
                    confidence_threshold=confidence_thresh
                )
                
                topic_label = result["predicted_topic"].replace("_", " ").title()
                conf = result["confidence"] * 100
                status = result["search_status"]
                
                st.markdown(
                    f'<div class="badge-container">'
                    f'<span class="meta-badge badge-topic-node">CLASSIFICATION: {topic_label.upper()}</span>'
                    f'<span class="meta-badge badge-conf-node">POSTERIOR CONF: {conf:.1f}%</span>'
                    f'<span class="meta-badge badge-source-node">RETRIEVAL: {status}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
                st.markdown(result["answer"])
                
                sources = result.get("sources", [])
                if sources:
                    with st.expander(f"Verified Evidence Records ({len(sources)})", expanded=True):
                        for idx, src in enumerate(sources, start=1):
                            title = src.get("title", f"Document {idx}")
                            url = src.get("url", "#")
                            domain = src.get("source", "Accredited Domain")
                            pub_date = src.get("published_date", "Recent")
                            snippet = src.get("snippet", "No excerpt provided.")
                            is_static = src.get("is_static_fallback", False)
                            
                            st.markdown(f"**[{idx}] [{title}]({url})**")
                            fallback_note = " *(Official Reference Archive)*" if is_static else ""
                            st.caption(f"Domain: `{domain}` | Published: `{pub_date}`{fallback_note}")
                            st.info(f"Evidence Excerpt: {snippet}")
                            st.markdown("---")
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "topic": result["predicted_topic"],
            "confidence": result["confidence"],
            "status": result["search_status"],
            "sources": result.get("sources", [])
        })

    if len(st.session_state.messages) > 0:
        if st.button("Reset Conversation Session", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# -------------------------------------------------------------
# TAB 2: Meteorological Advisory
# -------------------------------------------------------------
with tab_weather:
    st.subheader("Regional Agrometeorological Advisory")
    st.caption("Automated ingestion from Open-Meteo High-Resolution Meteorological Forecasting Models.")
    
    loc_query = st.text_input("Target Agricultural District or Municipality:", value="Coimbatore", key="weather_input")
    fetch_weather_btn = st.button("Retrieve Agromet Observations", use_container_width=True)
        
    if loc_query:
        weather_res = get_live_weather(loc_query)
        if weather_res.get("status") == "LIVE_SUCCESS":
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("SURFACE TEMPERATURE", f"{weather_res['current_temperature']} °C")
                st.metric("WIND VELOCITY (10M)", f"{weather_res['wind_speed_kmh']} km/h")
            with m_col2:
                st.metric("RELATIVE HUMIDITY", f"{weather_res['humidity']} %")
                st.metric("SKY CONDITION", weather_res['current_condition'])
            
            st.markdown("#### Agronomic Hazard & Spraying Advisories:")
            for adv in weather_res.get("agricultural_advisory", []):
                st.warning(adv)
                
            st.markdown("#### 5-Day Numerical Forecast Table:")
            forecast_df = pd.DataFrame(weather_res.get("daily_forecast", []))
            if not forecast_df.empty:
                forecast_df.rename(columns={
                    "date": "Date",
                    "max_temp": "Max (°C)",
                    "min_temp": "Min (°C)",
                    "rain_prob": "Precip Prob (%)",
                    "rain_sum_mm": "Precip Sum (mm)",
                    "condition": "Condition"
                }, inplace=True)
                st.dataframe(forecast_df, use_container_width=True)
                
            st.caption(f"Provider: [{weather_res['source']}]({weather_res['source_url']})")
        else:
            st.error(weather_res.get("message", "Meteorological telemetry is currently unreachable."))

# -------------------------------------------------------------
# TAB 3: Wholesale Mandi Rates
# -------------------------------------------------------------
with tab_market:
    st.subheader("Agmarknet Wholesale Mandi Bulletin")
    st.caption("Wholesale spot rates reported by Agricultural Produce Market Committees (APMC).")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        commodities_list = ["All Commodities"] + sorted(list({r["commodity"] for r in MANDI_PRICE_REGISTRY}))
        selected_commodity = st.selectbox("Commodity Filter:", options=commodities_list)
    with col_m2:
        states_list = ["All States"] + sorted(list({r["state"] for r in MANDI_PRICE_REGISTRY}))
        selected_state = st.selectbox("Jurisdiction Filter:", options=states_list)
        
    target_comm = None if selected_commodity == "All Commodities" else selected_commodity
    target_state = None if selected_state == "All States" else selected_state
    
    price_data = get_mandi_prices(commodity=target_comm, state=target_state)
    st.markdown(format_mandi_price_markdown(price_data))

# -------------------------------------------------------------
# TAB 4: Classifier Telemetry & Evaluation
# -------------------------------------------------------------
with tab_analytics:
    st.subheader("Multinomial Naive Bayes Model Telemetry")
    st.caption("Empirical evaluation metrics calculated on held-out test data (80-20 stratified split).")
    
    metrics = load_cached_metrics()
    if metrics:
        met_col1, met_col2 = st.columns(2)
        with met_col1:
            st.metric("TEST ACCURACY", f"{metrics['accuracy'] * 100:.2f}%")
            st.metric("MACRO RECALL", f"{metrics['macro_recall'] * 100:.2f}%")
        with met_col2:
            st.metric("MACRO PRECISION", f"{metrics['macro_precision'] * 100:.2f}%")
            st.metric("MACRO F1 SCORE", f"{metrics['macro_f1'] * 100:.2f}%")
        
        st.markdown("---")
        st.markdown("#### Confusion Matrix Heatmap (Actual vs Predicted)")
        matrix_img_path = os.path.join("results", "confusion_matrix.png")
        if os.path.exists(matrix_img_path):
            img = Image.open(matrix_img_path)
            st.image(img, caption="Multinomial Naive Bayes Confusion Matrix across 16 Classes", use_container_width=True)
            
        st.markdown("#### Hyperparameter & Pipeline Manifest:")
        st.write(f"- **Total Sample Corpus:** 480 verified questions")
        st.write(f"- **Held-Out Evaluation Split:** {metrics['test_samples']} questions (20%)")
        st.write(f"- **Target Domain Classes:** {metrics['num_classes']}")
        st.write(f"- **Algorithm Architecture:** Multinomial Naive Bayes (`MultinomialNB`, Laplace $\\alpha = 0.5$)")
        st.write(f"- **Feature Space:** Sublinear Term-Frequency Inverse Document Frequency (`TF-IDF`)")
        
        st.markdown("---")
        with st.expander("Academic Per-Class Classification Report (`evaluation_report.txt`)"):
            report_path = os.path.join("results", "evaluation_report.txt")
            if os.path.exists(report_path):
                with open(report_path, "r", encoding="utf-8") as f:
                    st.text(f.read())
    else:
        st.info("Execute `python evaluate_model.py` to compile telemetry metrics.")
