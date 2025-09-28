# ===================================
# YouTube Ad Revenue Estimator - Streamlit App
# ===================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -----------------------------
# Load Model and Dataset
# -----------------------------
model = joblib.load("C:/Ganapathy/Proj 3/artifacts/model_pipeline.joblib")
df = pd.read_csv("C:/Ganapathy/Proj 3/youtube_ad_revenue_dataset.csv")

st.set_page_config(page_title="YouTube Ad Revenue Estimator", layout="wide")

# -----------------------------
# Sidebar: User Inputs
# -----------------------------
st.sidebar.header("Input Video Metrics")
views = st.sidebar.number_input("Views", min_value=0, value=10000, step=100)
likes = st.sidebar.number_input("Likes", min_value=0, value=500, step=1)
comments = st.sidebar.number_input("Comments", min_value=0, value=20, step=1)
watch_time = st.sidebar.number_input("Watch time (minutes)", min_value=0, value=1500, step=10)
video_length = st.sidebar.number_input("Video length (minutes)", min_value=0, value=10, step=1)
subscribers = st.sidebar.number_input("Subscribers", min_value=0, value=10000, step=100)

category = st.sidebar.text_input("Category", "Education")
device = st.sidebar.text_input("Device", "Mobile")
country = st.sidebar.text_input("Country", "US")

# Engagement rate
engagement_rate = (likes + comments) / max(views, 1)

input_df = pd.DataFrame([{
    "views": views,
    "likes": likes,
    "comments": comments,
    "watch_time_minutes": watch_time,
    "video_length_minutes": video_length,
    "subscribers": subscribers,
    "engagement_rate": engagement_rate,
    "category": category,
    "device": device,
    "country": country
}])

# -----------------------------
# Tabs for App Layout
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "Dataset Overview", "Feature Insights", "What-If Simulator"])

# -----------------------------
# Tab 1: Prediction
# -----------------------------
with tab1:
    st.header("🎯 Predict Ad Revenue")
    st.write("### Input Summary")
    st.write(input_df)
    
    if st.button("Predict Revenue"):
        pred = model.predict(input_df)[0]
        st.success(f"Estimated Ad Revenue: ${pred:,.2f}")

# -----------------------------
# Tab 2: Dataset Overview
# -----------------------------
with tab2:
    st.header("📊 Dataset Exploration")
    st.write("### Top 5 rows of dataset")
    st.dataframe(df.head())

    st.write("### Views Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['views'], bins=50, ax=ax, color="skyblue")
    ax.axvline(views, color='red', linestyle='--', label="Your Input")
    ax.set_xlabel("Views")
    ax.set_ylabel("Frequency")
    ax.legend()
    st.pyplot(fig)

    st.write("### Average Revenue by Category")
    category_avg = df.groupby("category")["ad_revenue_usd"].mean().sort_values(ascending=False).head(10)
    st.bar_chart(category_avg)

# -----------------------------
# Tab 3: Feature Insights
# -----------------------------
with tab3:
    st.header("🔍 Feature Importance")
    if hasattr(model.named_steps['model'], 'feature_importances_'):
        importances = model.named_steps['model'].feature_importances_
        feature_names = list(model.named_steps['preprocessor'].transformers_[0][2]) + \
                        list(model.named_steps['preprocessor'].transformers_[1][1]['onehot'].get_feature_names_out())
        feat_imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
        st.bar_chart(feat_imp_df.set_index("Feature"))
    else:
        st.info("Feature importance not available for linear models.")

# -----------------------------
# Tab 4: What-If Simulator
# -----------------------------
with tab4:
    st.header("⚡ Simulate Revenue for Different Views")
    min_views, max_views = st.slider("Select Views Range", min_value=0, max_value=100000, value=(1000, 20000), step=1000)
    pred_list = []

    for v in range(min_views, max_views+1, 1000):
        temp_df = pd.DataFrame([{
            "views": v,
            "likes": likes,
            "comments": comments,
            "watch_time_minutes": watch_time,
            "video_length_minutes": video_length,
            "subscribers": subscribers,
            "engagement_rate": (likes + comments) / max(v, 1),
            "category": category,
            "device": device,
            "country": country
        }])
        pred_list.append(model.predict(temp_df)[0])

    sim_df = pd.DataFrame({"Views": range(min_views, max_views+1, 1000), "Predicted Revenue": pred_list})
    st.line_chart(sim_df.set_index("Views")["Predicted Revenue"])
