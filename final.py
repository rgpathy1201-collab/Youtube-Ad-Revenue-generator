
import streamlit as st
import pandas as pd
import joblib
# Load model
model = joblib.load("C:/Ganapathy/Proj 3/artifacts/model_pipeline.joblib")

st.title("📺 YouTube Ad Revenue Estimator")

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

engagement_rate = (likes + comments) / max(views, 1)

# Prepare input
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

st.write("### Input Summary")
st.write(input_df)

if st.button("Predict Revenue"):
    pred = model.predict(input_df)[0]
    st.success(f"Estimated Ad Revenue: ${pred:,.2f}")