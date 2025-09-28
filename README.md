📺 YouTube Content Monetization Modeler
🔹 Project Overview

This project builds a Content Monetization Modeler that predicts YouTube ad revenue for videos based on their performance and contextual features.
It uses machine learning regression models and a Streamlit web app to provide creators and media companies with actionable insights.

✨ Key Features

Predict YouTube ad revenue from video metrics (views, likes, comments, etc.)

Explore dataset trends and perform EDA (Exploratory Data Analysis)

Visualize revenue patterns by category, views, and engagement

See feature importance (which factors matter most)

Run what-if simulations (e.g., how revenue changes if views increase)

🔹 Skills You’ll Learn

Regression Models (Linear, Ridge, Lasso, Random Forest, etc.)

Feature Engineering (e.g., engagement rate = (likes + comments) / views)

Data Cleaning & Preprocessing (missing values, outliers, encoding)

Model Evaluation (R², RMSE, MAE)

Data Visualization with Matplotlib & Seaborn

Interactive Web App with Streamlit

🔹 Dataset

Name: YouTube Monetization Modeler

Format: CSV (~122,000 rows, synthetic data)

Target Variable: ad_revenue_usd

Columns:

video_id, date

views, likes, comments

watch_time_minutes, video_length_minutes

subscribers, category, device, country

ad_revenue_usd (target)

📝 Steps Followed in the Project
1️⃣ Problem Understanding

Goal: Predict YouTube ad revenue (ad_revenue_usd) using video performance + contextual data.

Domain: Social Media Analytics.

Business impact:

Creators can optimize content strategy.

Media companies can forecast income.

Advertisers can estimate ROI.

2️⃣ Dataset Understanding

Size: ~122,000 rows (synthetic for learning).

Target variable: ad_revenue_usd.

Key features:

Quantitative → views, likes, comments, watch_time_minutes, video_length_minutes, subscribers.

Categorical → category, device, country.

Derived → engagement_rate = (likes + comments) / views.

3️⃣ Exploratory Data Analysis (EDA)

Checked dataset structure: df.info(), df.describe().

Visualizations:

Histogram of views, likes, comments.

Boxplots to check outliers.

Correlation heatmap (sns.heatmap).

Business insights:

High views → high revenue, but engagement rate also matters.

Certain categories/countries yield higher ad revenue.

4️⃣ Data Preprocessing

Missing values (~5%) → handled with mean/median for numeric, most frequent for categorical.

Duplicates (~2%) → removed with df.drop_duplicates().

Outlier detection → used IQR (Interquartile Range) method.

Encoding categorical features:

category, device, country → OneHotEncoding.

Scaling numeric features:

StandardScaler used for continuous variables (views, watch_time_minutes, etc.).

5️⃣ Feature Engineering

Created engagement_rate: (likes + comments) / views.

Log transformation on skewed variables (like views) to stabilize variance.

Interaction features tested (e.g., watch_time / video_length).

6️⃣ Model Building

Tried 5 regression models:

Linear Regression

Ridge Regression

Lasso Regression

Random Forest Regressor

Gradient Boosting Regressor

Built pipeline with:

ColumnTransformer → scaling + encoding.

Model → regression algorithm.

7️⃣ Model Evaluation

Metrics used:

R² → goodness of fit.

RMSE → penalizes large errors.

MAE → average absolute error.

Found that tree-based models (Random Forest, Gradient Boosting) performed better than plain Linear Regression.

Saved best model using joblib.dump().

8️⃣ Insights

Most important drivers of revenue:

Views & Watch Time (direct revenue correlation).

Engagement Rate (high engagement boosts CPM).

Category & Country (advertiser demand varies).

Example: Videos in Finance/Education categories generate more ad revenue than general Entertainment.

9️⃣ Streamlit App Development

Created app.py for interactive dashboard.

Features in app:

Prediction Tab – enter video metrics, get revenue estimate.

Dataset Overview – show dataset, histograms, bar charts.

Feature Insights – visualize feature importance.

What-If Simulator – simulate revenue by varying views.

User enters inputs → model predicts revenue → app displays results with charts.

🔟 Final Deliverables

Clean dataset (youtube_monetization.csv).

Trained model artifact (model_pipeline.joblib).

Jupyter Notebook (notebook.ipynb) with full workflow:

EDA, preprocessing, feature engineering, model building, evaluation.

Streamlit app (app.py) for interactive demo.

README.md with instructions & project overview.
