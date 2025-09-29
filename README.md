# YouTube Trending Data Analysis

This project provides a complete end-to-end data pipeline for analyzing YouTube Trending Videos, including data cleaning, sentiment/emotion analysis, exploratory data analysis (EDA), machine learning models, model comparison, and SHAP explainability.
The central question of this project is whether engagement metrics and emotional signals in video metadata can reliably predict whether a YouTube video will stay trending for at least 7 days.

---
## Project Overview

This project investigates what factors drive a YouTube video to become a long-term trending video (≥ 7 days). The pipeline integrates data cleaning, exploratory analysis, and multiple machine learning models.

I evaluated three models:

1. Logistic Regression (baseline) – chosen for simplicity and interpretability, serving as a benchmark for linear relationships. The model is wrapped with CalibratedClassifierCV because the dataset is class-imbalanced (only a minority of videos trend longer than 7 days). In such cases, raw logistic regression scores can be poorly calibrated, meaning a predicted probability like 0.7 does not actually correspond to a 70% chance. Calibration adjusts these outputs so that predicted probabilities better match the true likelihood of a video being long-term trending.

2. Logistic Regression with sentiment features – same setup as the baseline, but extended with Hugging Face emotion classification (joy, anger, etc.) from titles, tags, and descriptions to test whether emotional tone adds predictive value. This model is also calibrated, ensuring its probability outputs remain interpretable and directly comparable to both the baseline and nonlinear models.

3. XGBoost with sentiment features – a nonlinear tree-based model widely used in practice, included to test whether more complex patterns outperform linear baselines.

The factors considered include engagement metrics (views, likes, comments), categorical features (video category, channel), and sentiment features from text metadata. The main research questions were: Do emotions improve prediction beyond engagement metrics? and Do nonlinear models outperform simple linear baselines?

Results show that all models performed similarly, with ROC-AUC around 0.67–0.68. Adding emotion features led to only marginal improvements (+0.004–0.005 in precision/recall), suggesting that engagement dominates prediction while sentiment signals contribute little additional value. XGBoost reached slightly higher precision, but nonlinear modeling did not drastically outperform the linear baseline. These findings indicate that video virality in this dataset is explained mostly by basic engagement metrics rather than emotional tone.

---
## Dataset  
The dataset comes from [Kaggle – YouTube Trending Video Dataset](https://www.kaggle.com/datasnaek/youtube-new), which contains daily records of trending videos across multiple regions.  
This project only focus on the US region, using:
* **USvideos.csv** – trending video metadata  
* **US_category_id.json** – mapping of category IDs to human-readable names  

---

## Results & Insights

### Exploratory Data Analysis (EDA)
### Dataset
- **Raw dataset** (before cleaning):  
 ~40,000 daily trending records from Kaggle (U.S. region).  
  Contains 6,800+ unique video IDs, but with duplicates (same video trending on multiple days).  
  Includes 16 metadata fields (title, tags, views, likes, dislikes, comments, publish_time, etc.).  
  Issues: duplicates, missing descriptions, placeholder tags (“[none]”), disabled ratings/comments, removed videos.
- **After cleaning**:  
  6,351 unique trending videos from 2,198 channels
- **Views**:  
  - Mean ≈ 758 K  
  - SD ≈ 1.9 M, heavy-tailed; small fraction of viral videos dominate attention.
- **Likes**:  
  - Mean ≈ 34.5 K, strong right skewness driven by mega-hits.
- **Upload timing**: Afternoon & evening uploads yield higher average views, matching user engagement peaks.
- **Trending lifespan**:  
  - Most videos: 1–3 days  
  - Minority: &gt; 1 week
- **Sentiment**: Overwhelmingly positive; likes vastly outnumber dislikes across categories.
- **Category landscape**:  
  - **Music** & **Entertainment** dominate volume & views.  
  - **Education**, **News** appear far less frequently.
- **Tags**: Thematic clusters align with pop culture, trending events, fandom communities.

### Machine Learning Models
Goal: Predict long-term trending (≥ 7 days).

| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|-----|
| Baseline (no emotion) | 0.680 | 0.607 | 0.610 | 0.624 |
| With Emotion Features | 0.678 | 0.612 | 0.615 | 0.629 |
| XGBoost + Emotion | 0.672 | 0.620 | 0.618 | 0.626 |

- Emotion signals yield **modest** (+0.004‐0.005) precision/recall gain, some predictive value for virality.
- XGBoost achieves **highest precision**, but overall improvements remain **incremental**.
- SHAP insights: Top drivers are engagement metrics (likes, views, comments) & emotional categories.
---

## Project Structure
```markdown
youtube-trending-analysis/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_cleaning.py
│   ├── ai_sentiment_analysis.py
│   ├── eda/
│   │   ├── __init__.py
│   │   ├── eda_distribution.py
│   │   ├── eda_summary.py
│   │   └── eda_visualization.py
│   └── ml_models/
│       ├── __init__.py
│       ├── ml_modeling_basecase.py
│       ├── ml_modeling_with_emotion.py
│       ├── ml_modeling_xgboost.py
│       ├── ml_modeling_shap.py
│       └── compare_models_visual.py
├── test/
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_eda_distribution.py
│   ├── test_eda_summary.py
│   └── test_sentiment.py
├── main.py
├── requirements.txt
├── README.md
├── datasource/
└── outputs/
```

## Setup

```bash
# Clone repo
git clone https://github.com/xtzx402/YouTube-Trending-Analysis.git
cd YouTube-Trending-Analysis

# Install dependencies
pip install -r requirements.txt

# Update src/config.py with your DB connection:
DB_URI = "mysql+pymysql://username:password@localhost:3306/youtube"
```

## Usage
```bash
#Run the pipeline with:
python main.py --task <task_name>
```
Available Tasks
| Task Name            | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `clean_data`         | Load raw CSV/JSON, clean data, engineer features, save to DB |
| `sentiment`          | Hugging Face emotion analysis and save to DB                 |
| `distribution`       | Detect outliers (Z-score) and export Excel                   |
| `eda_summary`        | Compute summary statistics, save to DB                       |
| `eda_visual`         | Generate HTML report with plots and wordclouds               |
| `baseline_model`     | Logistic regression (no emotion features)                    |
| `with_emotion_model` | Logistic regression (with emotion features)                  |
| `xgb_model`          | XGBoost model with emotion features                          |
| `shap`               | SHAP explainability report for XGBoost                       |
| `model_comparison`   | Compare models (Chart.js HTML report)                        |
| `full_pipeline`      | Run **all steps sequentially**                               |


Example
```bash
python main.py --task full_pipeline
````
## Testing
Unit tests are included under `tests/`, covering data cleaning and EDA to ensure reproducibility of key functions.
```bash
#Run all tests:
pytest
```
Key test coverage:

test_data_cleaning.py: checks missing values & feature engineering

test_eda_distribution.py: verifies outlier detection + Excel outputs

test_eda_summary.py: validates summary statistics, tag frequency counts, and word–view analysis

## Outputs
All outputs are automatically written to the `outputs/` folder and selected results are also stored in the database for reproducibility and comparison.

| File / Table                          | Content                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `eda_analysis_report.html`            | Full EDA visualization report (matplotlib + Plotly + wordclouds)        |
| `distribution_analysis.xlsx`          | Outlier detection results (Z-score) + cleaned dataset                   |
| `ml_test_predictions_baseline.csv`    | Test predictions from baseline logistic regression (no emotion)         |
| `ml_test_predictions_with_emotion.csv`| Test predictions from logistic regression with emotion features         |
| `ml_test_predictions_xgboost.csv`     | Test predictions from XGBoost with emotion features                     |
| `shap_report.html`                    | SHAP explainability report (global importance bar + beeswarm plots)     |
| DB: `model_metrics`          | ROC-AUC, precision, recall, and F1 for all models (baseline, emotion, XGB) |
| DB: `model_params` (XGBoost)| Hyperparameters used in training (e.g., depth, learning rate, subsample)|
| DB: `feature_importance` (XGBoost)| Feature-level importance values exported from trained model             |

**Highlights:**
- All models use a consistent pipeline structure, ensuring outputs are directly comparable.  
- Logistic regression models are calibrated, making predicted probabilities interpretable.  
- XGBoost includes saved hyperparameters and feature importance, supporting deeper inspection of model behavior.  
- The SHAP report provides a global interpretability view, showing both ranking and distribution of feature impacts.


## Tech Stack
| Category       | Tools                                   |
| -------------- | --------------------------------------- |
| Language       | Python ≥ 3.9                            |
| Data           | Pandas, NumPy                           |
| Database       | MySQL + SQLAlchemy                      |
| Sentiment      | Hugging Face Transformers               |
| ML             | scikit-learn, XGBoost                   |
| Explainability | SHAP                                    |
| Visualization  | Matplotlib, Plotly, WordCloud, Chart.js,openpyxl |



## Notes

Requires a running MySQL database (configured via config.py).

outputs/ folder will be auto-created.

SHAP and Hugging Face models may take time on CPU; GPU is recommended.

## Future Work

### 1. Cross-region analysis
The current analysis is limited to the U.S. dataset, which makes the findings region-specific. A next step is to expand the study to include other regions such as Japan, Mexico, and the United Kingdom. This would allow for cross-cultural comparisons and help evaluate whether the same trends and predictive patterns hold globally or if regional differences lead to distinct dynamics.

### 2. Model optimization & feature engineering
At present, Logistic Regression and XGBoost achieved very similar performance, indicating that the available features (views, likes, comments, sentiment) capture mostly linear relationships with the target. The addition of sentiment features offered only marginal improvement, likely because I used Hugging Face’s emotion classification labels (joy, anger, etc.), which are coarse but interpretable. I did not include embeddings in this version due to the engineering overhead—handling high-dimensional vectors, storage, and explainability—but plan to explore them in the future to capture richer semantic signals. 

Another improvement will be addressing the class imbalance between short-term and long-term trending videos, which currently limits the model’s ability to learn minority cases. Techniques such as class weights or SMOTE can mitigate this issue, while hyperparameter tuning and testing of alternative models like LightGBM and CatBoost may reveal nonlinear patterns not captured by the baseline setup.

### 3. Deployment
Currently, the pipeline runs offline and produces static outputs such as reports and prediction files. A longer-term goal is to make the workflow more practical by packaging it as an API using FastAPI and containerizing it with Docker. This would enable real-time predictions and lay the foundation for cloud deployment, bringing the project closer to production use.

Together, these extensions would move the project from an exploratory analysis toward a scalable, production-ready system with broader generalizability.