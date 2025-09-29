import os
import io
import base64
import logging
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.config import DB_URI

#import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for embedding in HTML."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64


def generate_shap_report(output_file: str = "outputs/shap_report.html"):
    """Generate SHAP global interpretation report for trained XGBoost model.

    Loads cleaned video data and trained model pipeline, computes SHAP values,
    produces global importance plots, and saves as static HTML.

    Args:
        output_file (str): Path to save the SHAP HTML report.

    Returns:
        str: Path to generated HTML report, or empty string if failed.
    """
    logging.info("Starting SHAP report generation...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load dataset
    query = """
    SELECT v.video_id, v.video_category, v.likes, v.dislikes, v.comment_count, v.views,
           v.publish_hour, v.title_len, v.tags_count, v.long_term_trending, e.emotion
    FROM us_videos v
    LEFT JOIN video_emotion e
    ON v.video_id = e.video_id;
    """
    try:
        engine = create_engine(DB_URI, echo=False)
        data = pd.read_sql(query, con=engine)
        logging.info(f"Loaded {len(data)} rows for SHAP analysis.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return ""

    # Define features
    num_cols = ["publish_hour", "likes", "dislikes", "comment_count", "views", "title_len", "tags_count"]
    cat_cols = ["video_category", "emotion"]
    X = data[num_cols + cat_cols].copy()

    # Load trained pipeline
    try:
        clf = joblib.load("outputs/xgb_pipeline.pkl")
    except Exception as e:
        logging.error(f"Failed to load trained model: {e}")
        return ""

    # Transform features
    try:
        X_trans = clf.named_steps["prep"].transform(X)
        feature_names = clf.named_steps["prep"].get_feature_names_out()
    except Exception as e:
        logging.error(f"Feature transformation failed: {e}")
        return ""

    # Compute SHAP values
    explainer = shap.TreeExplainer(clf.named_steps["model"])
    shap_values = explainer.shap_values(X_trans)

    # SHAP bar plot (global importance)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, plot_type="bar", show=False)
    bar64 = fig_to_base64(fig)

    # SHAP beeswarm plot (global distribution)
    fig = plt.figure()
    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    beeswarm64 = fig_to_base64(fig)

    # Assemble HTML
    html = f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <title>SHAP Report</title>
      <style>
        body {{ font-family: sans-serif; margin: 24px; background: #fafafa; }}
        h1 {{ margin-bottom: 12px; }}
        .card {{ background: #fff; border: 1px solid #eee; border-radius: 12px; padding: 16px; margin: 16px 0; }}
        img {{ max-width: 100%; height: auto; display: block; margin: auto; }}
        .caption {{ margin-top: 8px; color: #666; font-size: 14px; }}
      </style>
    </head>
    <body>
      <h1>SHAP Visualization Report</h1>

      <div class="card">
        <h2>Global Importance (Bar Plot)</h2>
        <img src="data:image/png;base64,{bar64}" alt="SHAP summary bar plot">
        <div class="caption">Overall ranking of feature contributions.</div>
      </div>

      <div class="card">
        <h2>Global Distribution (Beeswarm Plot)</h2>
        <img src="data:image/png;base64,{beeswarm64}" alt="SHAP summary beeswarm plot">
        <div class="caption">Feature impact distribution across all samples.</div>
      </div>
    </body>
    </html>
    """

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"SHAP HTML report saved: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Failed to save SHAP report: {e}")
        return ""


def run_shap():
    """Wrapper function for main.py integration."""
    return generate_shap_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_shap_report()
