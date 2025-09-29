import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
import json

from src.config import DB_URI

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def generate_model_comparison(output_file: str = "outputs/model_comparison.html"):
    """Generate HTML comparison report for three ML models.

    Reads model metrics from database and creates an interactive
    Chart.js visualization (bar + line charts) comparing performance.

    Args:
        output_file (str): Path to save the HTML report.

    Returns:
        str: Path to the generated HTML file, or empty string if failed.
    """
    logging.info("Starting model comparison report generation...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load model metrics from DB
    try:
        engine = create_engine(DB_URI, echo=False)
        df = pd.read_sql("SELECT * FROM model_metrics", con=engine)
        logging.info(f"Loaded {len(df)} rows from model_metrics table.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load model_metrics: {e}")
        return ""

    try:
        baseline = df[df.model == "baseline_no_emotion"].iloc[0]
        emotion = df[df.model == "with_emotion"].iloc[0]
        xgb_emotion = df[df.model == "xgboost_with_emotion"].iloc[0]
    except IndexError:
        logging.error("Missing expected model rows in model_metrics table.")
        return ""

    metrics = ["roc_auc", "precision", "recall", "f1"]
    print(df[['model', 'roc_auc', 'precision', 'recall', 'f1']])

    baseline_vals = [float(round(baseline[m], 3)) for m in metrics]
    emotion_vals = [float(round(emotion[m], 3)) for m in metrics]
    xgb_emotion_vals = [float(round(xgb_emotion[m], 3)) for m in metrics]

    # Convert to JSON for embedding
    baseline_js = json.dumps(baseline_vals)
    emotion_js = json.dumps(emotion_vals)
    xgb_emotion_js = json.dumps(xgb_emotion_vals)

    # Determine y-axis range for line chart
    all_vals = baseline_vals + emotion_vals + xgb_emotion_vals
    y_min = max(0.0, round(min(all_vals) - 0.01, 3))
    y_max = min(1.0, round(max(all_vals) + 0.01, 3))
    if y_max - y_min < 0.02:  
        mid = (y_min + y_max) / 2
        y_min = max(0.0, round(mid - 0.01, 3))
        y_max = min(1.0, round(mid + 0.01, 3))

    # Build HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Model Comparison (3 Models)</title>
      <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
      <style>
        body {{
          font-family: Arial, sans-serif;
          margin: 40px;
          background: #fdfdfd;
        }}
        h1 {{
          text-align: center;
          color: #333;
        }}
        .chart-container {{
          width: 80%;
          margin: 40px auto;
        }}
      </style>
    </head>
    <body>
      <h1>Model Comparison (BaseCase vs With Emotion vs XGBoost+Emotion)</h1>

      <div class="chart-container">
        <h2 style="text-align:center;">Bar Chart Comparison</h2>
        <canvas id="barChart"></canvas>
      </div>

      <div class="chart-container">
        <h2 style="text-align:center;">Line Chart Trend</h2>
        <canvas id="lineChart"></canvas>
      </div>

      <script>
        const labels = ["ROC AUC", "Precision", "Recall", "F1"];
        const baseline = {baseline_js};
        const withEmotion = {emotion_js};
        const xgbEmotion = {xgb_emotion_js};

        // Bar chart
        new Chart(document.getElementById("barChart"), {{
          type: "bar",
          data: {{
            labels: labels,
            datasets: [
              {{ label: "BaseCase", data: baseline, backgroundColor: "rgba(54, 162, 235, 0.6)" }},
              {{ label: "With Emotion", data: withEmotion, backgroundColor: "rgba(255, 99, 132, 0.6)" }},
              {{ label: "XGBoost+Emotion", data: xgbEmotion, backgroundColor: "rgba(255, 206, 86, 0.6)" }}
            ]
          }},
          options: {{ 
            responsive: true, 
            scales: {{ 
              y: {{ beginAtZero: true, max: 1 }} 
            }} 
          }}
        }});

        // Line chart with dynamic y-axis range
        new Chart(document.getElementById("lineChart"), {{
          type: "line",
          data: {{
            labels: labels,
            datasets: [
              {{ label: "BaseCase", data: baseline, borderColor: "rgba(54, 162, 235, 1)", fill: false, tension: 0.1 }},
              {{ label: "With Emotion", data: withEmotion, borderColor: "rgba(255, 99, 132, 1)", fill: false, tension: 0.1 }},
              {{ label: "XGBoost+Emotion", data: xgbEmotion, borderColor: "rgba(255, 206, 86, 1)", fill: false, tension: 0.1 }}
            ]
          }},
          options: {{ 
            responsive: true, 
            scales: {{ 
              y: {{
                min: {y_min},
                max: {y_max}
              }}
            }} 
          }}
        }});
      </script>

    </body>
    </html>
    """

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        logging.info(f"Model comparison report saved: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Failed to save report: {e}")
        return ""


def run_model_comparison():
    """Wrapper for main.py to run model comparison task."""
    return generate_model_comparison()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_model_comparison()