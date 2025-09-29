import os
import logging
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import DB_URI

#import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def run_with_emotion_model(output_file: str = "outputs/ml_test_predictions_with_emotion.csv",
                           model_name: str = "with_emotion"):
    """Train and evaluate logistic regression model with emotion features.

    Reads cleaned video data joined with emotion labels, preprocesses numeric
    and categorical features, trains a calibrated logistic regression model,
    evaluates performance, and exports predictions and metrics.

    Args:
        output_file (str): Path to save prediction CSV.
        model_name (str): Identifier for storing metrics in the database.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load dataset with emotion join
    query = """
    SELECT 
        v.video_id, v.title, v.video_category, v.likes, v.dislikes,
        v.comment_count, v.views, v.publish_time, v.publish_date,
        v.publish_hour, v.title_len, v.tags_count, v.days_on_trending,
        v.long_term_trending, e.emotion
    FROM us_videos v
    LEFT JOIN video_emotion e
    ON v.video_id = e.video_id;
    """
    try:
        engine = create_engine(DB_URI, echo=False)
        df = pd.read_sql(query, con=engine)
        logging.info(f"Loaded {len(df)} rows from us_videos + emotion.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return {}

    # Features and labels
    num_cols = ["publish_hour", "likes", "dislikes", "comment_count", "views", "title_len", "tags_count"]
    cat_cols = ["video_category", "emotion"]
    X = df[num_cols + cat_cols].copy()
    y = df["long_term_trending"].copy()

    # Preprocessing pipeline
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])

    # Model definition
    base_lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf = Pipeline([
        ("prep", preprocessor),
        ("cal", CalibratedClassifierCV(base_lr, method="sigmoid", cv=5)),
    ])

    # Train-test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, df.index, test_size=0.2, random_state=42, stratify=y
    )
    clf.fit(X_train, y_train)

    # Evaluation
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_prob)
    logging.info(f"ROC AUC: {roc_auc:.3f}")
    logging.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, digits=3)}")

    # Export predictions
    pred_df = df.loc[idx_test, ["video_id", "title", "video_category", "emotion",
                                "likes", "views", "days_on_trending"]].copy()
    pred_df["true_label"] = y_test.values
    pred_df["pred_prob"] = y_prob
    pred_df = pred_df.sort_values("pred_prob", ascending=False)
    pred_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    logging.info(f"Predictions exported: {output_file}")

    # Save metrics
    report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
    metrics = {
        "model": model_name,
        "roc_auc": roc_auc,
        "precision": report_dict["weighted avg"]["precision"],
        "recall": report_dict["weighted avg"]["recall"],
        "f1": report_dict["weighted avg"]["f1-score"],
    }

    try:
        with engine.begin() as conn:
            pd.DataFrame([metrics]).to_sql("model_metrics", con=conn, if_exists="append", index=False)
        logging.info(f"Metrics saved to model_metrics ({model_name})")
    except SQLAlchemyError as e:
        logging.error(f"Failed to save metrics: {e}")

    return metrics


def run_model():
    """Wrapper for main.py integration."""
    return run_with_emotion_model()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_with_emotion_model()
