import os
import logging
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql import text
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
#from src.config import DB_URI

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_URI

def run_xgboost_with_emotion(output_pred: str = "outputs/ml_test_predictions_xgboost.csv",
                             model_name: str = "xgboost_with_emotion",
                             model_path: str = "outputs/xgb_pipeline.pkl"):
    """Train and evaluate XGBoost model with emotion features.

    Loads cleaned data joined with emotion labels, preprocesses numeric and
    categorical features, trains XGBoost classifier, evaluates performance,
    and stores predictions, metrics, parameters, and feature importance.

    Args:
        output_pred (str): Path to save predictions CSV.
        model_name (str): Model identifier for storing metrics in DB.
        model_path (str): Path to save the trained pipeline.

    Returns:
        dict: Evaluation metrics for the trained model.
    """
    os.makedirs(os.path.dirname(output_pred), exist_ok=True)

    # Load data
    try:
        engine = create_engine(DB_URI, echo=False)
        usvideos = pd.read_sql("SELECT * FROM us_videos", con=engine)
        video_emotion = pd.read_sql("SELECT * FROM video_emotion", con=engine)
        logging.info(f"Loaded {len(usvideos)} cleaned videos and {len(video_emotion)} emotions.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return {}

    # Label generation (safety check)
    if "long_term_trending" not in usvideos.columns:
        usvideos["long_term_trending"] = (usvideos["days_on_trending"] >= 7).astype(int)

    # Merge emotion table
    data = usvideos.merge(video_emotion, on="video_id", how="left")

    # Features and labels
    num_cols = ["publish_hour", "likes", "dislikes", "comment_count", "views", "title_len", "tags_count"]
    cat_cols = ["video_category", "emotion"]
    X = data[num_cols + cat_cols].copy()
    y = data["long_term_trending"].copy()

    # Preprocessing
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
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
    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
    clf = Pipeline([
        ("prep", preprocessor),
        ("model", xgb),
    ])

    # Train-test split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, data.index, test_size=0.2, random_state=42, stratify=y
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
    pred_df = data.loc[idx_test, ["video_id", "title", "video_category", "emotion",
                                  "likes", "views", "days_on_trending"]].copy()
    pred_df["true_label"] = y_test.values
    pred_df["pred_prob"] = y_prob
    pred_df = pred_df.sort_values("pred_prob", ascending=False)
    pred_df.to_csv(output_pred, index=False, encoding="utf-8-sig")
    logging.info(f"Predictions exported: {output_pred}")

    # Save metrics
    report = classification_report(y_test, y_pred, digits=3, output_dict=True)
    metrics = {
        "model": model_name,
        "roc_auc": roc_auc,
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
        "f1": report["weighted avg"]["f1-score"],
    }

    try:
        with engine.begin() as conn:
            pd.DataFrame([metrics]).to_sql("model_metrics", con=conn, if_exists="append", index=False)
        logging.info(f"Metrics saved to DB (model_metrics: {model_name})")
    except SQLAlchemyError as e:
        logging.error(f"Failed to save metrics: {e}")

    # Save model parameters
    try:
        params = xgb.get_xgb_params()
        params_df = pd.DataFrame([params])
        params_df["model"] = model_name
        params_df.to_sql("model_params", con=engine, if_exists="append", index=False)
        logging.info("Model parameters saved to DB (model_params).")
    except SQLAlchemyError as e:
        logging.error(f"Failed to save parameters: {e}")

    # Save feature importance
    try:
        importances = clf.named_steps["model"].feature_importances_
        feature_names = clf.named_steps["prep"].get_feature_names_out()
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df["model"] = model_name
        fi_df.to_sql("feature_importance", con=engine, if_exists="replace", index=False)
        logging.info("Feature importance saved to DB (feature_importance).")
    except SQLAlchemyError as e:
        logging.error(f"Failed to save feature importance: {e}")

    # Save trained pipeline
    joblib.dump(clf, model_path)
    logging.info(f"Trained model saved: {model_path}")

    return metrics


def run_model():
    """Wrapper for main.py integration."""
    return run_xgboost_with_emotion()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_xgboost_with_emotion()
