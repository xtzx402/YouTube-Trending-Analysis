# main.py
import argparse
import logging

from src.data_cleaning import run_data_cleaning
from src.ai_sentiment_analysis import run_sentiment
from src.eda.eda_distribution import run_eda_distribution
from src.eda.eda_summary import run_eda
from src.eda.eda_visualization import run_eda_visualization
from src.ml_models.ml_modeling_basecase import run_model as run_baseline
from src.ml_models.ml_modeling_with_emotion import run_model as run_with_emotion
from src.ml_models.ml_modeling_xgboost import run_model as run_xgb_emotion
from src.ml_models.ml_modeling_shap import run_shap
from src.ml_models.compare_models_visual import run_model_comparison


def main():
    parser = argparse.ArgumentParser(
        description="YouTube Trending Data Analysis Pipeline"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "clean_data",
            "sentiment",
            "distribution",
            "eda_summary",
            "eda_visual",
            "baseline_model",
            "with_emotion_model",
            "xgb_model",
            "shap",
            "model_comparison",
            "full_pipeline",
        ],
        help="Task to run in the pipeline",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    if args.task == "clean_data":
        run_data_cleaning()
    elif args.task == "sentiment":
        run_sentiment()
    elif args.task == "distribution":
        run_eda_distribution()
    elif args.task == "eda_summary":
        run_eda()
    elif args.task == "eda_visual":
        run_eda_visualization()
    elif args.task == "baseline_model":
        run_baseline()
    elif args.task == "with_emotion_model":
        run_with_emotion()
    elif args.task == "xgb_model":
        run_xgb_emotion()
    elif args.task == "shap":
        run_shap()
    elif args.task == "model_comparison":
        run_model_comparison()
    elif args.task == "full_pipeline":
        logging.info(
            "Running full pipeline: clean_data -> sentiment -> distribution -> eda -> models -> shap -> comparison"
        )
        run_data_cleaning()
        run_sentiment()
        run_eda_distribution()
        run_eda()
        run_eda_visualization()
        run_baseline()
        run_with_emotion()
        run_xgb_emotion()
        run_shap()
        run_model_comparison()


if __name__ == "__main__":
    main()
