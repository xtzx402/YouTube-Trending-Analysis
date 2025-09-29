import pandas as pd
import numpy as np
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
#from src.config import DB_URI

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DB_URI


def load_data():
    """Load cleaned YouTube video data from the database.

    Returns:
        pd.DataFrame: DataFrame containing cleaned video records,
        or an empty DataFrame if loading fails.
    """
    try:
        engine = create_engine(DB_URI, echo=False)
        df = pd.read_sql("SELECT * FROM us_videos", con=engine)
        logging.info(f"Cleaned data loaded successfully: {len(df)} rows.")
        return df
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return pd.DataFrame()


def z_score_outliers(
    series: pd.Series, high: float = None, low: float = None, log_transform: bool = False
) -> pd.Index:
    """Detect outlier indices in a numeric series using Z-score method.

    Args:
        series (pd.Series): Numeric series to evaluate.
        high (float, optional): Upper Z-score threshold.
        low (float, optional): Lower Z-score threshold.
        log_transform (bool, optional): Whether to apply log1p transform before calculation.

    Returns:
        pd.Index: Index of rows considered outliers.
    """
    data = np.log1p(series) if log_transform else series
    if data.std() == 0:
        return pd.Index([])

    z_scores = (data - data.mean()) / data.std()
    mask = pd.Series(False, index=series.index)

    if high is not None:
        mask |= z_scores >= high
    if low is not None:
        mask |= z_scores <= low

    return series[mask].index


def run_distribution_analysis(output_file: str = "outputs/distribution_analysis.xlsx") -> str:
    """Perform distribution analysis (outlier detection) and export results to Excel.

    Args:
        output_file (str, optional): Path to output Excel file.
            Defaults to "outputs/distribution_analysis.xlsx".

    Returns:
        str: Path to the saved Excel file, or empty string if failed.
    """
    logging.info("Starting distribution analysis...")
    usvideos = load_data()
    if usvideos.empty:
        logging.error("No data available for distribution analysis.")
        return ""

    # Outlier detection
    outliers_likes_high = usvideos.loc[z_score_outliers(usvideos["likes"], high=1, log_transform=True)]
    outliers_likes_low = usvideos.loc[z_score_outliers(usvideos["likes"], low=-1, log_transform=True)]
    outliers_views_high = usvideos.loc[z_score_outliers(usvideos["views"], high=3, log_transform=True)]
    outliers_views_low = usvideos.loc[z_score_outliers(usvideos["views"], low=-3, log_transform=True)]

    # Export results
    tabs = {
        "High_Likes": outliers_likes_high,
        "Low_Likes": outliers_likes_low,
        "High_Views": outliers_views_high,
        "Low_Views": outliers_views_low,
    }

    try:
        with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
            usvideos.to_excel(writer, sheet_name="Cleaned_Data", index=False)
            for name, df_out in tabs.items():
                if df_out.empty:
                    df_out = pd.DataFrame({"note": ["no outliers found"]})
                df_out.to_excel(writer, sheet_name=name, index=False)
        logging.info(f"Distribution analysis export completed: {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Failed to export distribution analysis: {e}")
        return ""


def run_eda_distribution() -> str:
    """Wrapper for main.py to run distribution analysis.

    Returns:
        str: Path to the saved Excel file, or empty string if failed.
    """
    return run_distribution_analysis()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_distribution_analysis()
