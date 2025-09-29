import pandas as pd
import json
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.config import DB_URI

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def load_raw_data(video_path: str, category_path: str):
    """Load raw YouTube trending video data and category mapping.

    Args:
        video_path (str): Path to the CSV file containing video metadata.
        category_path (str): Path to the JSON file with category mapping.

    Returns:
        tuple[pd.DataFrame, dict]: Raw video DataFrame and category mapping dictionary.
    """
    df = pd.read_csv(video_path)
    with open(category_path, "r") as f:
        categories = json.load(f)

    dict_category = {i["id"]: i["snippet"]["title"] for i in categories["items"]}
    return df, dict_category


def clean_data(df: pd.DataFrame, dict_category: dict) -> pd.DataFrame:
    """Clean raw YouTube data (deduplication, text cleaning, tags, category mapping).

    Args:
        df (pd.DataFrame): Raw video DataFrame.
        dict_category (dict): Mapping from category_id to category name.

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized fields.
    """
    df = df.copy().drop_duplicates()

    # Convert datetime fields
    df["publish_time"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m", errors="coerce")

    # Map category_id -> video_category
    df["category_id"] = df["category_id"].astype(str)
    df["video_category"] = df["category_id"].map(dict_category)

    # Clean text fields
    df["title"] = df["title"].fillna("")
    df["description"] = (
        df["description"].fillna("")
        .str.lower()
        .str.replace(r"<[^<>]*>", "", regex=True)  # remove HTML tags
        .str.replace(r"http\S+", "", regex=True)  # remove URLs
        .str.replace("\n", " ", regex=False)      # normalize newlines
        .str.replace("\r", " ", regex=False)
        .str.replace(r"[^\x00-\x7F]+", "", regex=True)  # remove non-ASCII
        .str.replace(r"\\n", " ", regex=True)
    )

    # Normalize tags
    df["tags_clean"] = df["tags"].fillna("").apply(
        lambda s: [t.strip().lower() for t in str(s).split("|") if t.strip() and t.strip().lower() != "none"]
    )
    df["tags_count"] = df["tags_clean"].apply(len)

    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate derived features such as publish_date, title length, and long-term trending flag.

    Args:
        df (pd.DataFrame): Cleaned DataFrame.

    Returns:
        pd.DataFrame: DataFrame with additional engineered features.
    """
    # Count number of unique trending days per video
    days_on_trending = df.groupby("video_id")["trending_date"].nunique().rename("days_on_trending")

    # Keep first trending snapshot for each video
    first_snap = (
        df.sort_values("trending_date")
        .drop_duplicates("video_id", keep="first")
        .merge(days_on_trending, on="video_id", how="left")
        .reset_index(drop=True)
    )

    # Derived features
    first_snap["publish_date"] = first_snap["publish_time"].dt.date
    first_snap["publish_hour"] = first_snap["publish_time"].dt.hour
    first_snap["title_len"] = first_snap["title"].fillna("").str.len()
    first_snap["long_term_trending"] = (first_snap["days_on_trending"] >= 7).astype(int)
    first_snap["tags_clean"] = first_snap["tags_clean"].apply(
        lambda x: ",".join(x) if isinstance(x, list) else ""
    )
    return first_snap


def save_to_db(df: pd.DataFrame, table_name: str = "us_videos", chunksize: int = 1000):
    """Persist DataFrame to a SQL database in chunked batches.

    Args:
        df (pd.DataFrame): Final cleaned and feature-engineered dataset.
        table_name (str, optional): Target database table. Defaults to "us_videos".
        chunksize (int, optional): Batch size for insertion. Defaults to 1000.
    """
    engine = create_engine(DB_URI, echo=False)
    total_rows = len(df)
    start = 0

    while start < total_rows:
        end = min(start + chunksize, total_rows)
        chunk = df.iloc[start:end]
        try:
            chunk.to_sql(
                table_name,
                con=engine,
                if_exists="append" if start > 0 else "replace",
                index=False,
            )
            logging.info(f"Inserted rows {start} - {end}")
        except SQLAlchemyError as e:
            logging.error(f"Failed to insert rows {start} - {end}: {e}")
        start = end


def run_data_cleaning():
    """Full pipeline: load raw data, clean, engineer features, and save to DB."""
    logging.info("Starting data cleaning pipeline...")

    # Step 1. Load raw data
    raw_df, dict_category = load_raw_data("datasource/USvideos.csv", "datasource/US_category_id.json")

    # Step 2. Clean data
    cleaned_df = clean_data(raw_df, dict_category)

    # Step 3. Generate engineered features
    features_df = generate_features(cleaned_df)

    # Step 4. Save to database
    save_to_db(features_df)

    logging.info("Data cleaning pipeline finished. Final table: us_videos")
    return features_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_data_cleaning()
