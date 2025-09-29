import pandas as pd
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from collections import defaultdict
from src.config import DB_URI

#import sys, os
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def generate_summary():
    """Generate EDA summary statistics from cleaned video dataset.

    Loads cleaned data from the database, computes various summary
    statistics, top tags, and title word analysis, then saves results
    back to the database.

    Returns:
        tuple:
            summary_df (pd.DataFrame): One-row summary table with key statistics.
            toptags (pd.Series): Top 20 most frequent tags.
            word_view_df (pd.DataFrame): Average views per unique word in titles.
    """
    logging.info("Starting EDA summary")

    # 1. Data Loading
    try:
        engine = create_engine(DB_URI, echo=False)
        df = pd.read_sql("SELECT * FROM us_videos", con=engine)
        logging.info(f"Cleaned data loaded successfully: {len(df)} rows.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return None, None, None

    df_unique = df.copy()

    # 2. Summary Statistics
    total_videos = len(df_unique)
    unique_videos = df_unique["video_id"].nunique()
    unique_channels = df_unique["channel_title"].nunique()
    number_of_category = df_unique["video_category"].nunique()

    # 3. Video Performance
    max_likes = df_unique.loc[df_unique["likes"].idxmax()]
    min_likes = df_unique.loc[df_unique["likes"].idxmin()]
    max_views = df_unique.loc[df_unique["views"].idxmax()]
    min_views = df_unique.loc[df_unique["views"].idxmin()]

    likes_avg = df_unique["likes"].mean()
    likes_std = df_unique["likes"].std()
    views_avg = df_unique["views"].mean()
    views_std = df_unique["views"].std()

    # 4. Channel & Category Analysis
    channel_posts = df_unique["channel_title"].value_counts()
    popular_category = df_unique["video_category"].value_counts()

    # 5. Trending Dynamics
    max_days = df_unique["days_on_trending"].max()
    min_days = df_unique["days_on_trending"].min()
    mean_days = df_unique["days_on_trending"].mean()
    mode_days = df_unique["days_on_trending"].mode()[0]

    # 6. Upload Behavior
    most_common_hour = df_unique["publish_hour"].mode()[0]

    # 7. Restrictions
    comments_disabled_pct = df_unique["comments_disabled"].mean() * 100
    ratings_disabled_pct = df_unique["ratings_disabled"].mean() * 100

    # 8. Tags Analysis
    toptags = (
        df_unique["tags_clean"]
        .dropna()
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(20)
    )

    # 9. Title Word Analysis
    df_unique["title_clean"] = (
        df_unique["title"]
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
    )

    word_views = defaultdict(int)
    word_counts = defaultdict(int)
    for title, views in zip(df_unique["title_clean"], df_unique["views"]):
        words = set(title.split())
        for word in words:
            word_views[word] += views
            word_counts[word] += 1

    word_avg_views = {
        word: word_views[word] / word_counts[word] for word in word_counts
    }
    word_view_df = (
        pd.DataFrame(list(word_avg_views.items()), columns=["word", "avg_views"])
        .round({"avg_views": 0})
        .sort_values(by="avg_views", ascending=False)
    )

    # 10. Save summary results
    summary_data = {
        "Region": "US",
        "total_videos": total_videos,
        "unique_videos": unique_videos,
        "unique_channels": unique_channels,
        "number_of_category": number_of_category,
        "avg_views": round(views_avg, 2),
        "std_views": round(views_std, 2),
        "avg_likes": round(likes_avg, 2),
        "std_likes": round(likes_std, 2),
        "max_likes_title": max_likes["title"],
        "max_likes_count": int(max_likes["likes"]),
        "min_likes_title": min_likes["title"],
        "min_likes_count": int(min_likes["likes"]),
        "max_views_title": max_views["title"],
        "max_views_count": int(max_views["views"]),
        "min_views_title": min_views["title"],
        "min_views_count": int(min_views["views"]),
        "max_trending_days": int(max_days),
        "min_trending_days": int(min_days),
        "avg_trending_days": round(mean_days, 2),
        "mode_trending_days": int(mode_days),
        "most_common_upload_hour": int(most_common_hour),
        "comments_disabled_pct": round(comments_disabled_pct, 2),
        "ratings_disabled_pct": round(ratings_disabled_pct, 2),
    }

    summary_df = pd.DataFrame([summary_data])

    try:
        summary_df.to_sql("video_summary", con=engine, if_exists="replace", index=False)
        logging.info("Summary statistics saved to MySQL (table: video_summary)")
    except SQLAlchemyError as e:
        logging.error(f"Failed to save summary: {e}")

    return summary_df, toptags, word_view_df


def run_eda():
    """Wrapper function to execute EDA summary for main.py."""
    return generate_summary()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_summary()
