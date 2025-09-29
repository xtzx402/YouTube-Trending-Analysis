import pandas as pd
import matplotlib.pyplot as plt
import io, base64, os
import plotly.express as px
import plotly.io as pio
from wordcloud import WordCloud
import logging
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from src.config import DB_URI

# Allow running this file directly without main.py
#import sys
#sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from config import DB_URI


def fig_to_base64(fig):
    """Convert a matplotlib figure to base64-encoded <img> tag."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{img_base64}" />'


def plotly_to_html(fig):
    """Convert a Plotly figure to embeddable HTML fragment."""
    return pio.to_html(fig, full_html=False)


def generate_visual_report(output_file: str = "outputs/eda_analysis_report.html"):
    """Generate a styled HTML report with EDA visualizations.

    Loads cleaned data from the database, computes summary statistics,
    generates multiple visualizations (matplotlib, Plotly, wordcloud),
    and exports them into a single styled HTML report.

    Args:
        output_file (str): Path to save the HTML report.

    Returns:
        str: Path to the generated HTML report.
    """
    logging.info("Starting EDA visualization report...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Load cleaned dataset
    try:
        engine = create_engine(DB_URI, echo=False)
        df_unique = pd.read_sql("SELECT * FROM us_videos", con=engine)
        logging.info(f"Loaded {len(df_unique)} rows from database.")
    except SQLAlchemyError as e:
        logging.error(f"Failed to load data: {e}")
        return ""

    # Summary statistics table
    summary_stats = {
        "Likes Mean": df_unique["likes"].mean(),
        "Likes Std": df_unique["likes"].std(),
        "Views Mean": df_unique["views"].mean(),
        "Views Std": df_unique["views"].std(),
        "Total Videos": len(df_unique),
        "Unique Channels": df_unique["channel_title"].nunique(),
    }
    summary_html = "<table>"
    for k, v in summary_stats.items():
        summary_html += f"<tr><th>{k}</th><td>{v:.2f}</td></tr>"
    summary_html += "</table>"

    html_parts = []

    # Upload hour distribution
    upload_hour_counts = df_unique["publish_hour"].value_counts().sort_index()
    fig, ax = plt.subplots()
    ax.bar(upload_hour_counts.index, upload_hour_counts.values, color="#CCCCCC")
    ax.set_title("Video Uploads by Hour")
    html_parts.append("<h2>Upload Hour</h2>" + fig_to_base64(fig))

    # Views vs upload hour
    fig, ax = plt.subplots()
    df_unique.plot(kind="scatter", x="publish_hour", y="views", color="#FF9999", ax=ax)
    ax.set_title("Views by Upload Hour")
    html_parts.append("<h2>Views vs Upload Hour</h2>" + fig_to_base64(fig))

    # Trending days distribution
    trending_days_per_video = df_unique.set_index("video_id")["days_on_trending"]
    fig, ax = plt.subplots()
    ax.hist(
        trending_days_per_video.dropna(),
        bins=range(1, int(trending_days_per_video.max()) + 1),
        color="#CCCCCC", edgecolor="grey"
    )
    ax.set_title("Distribution of Trending Days per Video")
    html_parts.append("<h2>Trending Days Distribution</h2>" + fig_to_base64(fig))

    # Average trending days by category
    avg_days_per_category = (
        df_unique.groupby("video_category")["days_on_trending"]
        .mean()
        .round(2)
        .sort_values(ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    avg_days_per_category.plot(kind="bar", color="#FF9999", ax=ax)
    ax.set_title("Average Trending Days by Category")
    html_parts.append("<h2>Average Trending Days by Category</h2>" + fig_to_base64(fig))

    # Bubble chart: category volume vs trending duration
    popular_category = df_unique["video_category"].value_counts()
    df_category_trending = pd.DataFrame({
        "video_category": popular_category.index,
        "count": popular_category.values,
        "trending_days_per_video": avg_days_per_category.reindex(popular_category.index).values
    })
    fig = px.scatter(
        df_category_trending,
        x="trending_days_per_video", y="count",
        size="count", color="video_category", text="video_category",
        labels={"trending_days_per_video": "Avg. Trending Days", "count": "Total Trending Videos"},
        title="Category Trending Duration vs Video Volume"
    )
    html_parts.append("<h2>Bubble Chart</h2>" + plotly_to_html(fig))

    # Category distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    df_unique.groupby("video_category")["video_id"].count().sort_values().plot(
        kind="barh", color="#FF9999", ax=ax
    )
    ax.set_title("Number of Videos per Category")
    html_parts.append("<h2>Category Distribution</h2>" + fig_to_base64(fig))

    # Category views & likes
    category_view_like = df_unique.groupby("video_category")[["views", "likes"]].sum()
    category_view_like["video_count"] = df_unique.groupby("video_category")["video_id"].count()
    category_view_like["views"] = category_view_like["views"] / 1e7
    category_view_like["likes"] = category_view_like["likes"] / 1e5
    fig, ax = plt.subplots(figsize=(12, 10))
    category_view_like.sort_values("video_count").plot(
        kind="barh", ax=ax, color=["#FF0000", "#CCCCCC", "#2C2C2C"], width=0.8
    )
    ax.set_title("Views, Likes, and Counts per Category")
    html_parts.append("<h2>Category Views & Likes</h2>" + fig_to_base64(fig))

    # Audience opinion (likes vs dislikes)
    total_likes = df_unique["likes"].fillna(0).sum()
    total_dislikes = df_unique["dislikes"].fillna(0).sum()
    fig, ax = plt.subplots()
    ax.pie([total_likes, total_dislikes], labels=["Likes", "Dislikes"],
           colors=["#FF9999", "#CCCCCC"], autopct="%1.1f%%")
    ax.set_title("Overall Like vs Dislike Ratio")
    html_parts.append("<h2>Audience Opinion</h2>" + fig_to_base64(fig))

    # Wordcloud of tags
    all_tags = df_unique["tags_clean"].dropna().astype(str).str.split(",")
    tags_flat = [tag.strip() for sublist in all_tags for tag in sublist if tag.strip()]
    tag_counts = pd.Series(tags_flat).value_counts().head(50)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(tag_counts.to_dict())
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Most Frequent Tags")
    html_parts.append("<h2>Video Tags</h2>" + fig_to_base64(fig))

    # Top 5 channels by average trending days
    top_channels = (
        df_unique.groupby("channel_title")["days_on_trending"]
        .mean()
        .nlargest(5)
        .reset_index()
    )
    top_channels_html = top_channels.to_html(index=False, justify="center", border=0)

    # Assemble final HTML
    html_report = f"""
    <html>
    <head>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <meta charset="utf-8">
    <title>YouTube Trending Analysis Report</title>
    <style>
      body {{
        font-family: Arial, sans-serif;
        margin: 20px;
        background-color: #f9f9f9;
        color: #333;
      }}
      h1 {{ text-align: center; }}
      h2 {{ margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
      table th, table td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
      table th {{ background-color: #f2f2f2; }}
      img {{ max-width: 100%; height: auto; margin: 10px 0; }}
      .section {{ background: #fff; padding: 20px; margin-bottom: 30px; border-radius: 6px;
                  box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
    </style>
    </head>
    <body>
    <h1>YouTube Trending Video Analysis Report</h1>

    <div class="section">
    <h2>Summary Statistics</h2>
    {summary_html}
    </div>

    <div class="section">
    <h2>Top 5 Channels by Average Trending Days</h2>
    {top_channels_html}
    </div>

    <div class="section">
    {''.join(html_parts)}
    </div>

    </body>
    </html>
    """

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_report)

    logging.info(f"Styled EDA report generated: {output_file}")
    return output_file


def run_eda_visualization():
    """Wrapper function to run EDA visualization (for main.py)."""
    return generate_visual_report()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generate_visual_report()
