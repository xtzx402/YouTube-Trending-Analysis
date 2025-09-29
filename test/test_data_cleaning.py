import pandas as pd
from src.data_cleaning import generate_features

def test_generate_features_basic():
    # Basic test with minimal data
    df = pd.DataFrame({
        "video_id": ["v1", "v2"],
        "publish_time": pd.to_datetime(["2021-01-01 12:34:56", "2021-01-02 08:00:00"]),
        "trending_date": pd.to_datetime(["21.01.01", "21.01.02"], format="%y.%d.%m", errors="coerce"),
        "title": ["Test Video 1", "Another Video"],
        "tags_clean": [["tag1", "tag2"], ["tag3"]]
    })

    result = generate_features(df)

    # Check new columns
    assert "publish_hour" in result.columns
    assert "long_term_trending" in result.columns
    assert result.loc[0, "publish_hour"] == 12
    assert result.loc[1, "publish_hour"] == 8

    # long_term_trending: days_on_trending >= 7 
    assert set(result["long_term_trending"].unique()) <= {0, 1}
