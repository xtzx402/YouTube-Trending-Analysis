import pytest
import pandas as pd
from src.eda.eda_summary import generate_summary

def test_generate_summary(monkeypatch):
    """Test that generate_summary produces summary_df, toptags, and word_view_df."""

    sample_data = pd.DataFrame({
        "video_id": ["a1", "a2", "a3"],
        "channel_title": ["chan1", "chan2", "chan1"],
        "video_category": ["cat1", "cat2", "cat1"],
        "likes": [10, 50, 0],
        "views": [100, 200, 300],
        "comments_disabled": [0, 1, 0],
        "ratings_disabled": [0, 0, 1],
        "days_on_trending": [5, 10, 7],
        "publish_hour": [1, 2, 3],
        "title": ["Hello World", "Another test", "Third video"],
        "tags_clean": ["tag1,tag2", "tag2", "tag3"]
    })

    def fake_read_sql(query, con):
        return sample_data

    monkeypatch.setattr("pandas.read_sql", fake_read_sql)


    summary_df, toptags, word_view_df = generate_summary()

    assert isinstance(summary_df, pd.DataFrame)
    assert "total_videos" in summary_df.columns
    assert summary_df["total_videos"].iloc[0] == 3

    assert isinstance(toptags, pd.Series)
    assert not toptags.empty
    assert "tag2" in toptags.index  

    assert isinstance(word_view_df, pd.DataFrame)
    assert "word" in word_view_df.columns
    assert "avg_views" in word_view_df.columns
