import os
import pandas as pd
import pytest
from src.eda.eda_distribution import run_distribution_analysis

def test_run_distribution_analysis(monkeypatch, tmp_path):
    sample_data = pd.DataFrame({
        "video_id": ["a1", "a2", "a3", "a4"],
        "likes": [10, 1000000, 5, 7],   
        "views": [50, 999999, 30, 25]  
    })

    def fake_load_data():
        return sample_data

    monkeypatch.setattr("src.eda.eda_distribution.load_data", fake_load_data)

    output_file = tmp_path / "distribution_test.xlsx"
    result_path = run_distribution_analysis(output_file=str(output_file))

    assert os.path.exists(result_path), "Excel file should be created"

    xl = pd.ExcelFile(result_path)
    assert "Cleaned_Data" in xl.sheet_names
    assert "High_Likes" in xl.sheet_names
    assert "Low_Likes" in xl.sheet_names
    assert "High_Views" in xl.sheet_names
    assert "Low_Views" in xl.sheet_names

    df_high_likes = pd.read_excel(result_path, sheet_name="High_Likes")
    assert not df_high_likes.empty
