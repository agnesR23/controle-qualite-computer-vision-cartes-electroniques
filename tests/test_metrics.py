"""
Tests for model evaluation metrics.

These tests validate the CSV files used by the Streamlit dashboard:
- benchmark results;
- YOLO tuning results;
- metric ranges;
- consistency of the selected best model.

They ensure that the dashboard is based on valid and exploitable results.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_DIR = PROJECT_ROOT / "app" / "assets"
BENCHMARK_RESULTS_PATH = METRICS_DIR / "benchmark_results.csv"
YOLO_TUNING_RESULTS_PATH = METRICS_DIR / "yolo_tuning_results.csv"

REQUIRED_COLUMNS = [
    "model_name",
    "map50_95",
    "map50",
    "training_time",
    "inference_time",
]


def load_results(path: Path) -> pd.DataFrame:
    """Load a metrics CSV file."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def test_benchmark_results_file_exists():
    """Check that benchmark results are available."""
    assert BENCHMARK_RESULTS_PATH.exists()


def test_yolo_tuning_results_file_exists():
    """Check that YOLO tuning results are available."""
    assert YOLO_TUNING_RESULTS_PATH.exists()


def test_metrics_files_have_required_columns():
    """Check that metrics files contain the columns used by the dashboard."""
    for path in [BENCHMARK_RESULTS_PATH, YOLO_TUNING_RESULTS_PATH]:
        df = load_results(path)

        for column in REQUIRED_COLUMNS:
            assert column in df.columns


def test_metrics_values_are_in_valid_ranges():
    """Check that mAP scores and runtime values are valid."""
    for path in [BENCHMARK_RESULTS_PATH, YOLO_TUNING_RESULTS_PATH]:
        df = load_results(path)

        assert df["map50_95"].between(0, 1).all()
        assert df["map50"].between(0, 1).all()
        assert (df["training_time"] > 0).all()
        assert (df["inference_time"] > 0).all()


def test_best_model_is_yolo_tuned_run():
    """Check that the best global result is a YOLO run."""
    benchmark_df = load_results(BENCHMARK_RESULTS_PATH)
    tuning_df = load_results(YOLO_TUNING_RESULTS_PATH)

    comparison_df = (
        pd.concat([benchmark_df, tuning_df], ignore_index=True)
        .drop_duplicates(subset="model_name")
        .sort_values("map50_95", ascending=False)
        .reset_index(drop=True)
    )

    best_model_name = comparison_df.iloc[0]["model_name"]

    assert "yolov8s" in best_model_name