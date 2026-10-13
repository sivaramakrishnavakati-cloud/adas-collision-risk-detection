from pathlib import Path

import pandas as pd


CSV_PATH = Path("outputs/approach_indicator_log.csv")
REPORT_PATH = Path("outputs/evaluation_report.txt")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH}"
        )

    df = pd.read_csv(CSV_PATH)

    required_columns = {
        "frame",
        "track_id",
        "vehicle_class",
        "confidence",
        "bbox_area",
        "area_growth_ratio",
        "indicator_status",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    lines = []

    lines.append("ADAS PROJECT - EVALUATION REPORT")
    lines.append("=" * 40)
    lines.append(f"Total observations: {len(df)}")
    lines.append(
        f"Unique tracking IDs: {df['track_id'].nunique()}"
    )
    lines.append(
        f"Unique video frames: {df['frame'].nunique()}"
    )
    lines.append(
        f"Duplicate rows: {df.duplicated().sum()}"
    )

    lines.append("\nINDICATOR STATUS COUNTS")
    lines.append("-" * 40)
    lines.append(
        df["indicator_status"]
        .value_counts()
        .to_string()
    )

    lines.append("\nVEHICLE CLASS SUMMARY")
    lines.append("-" * 40)

    class_summary = pd.crosstab(
        df["vehicle_class"],
        df["indicator_status"],
    )

    lines.append(class_summary.to_string())

    lines.append("\nGROWTH RATIO STATISTICS")
    lines.append("-" * 40)

    growth_summary = (
        df.groupby("indicator_status")["area_growth_ratio"]
        .agg(["count", "mean", "median", "max"])
        .round(2)
    )

    lines.append(growth_summary.to_string())

    lines.append("\nDATA QUALITY CHECK")
    lines.append("-" * 40)
    lines.append(
        f"Missing values:\n{df.isnull().sum().to_string()}"
    )

    lines.append(
        "\nLIMITATION: Bounding-box growth is an experimental "
        "image-based indicator. It is not a validated "
        "collision prediction or calibrated time-to-collision."
    )

    report = "\n".join(lines)

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(report)
    print(f"\nReport saved to: {REPORT_PATH.resolve()}")


if __name__ == "__main__":
    main()