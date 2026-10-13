from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("outputs/approach_indicator_log.csv")
OUTPUT_DIR = Path("outputs")

df = pd.read_csv(CSV_PATH)

# Chart 1: Count of each indicator status.
status_counts = df["indicator_status"].value_counts()

plt.figure(figsize=(9, 5))
status_counts.plot(kind="bar")
plt.title("ADAS Experimental Indicator Observations")
plt.xlabel("Indicator Status")
plt.ylabel("Number of Observations")
plt.xticks(rotation=15)
plt.tight_layout()

chart_path = OUTPUT_DIR / "indicator_status_chart.png"
plt.savefig(chart_path, dpi=150)
plt.close()

# Chart 2: Indicator status by vehicle class.
class_summary = pd.crosstab(
    df["vehicle_class"],
    df["indicator_status"],
)

class_summary.plot(kind="bar", figsize=(10, 6))
plt.title("Indicator Observations by Vehicle Class")
plt.xlabel("Vehicle Class")
plt.ylabel("Number of Observations")
plt.xticks(rotation=0)
plt.tight_layout()

class_chart_path = OUTPUT_DIR / "vehicle_class_chart.png"
plt.savefig(class_chart_path, dpi=150)
plt.close()

print("Charts created successfully!")
print(f"Status chart: {chart_path.resolve()}")
print(f"Vehicle class chart: {class_chart_path.resolve()}")