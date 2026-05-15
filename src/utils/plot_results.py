import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load detailed CSV
df = pd.read_csv("results/results_summary.csv")

# Add mean column across the 3 rounds
df["mean_rate"] = df[["round1_rate", "round2_rate", "round3_rate"]].mean(axis=1)

# -----------------------------
# Grouped bar chart
# -----------------------------
websites = df["website"].tolist()
x = np.arange(len(websites))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - 1.5 * width, df["round1_rate"], width, label="Round 1")
bars2 = ax.bar(x - 0.5 * width, df["round2_rate"], width, label="Round 2")
bars3 = ax.bar(x + 0.5 * width, df["round3_rate"], width, label="Round 3")
bars4 = ax.bar(
    x + 1.5 * width,
    df["mean_rate"],
    width,
    label="Mean",
    edgecolor="black",
    linewidth=1.5,
    hatch="//"
)

ax.set_title("Trigger Disruption Rate by Website Across Three Test Rounds")
ax.set_xlabel("Website")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 100)
ax.set_xticks(x)
ax.set_xticklabels(websites, rotation=20)
ax.legend()

plt.tight_layout()
plt.savefig("results/trigger_disruption_bar_chart.png", dpi=200, bbox_inches="tight")
plt.close()

# -----------------------------
# Line chart
# -----------------------------
round_labels = ["Round 1", "Round 2", "Round 3", "Mean"]
round_cols = ["round1_rate", "round2_rate", "round3_rate", "mean_rate"]

fig, ax = plt.subplots(figsize=(10, 6))

for _, row in df.iterrows():
    y = [row[col] for col in round_cols]
    if row["website"].lower() == "overall":
        ax.plot(
            round_labels,
            y,
            marker="o",
            linewidth=3,
            markersize=8,
            linestyle="--",
            label=row["website"],
        )
    else:
        ax.plot(
            round_labels,
            y,
            marker="o",
            linewidth=1.5,
            label=row["website"],
        )

ax.set_title("Trigger Disruption Trends Across Three Test Rounds")
ax.set_xlabel("Test Round")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 100)
ax.legend()

plt.tight_layout()
plt.savefig("results/trigger_disruption_line_chart.png", dpi=200, bbox_inches="tight")
plt.close()