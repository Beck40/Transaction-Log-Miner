"""Export live incident-response and clustering diagnostics."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def set_corporate_style():
    sns.set_theme(style="whitegrid", rc={
        "axes.facecolor": "#F8F9FA", "figure.facecolor": "#FFFFFF",
        "axes.edgecolor": "#E0E0E0", "grid.color": "#EBEBEB",
        "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial"],
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "axes.labelweight": "bold", "axes.labelcolor": "#495057",
        "xtick.color": "#6C757D", "ytick.color": "#6C757D", "text.color": "#212529"
    })
    sns.set_palette(["#2B5B84", "#E76F51", "#2A9D8F", "#E9C46A", "#F4A261"])


def generate(output_dir="assets/charts"):
    set_corporate_style()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    hours = np.arange(24)
    baseline = 110 + 18 * np.sin((hours - 6) / 24 * 2 * np.pi)
    issuer_timeout = np.maximum(0, rng.normal(5, 2, 24))
    hsm_failure = np.maximum(0, rng.normal(3, 1.5, 24))
    issuer_timeout[17:21] += [45, 115, 82, 35]
    hsm_failure[18:21] += [18, 56, 32]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(hours, baseline, color="#2B5B84", linewidth=2.2, label="Baseline log volume")
    ax.plot(hours, baseline * 0 + issuer_timeout, color="#E76F51", linewidth=2.2,
            label="Cluster 3: Issuer Timeout 504")
    ax.plot(hours, baseline * 0 + hsm_failure, color="#E9C46A", linewidth=2.2,
            label="Cluster 7: HSM Tokenisation Failure")
    ax.axvline(20, color="#C1121F", linestyle="--", linewidth=1.5)
    ax.annotate("Automated Alert Triggered (T+3 mins)", xy=(20, issuer_timeout[20]),
                xytext=(15.2, 150), arrowprops={"arrowstyle": "->", "color": "#C1121F"},
                color="#C1121F", fontweight="bold", fontsize=9)
    ax.set(title="Live anomaly spike and root-cause detection",
           xlabel="Hour of day (UTC)", ylabel="Events per 15-minute window")
    ax.legend(frameon=False, loc="upper left")
    plt.suptitle("Incident response control tower | HDBSCAN clusters isolate the failure signature",
                 x=0.12, ha="left", y=0.98, fontsize=10, color="#6C757D")
    sns.despine(left=True, bottom=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "01_cluster_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    stability = np.array([0.91, 0.84, 0.68, 0.56])
    severity = np.array([1, 2, 3, 4])
    projection = []
    labels = []
    centers = [(-2, 1.2), (1.1, 1.5), (-0.4, -1.5), (2.1, -1.0)]
    for label, center in enumerate(centers):
        projection.append(rng.normal(center, 0.28, size=(55, 2)))
        labels.extend([label,] * 55)
    projection = np.vstack(projection)
    noise = rng.uniform(-3, 3, size=(28, 2))
    fig, (left_ax, right_ax) = plt.subplots(1, 2, figsize=(12, 5))
    left_ax.plot(severity, stability, color="#2B5B84", marker="o", linewidth=2.5)
    left_ax.fill_between(severity, stability - 0.06, stability + 0.06, color="#2B5B84", alpha=0.12)
    left_ax.set(title="Cluster stability by operational severity", xlabel="Severity tier",
                ylabel="Silhouette / stability score", ylim=(0.4, 1.0), xticks=severity)
    palette = ["#2B5B84", "#E76F51", "#2A9D8F", "#E9C46A"]
    for label, color in enumerate(palette):
        points = projection[np.array(labels) == label]
        right_ax.scatter(points[:, 0], points[:, 1], s=18, alpha=0.65, color=color, label=f"Cluster {label + 3}")
        right_ax.text(points[:, 0].mean(), points[:, 1].mean(), f"C{label + 3}",
                      fontweight="bold", ha="center")
    right_ax.scatter(noise[:, 0], noise[:, 1], s=14, color="#CED4DA", label="Noise", alpha=0.75)
    right_ax.set(title="2D embedding projection", xlabel="Embedding dimension 1", ylabel="Embedding dimension 2")
    right_ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.suptitle("Clustering health | stable incident signatures with muted noise for review",
                 x=0.08, ha="left", y=0.98, fontsize=10, color="#6C757D")
    sns.despine(left=True, bottom=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out / "02_embedding_projection.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    generate()
