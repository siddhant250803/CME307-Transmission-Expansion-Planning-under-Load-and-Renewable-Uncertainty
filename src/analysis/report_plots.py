"""Generate publication-ready plots for the final report from existing CSV outputs."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk", font_scale=0.9)


def _save(fig, path: Path) -> None:
    """Tight layout and save helper."""
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved {path}")


def plot_cost_sensitivity(results_dir: Path, output_dir: Path) -> None:
    """Plot how the cost sweep never triggers expansion."""
    df = pd.read_csv(results_dir / "cost_sensitivity.csv").sort_values("cost_per_mw")

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    cost = df["cost_per_mw"]
    op_cost = df["operating_cost"] / 1e3  # k$

    ax1.semilogx(
        cost,
        op_cost,
        marker="o",
        linewidth=2.4,
        color="#1f77b4",
        label="Operating cost",
    )
    ax1.set_xlabel("Line cost [$ per MW] (log scale)")
    ax1.set_ylabel("Operating cost [$k]")
    ax1.set_title("Cost sweep: flat operating cost, zero buildout")
    ax1.grid(True, which="both", axis="x", alpha=0.3)
    baseline = op_cost.iloc[0]
    ax1.axhline(
        baseline, color="#ff7f0e", linestyle="--", linewidth=1.2, label="Baseline"
    )
    ax1.text(
        cost.min() * 1.05,
        baseline + 2,
        "No change across tested capex levels",
        color="#ff7f0e",
        fontsize=9,
    )
    ax1.legend(loc="upper left")

    ax2 = ax1.twinx()
    ax2.scatter(
        cost,
        df["lines_built"],
        color="#2ca02c",
        marker="s",
        s=50,
        label="Lines built",
        zorder=5,
    )
    ax2.set_ylabel("Lines built")
    ax2.set_ylim(-0.2, max(df["lines_built"].max(), 0) + 1)
    ax2.legend(loc="upper right")

    _save(fig, output_dir / "plot_cost_sensitivity.png")


def plot_load_shedding(results_dir: Path, output_dir: Path) -> None:
    """Plot shortage magnitude and economic penalty for the stress test."""
    df = pd.read_csv(results_dir / "load_shedding_periods.csv")
    df["load_shed_cost_musd"] = df["load_shed_cost"] / 1e6

    fig, ax1 = plt.subplots(figsize=(8.5, 4.5))
    sns.barplot(
        data=df,
        x="period",
        y="load_shed_pct",
        palette="Reds",
        ax=ax1,
        edgecolor="black",
    )
    ax1.set_xlabel("Period")
    ax1.set_ylabel("Load shed [% of demand]")
    ax1.set_title("Deliberate scarcity: 20% availability, 40% load bump")
    ax1.set_ylim(0, df["load_shed_pct"].max() * 1.15)
    ax1.bar_label(ax1.containers[0], fmt="%.1f%%", padding=3, fontsize=9)

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="period",
        y="load_shed_cost_musd",
        marker="o",
        linewidth=2.4,
        color="#1f77b4",
        ax=ax2,
    )
    ax2.set_ylabel("Penalty cost [USD million]")
    ax2.grid(False)

    _save(fig, output_dir / "plot_load_shedding.png")


def plot_robust_scenarios(results_dir: Path, output_dir: Path) -> None:
    """Plot scenario-level load and operating cost from the robust solve."""
    df = pd.read_csv(results_dir / "robust_tep_summary.csv")
    df["operating_cost_musd"] = df["operating_cost"] / 1e6

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    sns.barplot(
        data=df,
        x="scenario",
        y="operating_cost_musd",
        palette="Set2",
        ax=axes[0],
    )
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Operating cost [USD million]")
    axes[0].set_title("Scenario operating cost (robust TEP)")
    axes[0].bar_label(axes[0].containers[0], fmt="%.3f", padding=3, fontsize=9)
    axes[0].text(
        -0.2,
        0.92,
        "Builds two 200 MW links\n101-106 and 101-117",
        transform=axes[0].transAxes,
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#bbbbbb", boxstyle="round,pad=0.3"),
    )

    sns.barplot(
        data=df,
        x="scenario",
        y="total_load_mw",
        palette="Set3",
        ax=axes[1],
    )
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Total load [MW]")
    axes[1].set_title("Stress level by scenario")
    axes[1].bar_label(axes[1].containers[0], fmt="%.0f", padding=3, fontsize=9)

    fig.suptitle("Scenario-based robust expansion summary", fontsize=13)
    _save(fig, output_dir / "plot_robust_scenarios.png")


def main() -> None:
    results_dir = Path("results")
    output_dir = results_dir / "plots"

    plot_cost_sensitivity(results_dir, output_dir)
    plot_load_shedding(results_dir, output_dir)
    plot_robust_scenarios(results_dir, output_dir)


if __name__ == "__main__":
    main()
