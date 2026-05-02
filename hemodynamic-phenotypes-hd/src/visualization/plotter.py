"""Visualization functions for hemodynamic phenotype analysis."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

from config.settings import FIGURE_DPI, FIGURE_FORMAT, COLORS, PHENOTYPE_NAMES


def setup_chinese_font():
    """Setup Chinese font support for matplotlib."""
    font_paths = [
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            font_manager.fontManager.addfont(fp)
            prop = font_manager.FontProperties(fname=fp)
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def plot_mean_trajectories(
    labeled_sessions, output_path, title="Mean SBP Trajectories by Phenotype"
):
    """Plot mean SBP trajectories for each phenotype."""
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(12, 8))

    for pid in range(4):
        sessions = [s for s in labeled_sessions if s["phenotype_id"] == pid]
        trajectories = [s["time_sbp"] for s in sessions]

        max_len = max(len(t) for t in trajectories)
        padded = [
            np.pad(t, (0, max_len - len(t)), constant_values=np.nan)
            for t in trajectories
        ]
        traj_array = np.array(padded)
        mean_traj = np.nanmean(traj_array, axis=0)
        std_traj = np.nanstd(traj_array, axis=0)

        time_points = np.arange(len(mean_traj))
        ax.plot(
            time_points,
            mean_traj,
            color=COLORS[pid],
            linewidth=2.5,
            label=PHENOTYPE_NAMES[pid],
            alpha=0.9,
        )
        ax.fill_between(
            time_points,
            mean_traj - std_traj,
            mean_traj + std_traj,
            color=COLORS[pid],
            alpha=0.15,
        )

    ax.set_xlabel("Time Point", fontsize=14)
    ax.set_ylabel("SBP (mmHg)", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_center_distribution(dist_shenyi, dist_fuding, output_path):
    """Plot phenotype distribution comparison between centers."""
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    pids = sorted(set(list(dist_shenyi.keys()) + list(dist_fuding.keys())))
    x = np.arange(len(pids))
    width = 0.35

    shenyi_vals = [dist_shenyi.get(pid, 0) * 100 for pid in pids]
    fuding_vals = [dist_fuding.get(pid, 0) * 100 for pid in pids]

    bars1 = ax.bar(
        x - width / 2,
        shenyi_vals,
        width,
        label="Shenyi",
        color="#3498db",
        edgecolor="white",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        fuding_vals,
        width,
        label="Fuding",
        color="#e74c3c",
        edgecolor="white",
        linewidth=0.5,
    )

    for bar in bars1:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for bar in bars2:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Phenotype", fontsize=14)
    ax.set_ylabel("Distribution (%)", fontsize=14)
    ax.set_title("Phenotype Distribution by Center", fontsize=16, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [PHENOTYPE_NAMES[pid].split(":")[0] for pid in pids], fontsize=12
    )
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_idh_rate_by_phenotype(idh_rates, output_path):
    """Plot IDH rate by phenotype."""
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    pids = sorted(idh_rates.keys())
    rates = [idh_rates[pid] * 100 for pid in pids]
    colors_bar = [COLORS[pid] for pid in pids]

    bars = ax.bar(
        range(len(pids)), rates, color=colors_bar, edgecolor="white", linewidth=0.5
    )

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{rate:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlabel("Phenotype", fontsize=14)
    ax.set_ylabel("IDH Rate (%)", fontsize=14)
    ax.set_title(
        "Intradialytic Hypotension Rate by Phenotype", fontsize=16, fontweight="bold"
    )
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(
        [PHENOTYPE_NAMES[pid].split(":")[0] for pid in pids], fontsize=12
    )
    ax.set_ylim(0, max(rates) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_ufr_comparison(ufr_stats, output_path):
    """Plot UFR comparison by phenotype."""
    setup_chinese_font()

    fig, ax = plt.subplots(figsize=(10, 6))

    pids = sorted(ufr_stats.keys())
    means = [ufr_stats[pid]["mean_ufr"] for pid in pids]
    stds = [ufr_stats[pid]["std_ufr"] for pid in pids]

    bars = ax.bar(
        range(len(pids)),
        means,
        yerr=stds,
        capsize=5,
        color=[COLORS[pid] for pid in pids],
        edgecolor="white",
        linewidth=0.5,
    )

    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 5,
            f"{mean:.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    ax.set_xlabel("Phenotype", fontsize=14)
    ax.set_ylabel("Prescribed UFR (ml/hr)", fontsize=14)
    ax.set_title("Prescribed UFR by Phenotype", fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(
        [PHENOTYPE_NAMES[pid].split(":")[0] for pid in pids], fontsize=12
    )
    ax.grid(True, alpha=0.3, axis="y")
    ax.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
