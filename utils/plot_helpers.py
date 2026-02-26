"""
Matplotlib styling and chart helpers for consistent visualizations.
Supports EN and PT-BR label switching.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COLOR_PALETTE, LABELS, FIGURE_SIZE, FIGURE_SIZE_SMALL

# Module-level language setting
_current_lang = "en"


def set_language(lang: str):
    """Set the display language for chart labels. Supported: 'en', 'pt-br'."""
    global _current_lang
    if lang not in LABELS:
        raise ValueError(f"Unsupported language: {lang}. Use 'en' or 'pt-br'.")
    _current_lang = lang


def get_label(key: str) -> str:
    """Get a label string in the current language."""
    return LABELS[_current_lang].get(key, key)


def get_colors() -> dict:
    return COLOR_PALETTE.copy()


def get_user_label(user_key: str) -> str:
    return LABELS[_current_lang].get(user_key, user_key)


def setup_style():
    """Apply consistent matplotlib styling."""
    plt.rcParams.update({
        "figure.figsize": FIGURE_SIZE,
        "figure.dpi": 100,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
    })


def comparative_bar(df: pd.DataFrame, metric: str, title: str, ax=None, ylabel: str = None):
    """Side-by-side bar chart comparing Trump vs AOC on a metric."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    colors = get_colors()
    users = ["trump", "aoc"]
    values = [df[df["user"] == u][metric].mean() for u in users]
    labels = [get_user_label(u) for u in users]
    bars = ax.bar(labels, values, color=[colors[u] for u in users], width=0.5, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.15 if max(values) > 0 else 1)
    return ax


def comparative_boxplot(df: pd.DataFrame, metric: str, title: str, ax=None, ylabel: str = None):
    """Box plots comparing distributions for Trump vs AOC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    colors = get_colors()
    users = ["trump", "aoc"]
    data = [df[df["user"] == u][metric].dropna().values for u in users]
    labels = [get_user_label(u) for u in users]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.5,
                    medianprops={"color": "black", "linewidth": 1.5})
    for patch, user in zip(bp["boxes"], users):
        patch.set_facecolor(colors[user])
        patch.set_alpha(0.7)

    ax.set_title(title, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def comparative_hist(df: pd.DataFrame, metric: str, title: str, bins: int = 20, ax=None):
    """Overlapping histograms for Trump vs AOC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE_SMALL)

    colors = get_colors()
    for user in ["trump", "aoc"]:
        user_data = df[df["user"] == user][metric].dropna()
        ax.hist(user_data, bins=bins, alpha=0.5, label=get_user_label(user),
                color=colors[user], edgecolor="white")

    ax.set_title(title, fontweight="bold")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_ylabel("Frequency" if _current_lang == "en" else "Frequência")
    ax.legend()
    return ax


def time_series(df: pd.DataFrame, date_col: str, metric: str, title: str, ax=None):
    """Line plot over time for Trump vs AOC."""
    if ax is None:
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    colors = get_colors()
    for user in ["trump", "aoc"]:
        user_data = df[df["user"] == user].sort_values(date_col)
        ax.plot(user_data[date_col], user_data[metric],
                marker="o", markersize=4, label=get_user_label(user),
                color=colors[user], linewidth=1.5, alpha=0.8)

    ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    return ax


def format_large_numbers(ax, axis="y"):
    """Format axis ticks with K/M suffixes for large numbers."""
    def formatter(x, pos):
        if x >= 1_000_000:
            return f"{x/1_000_000:.1f}M"
        elif x >= 1_000:
            return f"{x/1_000:.0f}K"
        return f"{x:.0f}"

    if axis == "y":
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(formatter))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(formatter))


def add_summary_stats(ax, df: pd.DataFrame, metric: str, y_pos: float = 0.95):
    """Add mean/median text annotations to a plot."""
    stats_text = []
    colors = get_colors()
    for user in ["trump", "aoc"]:
        data = df[df["user"] == user][metric].dropna()
        label = get_user_label(user).split(" (")[0]
        mean_val = data.mean()
        median_val = data.median()
        stats_text.append(f"{label}: mean={mean_val:,.1f}, median={median_val:,.1f}")

    text = "\n".join(stats_text)
    ax.text(0.02, y_pos, text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.3",
            facecolor="white", alpha=0.8))
