#!/usr/bin/env python3
"""Plot exploitability vs gain for exact_sweep_main CSV output."""

import argparse
import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ALGORITHM_STYLES = {
    "ses":   {"color": "#1f77b4", "marker": "o", "label": "SES"},
    "cdrnr": {"color": "#ff7f0e", "marker": "s", "label": "CDRNR"},
    "ox":    {"color": "#2ca02c", "marker": "^", "label": "OX"},
}

# Fallback style for unknown algorithms
FALLBACK_COLORS = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
FALLBACK_MARKERS = ["D", "v", "P", "X"]


def style_for(alg: str, seen_unknown: dict) -> dict:
    if alg in ALGORITHM_STYLES:
        return ALGORITHM_STYLES[alg]
    if alg not in seen_unknown:
        idx = len(seen_unknown)
        seen_unknown[alg] = {
            "color": FALLBACK_COLORS[idx % len(FALLBACK_COLORS)],
            "marker": FALLBACK_MARKERS[idx % len(FALLBACK_MARKERS)],
            "label": alg.upper(),
        }
    return seen_unknown[alg]


def build_suffix(game: str, target_player: int, depth, depth_mode: str) -> str:
    return f"_{game}_p{target_player}_d{depth}{depth_mode}"


def insert_suffix(out_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(out_path)
    if ext.lower() == ".png":
        return base + suffix + ext
    return out_path + suffix + ".png"


def plot_group(df_group, title_prefix: str, out_path: str) -> dict:
    """Plot one (game, target_player, depth) group. Returns {alg: n_points}."""
    fig, ax = plt.subplots(figsize=(8, 5))
    seen_unknown: dict = {}
    counts: dict = {}

    for alg, df_alg in df_group.groupby("algorithm", sort=True):
        df_alg = df_alg.dropna(subset=["exploitability", "gain"])
        df_alg = df_alg.sort_values("p")
        if df_alg.empty:
            continue

        st = style_for(str(alg), seen_unknown)
        xs = df_alg["exploitability"].tolist()
        ys = df_alg["gain"].tolist()
        ps = df_alg["p"].tolist()

        ax.plot(xs, ys,
                color=st["color"],
                marker=st["marker"],
                label=st["label"],
                linewidth=1.5,
                markersize=6)

        for x, y, p in zip(xs, ys, ps):
            ax.annotate(
                f"p={p:.2g}",
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color=st["color"],
            )

        counts[alg] = len(df_alg)

    ax.set_xlabel("Exploitability (gain over Nash for opponent)", fontsize=11)
    ax.set_ylabel("Gain vs uniform opponent", fontsize=11)
    ax.set_title(title_prefix, fontsize=12)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.4g"))

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot exploitability vs gain from exact_sweep CSV output."
    )
    parser.add_argument("--csv", required=True, help="Path to input CSV file.")
    parser.add_argument("--out", required=True, help="Output image path (PNG).")
    parser.add_argument(
        "--title",
        default=None,
        help="Title prefix for the plot. Defaults to the game name.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: CSV file not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)

    required_cols = {"game", "algorithm", "p", "gain", "exploitability",
                     "target_player", "depth", "depth_mode"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: CSV is missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    groups = list(df.groupby(["game", "target_player", "depth", "depth_mode"], sort=True))

    written: list[str] = []
    for (game, target_player, depth, depth_mode), df_group in groups:
        title_prefix = args.title if args.title else f"{game}"
        if len(groups) > 1 or args.title is None:
            title_prefix = (
                f"{title_prefix}  |  player={target_player}"
                f"  depth={depth} ({depth_mode})"
            )

        if len(groups) == 1:
            out_path = args.out
            if not out_path.lower().endswith(".png"):
                out_path = out_path + ".png"
        else:
            suffix = build_suffix(game, target_player, depth, depth_mode)
            out_path = insert_suffix(args.out, suffix)

        counts = plot_group(df_group, title_prefix, out_path)
        written.append(out_path)

        algo_summary = ", ".join(
            f"{alg}={n}" for alg, n in sorted(counts.items())
        )
        print(f"Wrote: {out_path}  [{algo_summary}]")

    if not written:
        print("No plots produced (no valid groups found).")


if __name__ == "__main__":
    main()
