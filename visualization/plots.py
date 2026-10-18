"""
Visualizations for ForgetProof experiment results.

Generates 6 clean, publication-grade charts:
1. Accuracy vs Shard Count (SISA utility tradeoff)
2. Retrain Time Comparison (Single Shard vs Full Retrain)
3. Membership Inference Attack (Before vs After Forgetting)
4. Cumulative Compute Saved (Streaming Simulation)
5. Accuracy Drift during Continuous Withdrawals
6. Shard Size Distribution (Consent Batches)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless mode for LinuxONE VM / CI
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#e2e8f0",
    "text.color": "#e2e8f0",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#334155",
    "font.family": "sans-serif",
    "font.size": 11,
})

COLORS = {
    "primary": "#38bdf8",    # sky blue
    "secondary": "#a78bfa",  # purple
    "accent": "#22c55e",     # green
    "warning": "#ef4444",    # red
    "muted": "#94a3b8",      # gray
    "orange": "#f97316",
}


def save(fig, name, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_accuracy_vs_shards(sweep_results, output_dir="outputs"):
    """Chart 1: accuracy vs shard count."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ns = [r["n_shards"] for r in sweep_results]
    accs = [r["accuracy"] for r in sweep_results]

    ax.plot(ns, accs, "o-", color=COLORS["primary"], linewidth=2, markersize=8)
    ax.axhline(y=accs[0], color=COLORS["muted"], linestyle="--", alpha=0.5,
               label=f"Baseline (no sharding): {accs[0]:.4f}")

    ax.set_xlabel("Number of Shards")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy vs Shard Count (SISA Utility-Privacy Curve)")
    ax.set_xticks(ns)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return save(fig, "accuracy_vs_shards", output_dir)


def plot_retrain_comparison(shard_time, full_time, n_shards, output_dir="outputs"):
    """Chart 2: single shard retrain vs full retrain time."""
    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(["Full Model\nRetrain", f"Single Shard\nRetrain (1/{n_shards})"],
                  [full_time, shard_time],
                  color=[COLORS["warning"], COLORS["accent"]],
                  width=0.5, edgecolor="#334155")

    for bar, val in zip(bars, [full_time, shard_time]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}s", ha="center", va="bottom", fontweight="bold",
                color="#e2e8f0")

    speedup = full_time / shard_time if shard_time > 0 else float("inf")
    ax.set_title(f"Single Forget Latency ({speedup:.1f}× Faster with SISA)")
    ax.set_ylabel("Time (seconds)")
    ax.grid(True, axis="y", alpha=0.3)

    return save(fig, "retrain_comparison", output_dir)


def plot_mia_before_after(before, after, output_dir="outputs"):
    """Chart 3: MIA metrics and loss distributions before vs after forgetting."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # left: attack accuracy comparison
    ax = axes[0]
    metrics = ["attack_accuracy", "member_recall", "auc"]
    labels = ["Attack\nAccuracy", "Member\nRecall", "AUC"]
    x = np.arange(len(metrics))
    w = 0.35

    before_vals = [before[m] for m in metrics]
    after_vals = [after[m] for m in metrics]

    ax.bar(x - w/2, before_vals, w, label="Before Forget (Memorized)",
           color=COLORS["warning"], edgecolor="#334155")
    ax.bar(x + w/2, after_vals, w, label="After Forget (Unlearned)",
           color=COLORS["accent"], edgecolor="#334155")

    ax.axhline(y=0.5, color=COLORS["muted"], linestyle="--", alpha=0.5,
               label="Random Chance (0.50)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Membership Inference Attack (Canaries)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    # right: loss distributions
    ax = axes[1]
    if "member_losses" in before and "nonmember_losses" in before:
        ax.hist(before["member_losses"], bins=30, alpha=0.5,
                color=COLORS["warning"], label="Members (before)", density=True)
        ax.hist(before["nonmember_losses"], bins=30, alpha=0.5,
                color=COLORS["primary"], label="Non-members", density=True)
        if "member_losses" in after:
            ax.hist(after["member_losses"], bins=30, alpha=0.5,
                    color=COLORS["accent"], label="Forgotten (after)", density=True)
        ax.set_xlabel("Loss")
        ax.set_ylabel("Density")
        ax.set_title("Per-Sample Loss Distribution Shift")
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return save(fig, "mia_before_after", output_dir)


def plot_cumulative_savings(sim_log, output_dir="outputs"):
    """Chart 4: cumulative compute time saved across streaming requests."""
    fig, ax1 = plt.subplots(figsize=(10, 5))

    requests = [e["request_num"] for e in sim_log]
    sisa_times = [e["cumulative_sisa_time"] for e in sim_log]
    full_times = [e["cumulative_full_time"] for e in sim_log]
    savings = [e["compute_saved_pct"] for e in sim_log]

    ax1.fill_between(requests, full_times, sisa_times,
                     alpha=0.2, color=COLORS["accent"], label="Compute Saved")
    ax1.plot(requests, full_times, "-", color=COLORS["warning"],
             linewidth=2, label="Full Retrain (Cumulative)")
    ax1.plot(requests, sisa_times, "-", color=COLORS["accent"],
             linewidth=2, label="SISA Shard Retrain (Cumulative)")

    ax1.set_xlabel("Number of Consent Withdrawals")
    ax1.set_ylabel("Cumulative Retrain Time (seconds)")
    ax1.set_title("Cumulative Compute: SISA Sharding vs Full Retraining")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(requests, savings, "--", color=COLORS["orange"],
             alpha=0.8, label="% Saved")
    ax2.set_ylabel("Compute Saved (%)", color=COLORS["orange"])
    ax2.tick_params(axis="y", labelcolor=COLORS["orange"])

    fig.tight_layout()
    return save(fig, "cumulative_savings", output_dir)


def plot_accuracy_drift(sim_log, output_dir="outputs"):
    """Chart 5: test accuracy stability as records are continuously forgotten."""
    fig, ax = plt.subplots(figsize=(9, 5))

    requests = [e["request_num"] for e in sim_log]
    accs = [e["accuracy"] for e in sim_log]

    ax.plot(requests, accs, "-", color=COLORS["primary"], linewidth=2)
    ax.axhline(y=accs[0], color=COLORS["muted"], linestyle="--", alpha=0.5,
               label=f"Initial: {accs[0]:.4f}")

    ax.set_xlabel("Forget Requests Processed")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy Stability Under Continuous Consent Revocations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax.annotate(f"Final: {accs[-1]:.4f}",
                xy=(requests[-1], accs[-1]),
                xytext=(-60, 20), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color=COLORS["primary"]),
                color=COLORS["primary"], fontweight="bold")

    return save(fig, "accuracy_drift", output_dir)


def plot_shard_sizes(train_df, n_shards, output_dir="outputs"):
    """Chart 6: shard size distribution."""
    fig, ax = plt.subplots(figsize=(8, 5))

    sizes = [int((train_df["shard_id"] == s).sum()) for s in range(n_shards)]

    bars = ax.bar(range(n_shards), sizes,
                  color=COLORS["primary"], edgecolor="#334155", alpha=0.8)

    for bar, size in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(size), ha="center", va="bottom", fontsize=9, color="#e2e8f0")

    ax.set_xlabel("Shard ID (Consent Batch)")
    ax.set_ylabel("Number of Records")
    ax.set_title(f"Data Distribution Across {n_shards} Shards")
    ax.set_xticks(range(n_shards))
    ax.grid(True, axis="y", alpha=0.3)

    return save(fig, "shard_sizes", output_dir)
