import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

os.makedirs("plots", exist_ok=True)

COLORS = {
    "rl":    "#185FA5",
    "rule":  "#3B6D11",
    "rand":  "#993C1D",
    "accent": "#EF9F27",
}


def plot_training_results(tracker, save_path="plots/training.png"):
    rewards   = tracker.rewards
    survivals = tracker.survivals
    episodes  = list(range(1, len(rewards) + 1))
    smoothed_r = tracker.smoothed(rewards,   window=50)
    smoothed_s = tracker.smoothed([s * 100 for s in survivals], window=50)

    fig = plt.figure(figsize=(14, 9))
    fig.suptitle("Sepsis RL — Training Results (Q-Learning)", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Reward per episode
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episodes, rewards,    alpha=0.2, color=COLORS["rl"], linewidth=0.6, label="Raw reward")
    ax1.plot(episodes, smoothed_r, color=COLORS["rl"],            linewidth=2,   label="Smoothed (50-ep)")
    ax1.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total reward")
    ax1.set_title("Reward per episode")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. Survival rate
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episodes, smoothed_s, color=COLORS["rule"], linewidth=2)
    ax2.fill_between(episodes, smoothed_s, alpha=0.15, color=COLORS["rule"])
    ax2.set_ylim(0, 100)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Survival rate (%)")
    ax2.set_title("Rolling survival rate")
    ax2.grid(alpha=0.3)

    # 3. Epsilon decay
    decay  = 0.995
    epsilons = [max(0.05, 1.0 * (decay ** i)) for i in range(len(episodes))]
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episodes, epsilons, color=COLORS["accent"], linewidth=2)
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Epsilon (ε)")
    ax3.set_title("Exploration rate decay")
    ax3.grid(alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training plot saved → {save_path}")
    plt.show()


def plot_comparison(results, save_path="plots/comparison.png"):
    labels   = [r["label"] for r in results]
    s_rates  = [r["survival_rate"] for r in results]
    m_rewards = [r["mean_reward"] for r in results]
    colors   = [COLORS["rl"], COLORS["rule"], COLORS["rand"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Policy Comparison — RL vs Rule-based vs Random", fontsize=13, fontweight="bold")

    # Survival rate bar chart
    bars = axes[0].bar(labels, s_rates, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("Survival rate (%)")
    axes[0].set_title("Survival rate")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, s_rates):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 1, f"{val:.1f}%",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Mean reward bar chart
    bars2 = axes[1].bar(labels, m_rewards, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_ylabel("Mean total reward")
    axes[1].set_title("Mean reward per episode")
    axes[1].axhline(0, color="gray", linewidth=0.5, linestyle="--")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, val in zip(bars2, m_rewards):
        ypos = bar.get_height() + (1 if val >= 0 else -3)
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     ypos, f"{val:.1f}",
                     ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Comparison plot saved → {save_path}")
    plt.show()


def plot_episode(states_log, actions_log, rewards_log, save_path="plots/episode.png"):
    """Visualise a single episode's vitals over time."""
    vital_names = ["Heart rate (bpm)", "BP (mmHg)", "SpO2 (%)", "Temp (°C)", "Lactate (mmol/L)"]
    normal_lo   = [60,  90,  95,  36.5, 0.5]
    normal_hi   = [100, 120, 100, 37.5, 2.0]
    states_arr  = np.array(states_log)
    steps       = list(range(len(states_arr)))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("Episode trace — patient vitals over time", fontsize=13, fontweight="bold")
    axes_flat = axes.flatten()

    for i, (name, lo, hi) in enumerate(zip(vital_names, normal_lo, normal_hi)):
        ax = axes_flat[i]
        vals = states_arr[:, i]
        ax.plot(steps, vals, color=COLORS["rl"], linewidth=2, marker="o", markersize=4)
        ax.axhspan(lo, hi, alpha=0.1, color="green", label="Normal range")
        ax.axhline(lo, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.axhline(hi, color="green", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    # Reward over time
    ax = axes_flat[5]
    ax.bar(steps, rewards_log, color=[
        COLORS["rl"] if r >= 0 else COLORS["rand"] for r in rewards_log
    ], edgecolor="white", linewidth=0.3)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Reward per step")
    ax.set_xlabel("Step")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Episode plot saved → {save_path}")
    plt.show()
