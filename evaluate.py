import numpy as np
from environment.sepsis_env import SepsisEnv
from agent.q_agent import QLearningAgent
from utils.plot import plot_comparison


# ── Baselines ──────────────────────────────────────────────────────────

def random_policy(state):
    """Always picks a random action."""
    return np.random.randint(6)


def rule_based_policy(state):
    """
    Simple clinical rule-based policy:
      - Low BP           → vasopressors (action 2)
      - Low BP + high lac → IV + Vaso  (action 4)
      - Low SpO2         → antibiotics (action 3)
      - All bad          → all three   (action 5)
      - Otherwise        → no treatment (action 0)
    """
    hr, bp, spo2, temp, lac = state
    low_bp   = bp   < 90
    low_spo2 = spo2 < 95
    high_lac = lac  > 2.0

    if low_bp and high_lac:
        return 4   # IV fluids + vasopressors
    if low_bp and low_spo2:
        return 5   # all three
    if low_bp:
        return 2   # vasopressors only
    if low_spo2 or high_lac:
        return 3   # antibiotics
    return 0       # no treatment


# ── Evaluation runner ─────────────────────────────────────────────────

def evaluate_policy(policy_fn, n_episodes=500, label="Policy"):
    env    = SepsisEnv()
    rewards, survivals = [], []

    for _ in range(n_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            action = policy_fn(state)
            state, reward, done, _, info = env.step(action)
            ep_reward += reward

        rewards.append(ep_reward)
        survivals.append(int(info["alive"]))

    mean_r  = np.mean(rewards)
    mean_s  = np.mean(survivals) * 100
    std_r   = np.std(rewards)

    print(f"  {label:<25} | Reward: {mean_r:>7.1f} ± {std_r:.1f} | "
          f"Survival: {mean_s:>5.1f}%")
    return {"label": label, "rewards": rewards, "survivals": survivals,
            "mean_reward": mean_r, "survival_rate": mean_s}


def run_evaluation(model_path="models/q_agent.pkl", n_episodes=500):
    agent = QLearningAgent()
    agent.load(model_path)
    agent.epsilon = 0.0   # fully greedy at evaluation time

    print("\n" + "=" * 55)
    print("  Evaluation — 500 episodes per policy")
    print("=" * 55)

    results = [
        evaluate_policy(agent.best_action,  n_episodes, "RL Agent (Q-learning)"),
        evaluate_policy(rule_based_policy,  n_episodes, "Rule-based policy"),
        evaluate_policy(random_policy,      n_episodes, "Random policy"),
    ]

    print("=" * 55)
    winner = max(results, key=lambda x: x["survival_rate"])
    print(f"\n  Best policy: {winner['label']}  "
          f"({winner['survival_rate']:.1f}% survival)\n")

    plot_comparison(results)
    return results


if __name__ == "__main__":
    run_evaluation()
