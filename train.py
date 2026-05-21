import numpy as np
from environment.sepsis_env import SepsisEnv
from agent.q_agent import QLearningAgent
from utils.metrics import MetricsTracker
from utils.plot import plot_training_results


def train(
    n_episodes:    int   = 2000,
    alpha:         float = 0.1,
    gamma:         float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min:   float = 0.05,
    epsilon_decay: float = 0.995,
    save_path:     str   = "models/q_agent.pkl",
    verbose:       bool  = True,
    log_every:     int   = 100,
):
    env     = SepsisEnv()
    agent   = QLearningAgent(
        n_actions=env.N_ACTIONS,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )
    tracker = MetricsTracker()

    print("=" * 55)
    print("  Sepsis Treatment RL — Q-Learning Training")
    print("=" * 55)
    print(f"  Episodes   : {n_episodes}")
    print(f"  Alpha (lr) : {alpha}")
    print(f"  Gamma      : {gamma}")
    print(f"  ε start    : {epsilon_start}  → min {epsilon_min}")
    print("=" * 55)

    for ep in range(1, n_episodes + 1):
        state, _ = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            action              = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state      = next_state

        agent.decay_epsilon()
        tracker.record(ep_reward, info["alive"])

        if verbose and ep % log_every == 0:
            stats = tracker.rolling_stats(window=100)
            print(
                f"  Ep {ep:>5} | "
                f"Reward: {stats['mean_reward']:>7.1f} | "
                f"Survival: {stats['survival_rate']*100:>5.1f}% | "
                f"ε: {agent.epsilon:.3f}"
            )

    print("\nTraining complete.")
    print(f"  Final survival rate (last 100 eps): "
          f"{tracker.rolling_stats(100)['survival_rate']*100:.1f}%")

    import os
    os.makedirs("models", exist_ok=True)
    agent.save(save_path)

    plot_training_results(tracker)
    return agent, tracker


if __name__ == "__main__":
    train(n_episodes=2000)
