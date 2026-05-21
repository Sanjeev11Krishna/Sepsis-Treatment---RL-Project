"""
demo.py — Run a single episode with the trained agent and print every step.
Also plots the vitals trace for the episode.

Usage:
    python demo.py                    # uses saved model
    python demo.py --policy random    # compare against random
    python demo.py --policy rule      # compare against rule-based
"""
import argparse
import numpy as np
from environment.sepsis_env import SepsisEnv
from agent.q_agent import QLearningAgent
from evaluate import rule_based_policy, random_policy
from utils.plot import plot_episode

ACTION_NAMES = [
    "No treatment", "IV fluids", "Vasopressors",
    "Antibiotics", "IV + Vasopressors", "All three"
]
VITAL_NAMES = ["Heart rate", "Blood pressure", "SpO2", "Temperature", "Lactate"]
VITAL_UNITS = ["bpm", "mmHg", "%", "°C", "mmol/L"]
NORMAL_LO   = [60, 90, 95, 36.5, 0.5]
NORMAL_HI   = [100, 120, 100, 37.5, 2.0]


def format_vital(name, val, unit, lo, hi):
    status = "OK " if lo <= val <= hi else "!!!"
    return f"  [{status}] {name:<18} {val:>6.1f} {unit}"


def run_demo(policy_fn, label="Agent"):
    env = SepsisEnv(render_mode="human")
    state, _ = env.reset()

    states_log  = [state.copy()]
    actions_log = []
    rewards_log = []
    done        = False

    print("\n" + "=" * 55)
    print(f"  Episode demo — {label}")
    print("=" * 55)

    while not done:
        action = policy_fn(state)
        next_state, reward, done, _, info = env.step(action)

        print(f"\n  Step {info['steps']:>2} | Action: {ACTION_NAMES[action]:<22} | Reward: {reward:>6.1f}")
        for i, (n, u, lo, hi) in enumerate(zip(VITAL_NAMES, VITAL_UNITS, NORMAL_LO, NORMAL_HI)):
            print(format_vital(n, next_state[i], u, lo, hi))

        states_log.append(next_state.copy())
        actions_log.append(action)
        rewards_log.append(reward)
        state = next_state

    outcome = "SURVIVED" if info["alive"] else "DIED"
    total_r = sum(rewards_log)
    print(f"\n  Outcome: {outcome}  |  Total reward: {total_r:.1f}")
    print("=" * 55)

    plot_episode(states_log, actions_log, rewards_log)
    return info["alive"], total_r


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["rl", "random", "rule"], default="rl")
    parser.add_argument("--model",  default="models/q_agent.pkl")
    args = parser.parse_args()

    if args.policy == "rl":
        agent = QLearningAgent()
        agent.load(args.model)
        agent.epsilon = 0.0
        run_demo(agent.best_action, label="RL Agent (Q-learning)")
    elif args.policy == "random":
        run_demo(random_policy, label="Random policy")
    else:
        run_demo(rule_based_policy, label="Rule-based policy")
