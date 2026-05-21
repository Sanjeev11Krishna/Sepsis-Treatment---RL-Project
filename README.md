# Sepsis Treatment RL — Q-Learning Agent

A reinforcement learning project that trains an agent to decide optimal
treatment actions for sepsis patients, using tabular Q-learning.

---

## Project structure

```
sepsis_rl/
├── environment/
│   └── sepsis_env.py      # Custom Gymnasium environment
├── agent/
│   └── q_agent.py         # Q-learning agent with discretisation
├── utils/
│   ├── metrics.py         # Reward / survival tracking
│   └── plot.py            # All matplotlib charts
├── train.py               # Training loop
├── evaluate.py            # Compare RL vs rule-based vs random
├── demo.py                # Run a single episode with verbose output
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to run

### 1. Train the agent
```bash
python train.py
```
Trains for 2000 episodes. Saves the Q-table to `models/q_agent.pkl`.
Generates `plots/training.png` showing reward and survival curves.

### 2. Evaluate (compare all policies)
```bash
python evaluate.py
```
Runs 500 episodes each for:
- RL agent (Q-learning, greedy)
- Rule-based policy (clinical heuristics)
- Random policy (baseline)

Generates `plots/comparison.png`.

### 3. Demo a single episode
```bash
python demo.py                  # RL agent
python demo.py --policy rule    # rule-based
python demo.py --policy random  # random
```
Prints step-by-step vitals and actions, then plots the episode trace.

---

## State space (5 vitals)

| Vital           | Unit    | Normal range    |
|-----------------|---------|-----------------|
| Heart rate      | bpm     | 60 – 100        |
| Blood pressure  | mmHg    | 90 – 120        |
| SpO2 (oxygen)   | %       | 95 – 100        |
| Temperature     | °C      | 36.5 – 37.5     |
| Lactate         | mmol/L  | 0.5 – 2.0       |

## Action space (6 actions)

| # | Action                  |
|---|-------------------------|
| 0 | No treatment            |
| 1 | IV fluids               |
| 2 | Vasopressors            |
| 3 | Antibiotics             |
| 4 | IV fluids + Vasopressors|
| 5 | All three               |

## Reward function

| Event                      | Reward |
|----------------------------|--------|
| Vital in normal range      | +1     |
| Vital outside normal range | -1     |
| Vital improving            | +0.5   |
| SpO2 < 90% (hypoxia)       | -5     |
| BP < 70 (hypotension)      | -5     |
| Lactate > 4 (acidosis)     | -5     |
| Survival bonus (end)       | +20    |
| Death penalty (end)        | -20    |

---

## Q-Learning hyperparameters (defaults)

| Parameter       | Value  | Description                  |
|-----------------|--------|------------------------------|
| α (alpha)       | 0.1    | Learning rate                |
| γ (gamma)       | 0.95   | Discount factor              |
| ε start         | 1.0    | Initial exploration rate     |
| ε min           | 0.05   | Minimum exploration rate     |
| ε decay         | 0.995  | Per-episode decay multiplier |
| Episodes        | 2000   | Training episodes            |

---

## Extending the project

- Swap Q-learning for **DQN** (Deep Q-Network) using PyTorch
- Use real **MIMIC-III** patient trajectories as environment data
- Add a **Dueling DQN** or **Double DQN** architecture
- Implement **offline RL** (train from logged hospital data only)
- Add a **web dashboard** using Streamlit for live visualisation
