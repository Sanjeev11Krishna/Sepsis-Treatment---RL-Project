import numpy as np


class MetricsTracker:
    def __init__(self):
        self.rewards   = []
        self.survivals = []

    def record(self, reward: float, alive: bool):
        self.rewards.append(reward)
        self.survivals.append(int(alive))

    def rolling_stats(self, window: int = 100) -> dict:
        r = self.rewards[-window:]
        s = self.survivals[-window:]
        return {
            "mean_reward":   float(np.mean(r)) if r else 0.0,
            "std_reward":    float(np.std(r))  if r else 0.0,
            "survival_rate": float(np.mean(s)) if s else 0.0,
        }

    def smoothed(self, values: list, window: int = 50) -> list:
        out = []
        for i in range(len(values)):
            w = values[max(0, i - window + 1): i + 1]
            out.append(np.mean(w))
        return out
