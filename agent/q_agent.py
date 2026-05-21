import numpy as np
import pickle


class QLearningAgent:
    """
    Tabular Q-Learning agent for sepsis treatment.

    Discretises the continuous state space (5 vitals) into bins
    and maintains a Q-table over (state, action) pairs.
    """

    VITAL_BINS = [
        [40, 60, 80, 100, 120, 180],       # heart_rate
        [40, 70, 90, 110, 130, 180],       # blood_pressure
        [70, 88, 92, 95, 98, 100],         # spo2
        [34, 36, 36.5, 37.5, 38.5, 42],   # temperature
        [0,  1,  2,   3,   5,   8],        # lactate
    ]

    def __init__(
        self,
        n_actions: int   = 6,
        alpha:     float = 0.1,
        gamma:     float = 0.95,
        epsilon:   float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Build Q-table shape: (bins per vital) x n_actions
        self.bins   = [np.array(b) for b in self.VITAL_BINS]
        shape       = tuple(len(b) - 1 for b in self.VITAL_BINS) + (n_actions,)
        self.qtable = np.zeros(shape)

    # ------------------------------------------------------------------
    def discretise(self, state: np.ndarray) -> tuple:
        """Map continuous vitals vector → discrete state index tuple."""
        return tuple(
            int(np.digitize(state[i], self.bins[i][1:-1]))
            for i in range(len(self.bins))
        )

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        idx = self.discretise(state)
        return int(np.argmax(self.qtable[idx]))

    def update(
        self,
        state:      np.ndarray,
        action:     int,
        reward:     float,
        next_state: np.ndarray,
        done:       bool,
    ):
        """Standard Q-learning update (Bellman equation)."""
        s  = self.discretise(state)
        ns = self.discretise(next_state)

        target = reward
        if not done:
            target += self.gamma * np.max(self.qtable[ns])

        self.qtable[s][action] += self.alpha * (target - self.qtable[s][action])

    def decay_epsilon(self):
        """Call once per episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def best_action(self, state: np.ndarray) -> int:
        """Greedy action (no exploration) — for evaluation."""
        return int(np.argmax(self.qtable[self.discretise(state)]))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"qtable": self.qtable, "epsilon": self.epsilon}, f)
        print(f"Agent saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.qtable  = data["qtable"]
        self.epsilon = data["epsilon"]
        print(f"Agent loaded from {path}")
