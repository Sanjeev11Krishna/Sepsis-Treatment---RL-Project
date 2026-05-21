import numpy as np
import gymnasium as gym
from gymnasium import spaces


class SepsisEnv(gym.Env):
    """
    Sepsis Treatment RL Environment

    State:  [heart_rate, blood_pressure, spo2, temperature, lactate]
    Actions:
        0 - No treatment
        1 - IV fluids
        2 - Vasopressors
        3 - Antibiotics
        4 - IV fluids + Vasopressors
        5 - All three (IV + Vaso + Antibiotics)

    Reward:
        +1  per step vitals are in safe range
        -1  per step vitals are outside safe range
        +20 episode survival bonus
        -20 episode death penalty
    """

    metadata = {"render_modes": ["human"]}

    NORMAL_RANGES = {
        "heart_rate":      (60,  100),
        "blood_pressure":  (90,  120),
        "spo2":            (95,  100),
        "temperature":     (36.5, 37.5),
        "lactate":         (0.5,  2.0),
    }

    VITAL_NAMES = list(NORMAL_RANGES.keys())
    N_ACTIONS   = 6
    MAX_STEPS   = 10

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        low  = np.array([40.0,  40.0, 70.0, 34.0, 0.2], dtype=np.float32)
        high = np.array([180.0, 180.0, 100.0, 42.0, 8.0], dtype=np.float32)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space      = spaces.Discrete(self.N_ACTIONS)

        self.state    = None
        self.steps    = 0

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start patient with randomly abnormal vitals (simulating sepsis onset)
        self.state = np.array([
            np.random.uniform(80, 140),   # elevated heart rate
            np.random.uniform(60, 100),   # low blood pressure
            np.random.uniform(88,  97),   # borderline oxygen
            np.random.uniform(36,  39),   # near-normal temp
            np.random.uniform(1.5,  5.0), # elevated lactate
        ], dtype=np.float32)

        self.steps = 0
        return self.state.copy(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action)

        prev_state = self.state.copy()
        self._apply_action(action)
        self._natural_progression()
        self.steps += 1

        reward   = self._compute_reward(prev_state)
        alive    = self._is_alive()
        done     = (self.steps >= self.MAX_STEPS) or not alive

        if done:
            reward += 20 if alive else -20

        info = {"alive": alive, "steps": self.steps, "action_name": self._action_name(action)}
        return self.state.copy(), reward, done, False, info

    # ------------------------------------------------------------------
    def _apply_action(self, action):
        hr, bp, spo2, temp, lac = self.state

        if action == 1:   # IV fluids
            bp   += np.random.uniform(5, 12)
            lac  -= np.random.uniform(0.2, 0.5)

        elif action == 2: # Vasopressors
            bp   += np.random.uniform(10, 20)
            hr   -= np.random.uniform(2,  8)

        elif action == 3: # Antibiotics
            temp -= np.random.uniform(0.1, 0.4)
            lac  -= np.random.uniform(0.1, 0.3)

        elif action == 4: # IV + Vasopressors
            bp   += np.random.uniform(12, 22)
            lac  -= np.random.uniform(0.3, 0.6)
            hr   -= np.random.uniform(2,  6)

        elif action == 5: # All three
            bp   += np.random.uniform(15, 25)
            spo2 += np.random.uniform(1,  3)
            temp -= np.random.uniform(0.2, 0.5)
            lac  -= np.random.uniform(0.4, 0.8)

        self.state = np.array([hr, bp, spo2, temp, lac], dtype=np.float32)
        self._clip_state()

    def _natural_progression(self):
        """Without treatment, sepsis worsens naturally."""
        noise = np.random.normal(0, [3, 4, 0.5, 0.15, 0.2])
        deterioration = np.array([2, -3, -0.3, 0.1, 0.3])  # sepsis worsens over time
        self.state += noise + deterioration
        self._clip_state()

    def _clip_state(self):
        low  = self.observation_space.low
        high = self.observation_space.high
        self.state = np.clip(self.state, low, high)

    def _compute_reward(self, prev_state):
        reward = 0.0
        for i, (name, (lo, hi)) in enumerate(self.NORMAL_RANGES.items()):
            val  = self.state[i]
            prev = prev_state[i]
            if lo <= val <= hi:
                reward += 1.0
            else:
                reward -= 1.0
            # Bonus for improvement
            prev_dist = abs(prev - (lo + hi) / 2)
            curr_dist = abs(val  - (lo + hi) / 2)
            if curr_dist < prev_dist:
                reward += 0.5

        # Critical penalty for dangerous vitals
        if self.state[2] < 90:   reward -= 5   # severe hypoxia
        if self.state[1] < 70:   reward -= 5   # severe hypotension
        if self.state[4] > 4.0:  reward -= 5   # severe lactic acidosis

        return float(reward)

    def _is_alive(self):
        bp, spo2, lac = self.state[1], self.state[2], self.state[4]
        return bp >= 60 and spo2 >= 85 and lac <= 6.0

    def _action_name(self, action):
        names = ["No treatment", "IV fluids", "Vasopressors",
                 "Antibiotics", "IV + Vasopressors", "All three"]
        return names[action]

    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        hr, bp, spo2, temp, lac = self.state
        print(f"\n--- Step {self.steps} ---")
        print(f"  Heart rate:      {hr:.1f} bpm   (normal: 60-100)")
        print(f"  Blood pressure:  {bp:.1f} mmHg  (normal: 90-120)")
        print(f"  SpO2:            {spo2:.1f}%     (normal: 95-100)")
        print(f"  Temperature:     {temp:.1f} °C   (normal: 36.5-37.5)")
        print(f"  Lactate:         {lac:.2f} mmol/L (normal: 0.5-2.0)")
