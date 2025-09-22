# src/env.py
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class F1HybridEnv(gym.Env):
    def __init__(self, df, features, lap_len=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = features
        self.obs_shape = (len(self.features),)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.obs_shape, dtype=np.float32)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.sample_dt = 0.08
        self.lap_len = lap_len if lap_len is not None else df['idx'].nunique()
        self.current_lap_start = 0

    def reset(self, seed=None, options=None):
        # pick a random lap start so agent sees varied starting points
        self.current_lap_start = np.random.randint(0, len(self.df)-self.lap_len)
        self.t = int(self.current_lap_start)
        self.battery, self.time = 1.0, 0.0
        row = self.df.iloc[self.t]
        obs = np.array([row[f] for f in self.features], dtype=np.float32)
        return obs, {}

    def step(self, action):
        frac = float(np.clip(action[0], 0.0, 1.0))
        deploy = min(0.15, self.battery * frac)

        # harvest if braking
        if self.df.loc[self.t, 'brake'] > 0.6 and self.battery < 0.99:
            self.battery = min(1.0, self.battery + 0.05)

        self.battery = max(0.0, self.battery - deploy)
        dt = self.sample_dt * (0.95 if deploy > 0 else 1.0)
        self.time += dt
        self.t += 1

        terminated = (self.t - self.current_lap_start) >= self.lap_len
        truncated = False
        obs = np.zeros(self.obs_shape, dtype=np.float32) if terminated else np.array([self.df.iloc[self.t][f] for f in self.features], dtype=np.float32)

        # shaped reward (time penalty + deploy bonus - brake penalty)
        time_penalty = -dt
        deploy_bonus = deploy * 8.0
        brake_penalty = -0.05 * self.df.loc[self.t-1, 'brake']
        reward = time_penalty + deploy_bonus + brake_penalty

        return obs, reward, terminated, truncated, {'battery': self.battery, 'time': self.time}
