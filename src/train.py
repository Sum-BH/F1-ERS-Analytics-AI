# src/train.py
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from .env import F1HybridEnv
from .configs import DATA_CSV, OUTPUT_DIR

class RewardLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self):
        rewards = self.locals.get('rewards')
        dones = self.locals.get('dones')
        if rewards is None or dones is None:
            return True
        for r, d in zip(rewards, dones):
            self.current_rewards.append(r)
            if d:
                self.episode_rewards.append(sum(self.current_rewards))
                self.current_rewards = []
        return True

def run_training(total_timesteps=50000):
    df = pd.read_csv(DATA_CSV)
    features = ['speed_kmh','brake','battery']
    env = DummyVecEnv([lambda: F1HybridEnv(df, features)])
    model = PPO('MlpPolicy', env, verbose=0)
    reward_logger = RewardLogger()
    model.learn(total_timesteps=total_timesteps, callback=reward_logger)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6,4))
    plt.plot(reward_logger.episode_rewards)
    plt.title('PPO Episode Rewards (per lap)')
    plt.xlabel('Episode (lap)')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(out_dir / 'ppo_real_rewards.png')
    plt.close()
    model.save(str(out_dir / 'ppo_model.zip'))
    print('Training complete. Rewards plot and model saved to outputs/')

if __name__ == '__main__':
    run_training()
