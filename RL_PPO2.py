import gym
import numpy as np
#Using the baseline algorithms of RL
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.ppo2 import PPO2

env = gym.make('CartPole-v1')
env = DummyVecEnv([lambda: env])  

model = PPO2(MlpPolicy, env, verbose=0)

def evaluate1(model, ep=100):
  episode_rewards=[0.0]
  obs = env.reset()
  for i in range(ep):
    action, states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    episode_rewards[-1]+=rewards[0]
    if dones[0]:
      obs = env.reset()
      episode_rewards.append(0.0)
  mean_100_ep = round(np.mean(episode_rewards[-100:]),1)
  print("Mean rewards = ", mean_100_ep, "Total episodes = ", len(episode_rewards))
  return mean_100_ep
  x1 = evaluate1(model, ep=10000)
  print(x1)
