from tensorflow import keras
import gymnasium as gym
import numpy as np
import os
from keras.utils import to_categorical



root_path = os.path.abspath(os.path.dirname(__file__))


def main():
  total_reward = 0
  #Loading model
  '''
  ==keras.models.load_model==
  '''
  
  env = gym.make('roundabout-v0', render_mode='rgb_array')

  for _ in range(10):
    (obs,info) = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
      env.render()
      obs = obs.reshape(1,25)
      #predict action
      '''
      ==predict(obs)==
      '''
      obs, reward, done, truncated, info = env.step(int(action))
      total_reward += reward

  return int(total_reward)



if __name__ == "__main__":
  rewards = []
  for round in range(0,10):
    reward = main()
    rewards.append(reward)   
  print(rewards)