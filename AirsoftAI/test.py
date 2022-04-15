from stable_baselines3 import PPO
import gym_airsoft

env = gym_airsoft.AirsoftEnv()

model = PPO.load("models/PPO/595000.zip", env)
env.selfplay = PPO.load("models/PPO/530000.zip", env)

done = False
obs = env.reset()

while not done:
    action, _ = model.predict(obs)
    obs, reward, done, info = env.step(action)
    env.render()

print(reward)