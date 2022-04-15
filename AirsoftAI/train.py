from stable_baselines3 import PPO
import gym_airsoft
from wandb.integration.sb3 import WandbCallback
import wandb

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 5000,
    "env_name": "CartPole-v1",
}

run = wandb.init(
    project="AirsoftAI",
    config=config,
    sync_tensorboard=True,
    save_code=True
)

import random
from collections import deque
import os

models_dir = "models/PPO"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

TIMESTEPS = 5_000
SELF_PLAY_UPDATE = 5000

iters = 106

env = gym_airsoft.AirsoftEnv()
env.reset()

#model = PPO('MlpPolicy', env, verbose=2)
custom = {
    "verbose": 1
}
model = PPO.load("models/PPO/530000.zip", env, custom_objects=custom)

models = deque([], 10)

while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=
    WandbCallback(
        verbose=2
    )
    )
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
    print("SAVED")
    models.append(PPO.load(f"{models_dir}/{TIMESTEPS*iters}", env))

    if random.random() <= 0.5:
        env.selfplay = models[-1]
    else:
        env.selfplay = random.choice(models)

    print(env.selfplay, models)