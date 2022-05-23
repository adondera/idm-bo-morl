import gym
import random
import numpy as np
import torch

from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCar3Distance
from replay_buffer import ReplayBuffer
from config import default_params
from dqn import DQN

env = DiscreteMountainCar3Distance(gym.make(("MountainCar-v0")))

# TODO: Add GPU support
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((3,), torch.float32),
    "preferences": ((3,), torch.float32),
    "dones": ((1,), torch.bool)
}

config_params = default_params()

model = torch.nn.Sequential(
    torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, env.action_space.n))

learner = DQN(model, config_params)

total_timesteps = int(1e5)
update_step = 50
batch_size = 32

buffer = ReplayBuffer(env_params, buffer_size=int(1e5))
preference = (1.0, 1.0, 1.0)

state = env.reset()
states, actions, rewards, preferences, next_states, dones = [], [], [], [], [], []
for i in range(total_timesteps):
    action = random.randint(0, 2)  # TODO: Replace with epsilon greedy
    next_state, (r, z), done, info = env.step(action)

    # Keep track of transitions
    states.append(state)
    actions.append(action)
    rewards.append(r)
    preferences.append(preference)
    next_states.append(next_state)
    dones.append(done)  # TODO: Check correctness here (maybe off by 1 errors)

    if (i + 1) % update_step == 0:
        # TODO: Update network
        episode_batch = list(map(lambda x: torch.tensor(x), [states, actions, rewards, preferences, next_states]))
        buffer.store_episode(episode_batch)

        batch = buffer.sample(batch_size)

        learner.train(batch)

        states, actions, rewards, preferences, next_states = [], [], [], [], []
        exit()
