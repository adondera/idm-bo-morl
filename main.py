import gym
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCar3Distance
from replay_buffer import ReplayBuffer
from config import default_params
from dqn import DQN

matplotlib.use('TkAgg')

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

plt.ion()

fig = plt.figure()
plt.draw()

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
plot_frequency = 1000
epsilon = 0.1

buffer = ReplayBuffer(env_params, buffer_size=int(1e5))

# TODO: Select preference based on BO
preference = np.array([1.0, 1.0, 1.0], dtype=np.single)


state = env.reset()
states, actions, rewards, preferences, next_states, dones = [], [], [], [], [], []

# TODO: Add model checkpoints.
# TODO: Add metrics and plots
losses = []
for i in range(total_timesteps):
    # Select action and perform env step
    if np.random.rand() < epsilon:
        action = random.randint(0, env.action_space.n - 1)
    else:
        action = learner.get_greedy_value(torch.from_numpy(state), torch.from_numpy(preference))
    next_state, (r, z), done, info = env.step(action)  # TODO: Do something with Z
    # Store transitions in replay buffer
    states.append(state)
    actions.append(action)
    rewards.append(r)
    preferences.append(preference)
    next_states.append(next_state)
    dones.append(done)  # TODO: Check correctness here (maybe off by 1 errors)

    # Update network after `update_step` steps have been performed
    if (i + 1) % update_step == 0:
        # Convert observations to a list of tensors and store
        episode_batch = list(map(lambda x: torch.tensor(x), [states, actions, rewards, preferences, next_states]))
        buffer.store_episode(episode_batch)

        batch = buffer.sample(batch_size)

        loss = learner.train(batch)
        losses.append(loss)

        # Reset lists after content has been stored in replay buffer
        states, actions, rewards, preferences, next_states = [], [], [], [], []

    if (i + 1) % plot_frequency == 0:
        plt.plot(losses, color='blue')
        plt.draw()
        plt.pause(0.02)

plt.ioff()
plt.show()
