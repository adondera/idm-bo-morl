import tqdm
import gym
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCar3Distance
from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from wrappers.pendulum_wrapper import PendulumRewardWrapper
from replay_buffer import ReplayBuffer
from config import default_params
from dqn import DQN
from scipy.ndimage.filters import uniform_filter1d


matplotlib.use('TkAgg')

# env = DiscreteMountainCar3Distance(gym.make("MountainCar-v0"))
# env = PendulumRewardWrapper(gym.make("Pendulum-v1")) #this one doesn't work yet, because it has no env.action_space.n
env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "reward_names": (env.__dict__["reward_names"], torch.StringType),
    "dones": ((1,), torch.bool)
}

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1)

plt.ion()

plt.draw()

config_params = default_params()

model = torch.nn.Sequential(
    torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 215), torch.nn.ReLU(),
    torch.nn.Linear(215, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 1024), torch.nn.ReLU(),
    torch.nn.Linear(1024, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 215), torch.nn.ReLU(),
    torch.nn.Linear(215, env.action_space.n))

learner = DQN(model, config_params, device)
buffer = ReplayBuffer(env_params, buffer_size=int(1e5), device=device, k=config_params.get('k', 1))

total_timesteps = int(2e5)
update_step = 50
batch_size = 32
plot_frequency = 1000
epsilon = 0.3

# TODO: Select preference based on BO. Make sure they have the same amount of parameters
preference = np.array([1.0, 0.0], dtype=np.single)
assert preference.shape == env_params["preferences"][0]

state = env.reset()
states, actions, rewards, preferences, next_states, dones = [], [], [], [], [], []

# TODO: Add model checkpoints.
losses = []
rewardStats = []
for i in range(env_params["rewards"][0][0]):
    rewardStats.append([])
rewardsGlobal = []
episodeLengths = []
colors = ['green', 'red', 'purple', 'orange']
globalReward = 0
episodeLength = 0
for i in tqdm.tqdm(range(total_timesteps)):
    # Select action and perform env step

    # this does normal epsilon exploration
    # if np.random.rand() < epsilon:

    # this does linearly decaying epilon greedy exploration
    if np.random.rand() < (epsilon - epsilon * (i / total_timesteps) / 2):  # TODO: use better exploration method e.g.
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
    dones.append(done)

    state = next_state

    # for tracking cumulative global reward and episode length at end of episode
    globalReward += z
    episodeLength += 1
    if done:
        rewardsGlobal.append(globalReward)
        episodeLengths.append(episodeLength)
        globalReward = 0
        episodeLength = 0

    # Update network after `update_step` steps have been performed
    if (i + 1) % update_step == 0:
        # Convert observations to a list of tensors and store
        episode_batch = list(
            map(lambda x: torch.tensor(x).to(device), [states, actions, rewards, preferences, next_states])
        )
        buffer.store_episode(episode_batch)

        batch = buffer.sample(batch_size)

        loss = learner.train(batch)

        # for statistics
        losses.append(loss)
        for j in range(env_params["rewards"][0][0]):
            rewardStats[j].append(r[j])

        # Reset lists after content has been stored in replay buffer
        states, actions, rewards, preferences, next_states = [], [], [], [], []

    # If you want the code to run faster comment these lines to disable the interactive plot
    if (i + 1) % plot_frequency == 0:
        ax1.clear()
        ax1.plot(uniform_filter1d(losses, size=100), color='blue', label='loss')
        for j in range(env_params["rewards"][0][0]):
            ax2.plot(rewardStats[j], color=colors[j], label=env_params["reward_names"][0][j])
        ax3.clear()
        ax3.plot(uniform_filter1d(rewardsGlobal, size=100), color='black', label='global reward')
        ax4.clear()
        ax4.plot(uniform_filter1d(episodeLengths, size=100), color='yellow', label='episode length')
        # add legend on first iteration
        if i == 999:
            plt.legend()
        plt.draw()
        plt.pause(0.02)

    # If the episode ends reset the environment
    if done:
        state = env.reset()

ax1.clear()
ax1.plot(uniform_filter1d(losses, size=100), color='blue', label='loss')
for j in range(env_params["rewards"][0][0]):
    ax2.plot(rewardStats[j], color=colors[j], label=env_params["reward_names"][0][j])
ax3.clear()
ax3.plot(uniform_filter1d(rewardsGlobal, size=100), color='black', label='global reward')
ax4.clear()
ax4.plot(uniform_filter1d(episodeLengths, size=100), color='yellow', label='episode length')
# # plt.legend()
plt.draw()
plt.ioff()
plt.show()
#
# plt.plot(rewardsGlobal, color='black', label='global reward')
# plt.plot(episodeLengths, color='yellow', label='episode length')
# plt.legend()
# plt.draw()
# plt.ioff()
# plt.show()
