import tqdm
import gym
import random
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCar3Distance
from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from replay_buffer import ReplayBuffer
from config import default_params
from dqn import DQN

matplotlib.use('TkAgg')

# env = DiscreteMountainCar3Distance(gym.make("CartPole-v1"))
env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

# TODO: Get the reward anr preference shapes from the environment
env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((env.action_space.n,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "dones": ((1,), torch.bool)
}

# env_params = {
#     "states": (env.observation_space.shape, torch.float32),
#     "actions": (env.action_space.n, torch.long),
#     "rewards": (env.__dict__["numberPreferences"], torch.float32),
#     "preferences": (env.__dict__["numberPreferences"], torch.float32),
#     "dones": ((1,), torch.bool)
# }

plt.ion()

fig = plt.figure()
plt.draw()

config_params = default_params()

model = torch.nn.Sequential(
    torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 128), torch.nn.ReLU(),
    torch.nn.Linear(128, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 128), torch.nn.ReLU(),
    torch.nn.Linear(128, env.action_space.n))

learner = DQN(model, config_params, device)
buffer = ReplayBuffer(env_params, buffer_size=int(1e5), device=device)

total_timesteps = int(1e5)
update_step = 50
batch_size = 32
plot_frequency = 1000
epsilon = 0.1


# TODO: Select preference based on BO. Make sure they have the same amount of parameters
preference = np.array([1.0, 0.0], dtype=np.single)
assert preference.shape == env_params["preferences"][0]

state = env.reset()
states, actions, rewards, preferences, next_states, dones = [], [], [], [], [], []

# TODO: Add model checkpoints.
losses = []
rewards = []
rewards = []
# rewards3 = []
rewardsGlobal = []
episodeLengths = []
globalReward = 0
episodeLength = 0
for i in tqdm.tqdm(range(total_timesteps)):
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
    dones.append(done)

    state = next_state

    # for acummulating global reward at end of episode
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
            map(lambda x: torch.tensor(x).to(device), [states, actions, rewards, preferences, next_states]))
        buffer.store_episode(episode_batch)

        batch = buffer.sample(batch_size)

        loss = learner.train(batch)

        # for statistics
        losses.append(loss)
        rewards1.append(r[0])
        rewards2.append(r[1])
        
        # rewards3.append(r[2])
        rewardsGlobal.append(z)

        # Reset lists after content has been stored in replay buffer
        states, actions, rewards, preferences, next_states = [], [], [], [], []

    # If you want the code to run faster comment these lines to disable the interactive plot
    if (i + 1) % plot_frequency == 0:
        plt.plot(losses, color='blue', label='loss')
        plt.plot(rewards1, color='red', label='angle')
        plt.plot(rewards2, color='green', label='- energy')
        # plt.plot(rewards3, color='yellow', label='- distance to right')
        # plt.plot(rewardsGlobal, color='black', label='global reward')
        # plt.plot(episodeLengths, color='yellow', label='episode length')
        #add legend on first iteration
        if i == 999:
            plt.legend()
        plt.draw()
        plt.pause(0.02)

    # If the episode ends reset the environment
    if done:
        state = env.reset()

plt.plot(losses, color='blue', label='loss')
plt.plot(rewards1, color='red', label='angle')
plt.plot(rewards2, color='green', label='- energy')
# plt.plot(rewards3, color='yellow', label='- distance to right')
# plt.legend()
plt.draw()
plt.ioff()
plt.show()

plt.plot(rewardsGlobal, color='black', label='global reward')
plt.plot(episodeLengths, color='yellow', label='episode length')
plt.draw()
plt.ioff()
plt.show()