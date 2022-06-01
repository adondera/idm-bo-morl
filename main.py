import gym
import torch
import matplotlib
import numpy as np

from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCarNormal
from replay_buffer import ReplayBuffer
from experiment import Experiment
from config import default_params
from dqn import DQN
from RND import RNDUncertainty

matplotlib.use('TkAgg')


class RescaledEnv:
    def __init__(self, env, max_episode_length=None):
        self.env = env
        self.bounds = [(l, h) for l, h in zip(env.observation_space.low, env.observation_space.high)]
        if max_episode_length is not None: self.env._max_episode_steps = max_episode_length

    def rescale(self, state):
        return np.array([2 * (x - l) / (h - l) - 1 for x, (l, h) in zip(state, self.bounds)])

    def step(self, action):
        ns, r, d, x = self.env.step(action)
        return self.rescale(ns), r, d, x

    def reset(self):
        return self.rescale(self.env.reset())

    def render(self, mode="human"):
        self.env.render(mode)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        return self.env.seed()

# env = DiscreteMountainCar3Distance(gym.make("MountainCar-v0"))
# env = PendulumRewardWrapper(gym.make("Pendulum-v1")) #this one doesn't work yet, because it has no env.action_space.n
# env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

env = DiscreteMountainCarNormal(gym.make("MountainCar-v0"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "reward_names": (env.__dict__["reward_names"], torch.StringType),
    "dones": ((1,), torch.bool)
}

config_params = default_params()
config_params['max_steps'] = int(2E6)
config_params['intrinsic_reward'] = True
config_params['k'] = 0

preference = np.array([1.0], dtype=np.single)

model = torch.nn.Sequential(
    torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 215), torch.nn.ReLU(),
    torch.nn.Linear(215, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 1024), torch.nn.ReLU(),
    torch.nn.Linear(1024, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 215), torch.nn.ReLU(),
    torch.nn.Linear(215, env.action_space.n))

learner = DQN(model, config_params, device, env)
buffer = ReplayBuffer(env_params, buffer_size=int(1e5), device=device, k=config_params.get('k', 1))
rnd = RNDUncertainty(400, 2, device)
experiment = Experiment(learner=learner, buffer=buffer, env=env, reward_dim=env_params["rewards"][0][0],
                        preference=preference, params=config_params, device=device, uncertainty=rnd)
experiment.run()
