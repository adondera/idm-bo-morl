import gym
import matplotlib
import numpy as np
import torch

from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCarNormal
from replay_buffer import ReplayBuffer
from experiment import Experiment
from config import default_params
from dqn import DQN
from RND import RNDUncertainty

matplotlib.use("Qt5agg")

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
config_params["intrinsic_reward"] = True
config_params["uncertainty_scale"] = 400
config_params["k"] = 0
config_params['max_steps'] = int(2E6)
config_params["grad_repeats"] = int(10)

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
rnd = RNDUncertainty(config_params.get("uncertainty_scale"), 2, device)
experiment = Experiment(learner=learner, buffer=buffer, env=env, reward_dim=env_params["rewards"][0][0],
                        preference=preference, params=config_params, device=device, uncertainty=rnd)
experiment.run()
