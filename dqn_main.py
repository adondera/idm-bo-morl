import os
import gym
import matplotlib
import numpy as np
import torch
import wandb
from wrappers.cartpole_v1_wrapper import CartPoleNoisyRewardWrapper, SparseCartpole
from wrappers.mo_wrapper import RescaledReward

from replay_buffer import ReplayBuffer
from experiment import Experiment
from config import default_params
from dqn import DQN
from RND import RNDUncertainty

# if os.environ.get("DESKTOP_SESSION") == "i3":
#     matplotlib.use("tkagg")
# else:
matplotlib.use("Qt5agg")

env = RescaledReward(SparseCartpole(CartPoleNoisyRewardWrapper(gym.make("CartPole-v1"))))

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
config_params["intrinsic_reward"] = False
config_params["uncertainty_scale"] = 400
config_params["k"] = 5
config_params['max_steps'] = int(2E5)
config_params['max_episodes'] = int(1e4)
config_params["grad_repeats"] = int(1)
config_params['render_step'] = 0

wandb.init(project="test-project", entity="idm-morl-bo", tags=["DQN", env.spec.id], config=config_params)

preference = np.array([1., 0.], dtype=np.single)

wandb.log({
    "Preference": preference
})

model = torch.nn.Sequential(
    torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 215), torch.nn.ReLU(),
    torch.nn.Linear(215, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 1024), torch.nn.ReLU(),
    torch.nn.Linear(1024, 512), torch.nn.ReLU(),
    torch.nn.Linear(512, 215), torch.nn.ReLU(),
    torch.nn.Linear(215, env.action_space.n))

learner = DQN(model, config_params, device, env)
buffer = ReplayBuffer(env_params, buffer_size=int(1e5), device=device, k=config_params.get('k', 1))
rnd = RNDUncertainty(config_params.get("uncertainty_scale"), env.observation_space.shape[0], device)
experiment = Experiment(learner=learner, buffer=buffer, env=env, reward_dim=env_params["rewards"][0][0],
                        preference=preference, params=config_params, device=device, uncertainty=rnd)
experiment.run()
wandb.log({
    f"Experiment plot": experiment.fig,
})
