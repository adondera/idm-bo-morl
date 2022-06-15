import os
import gym
import matplotlib
import numpy as np
import torch
import wandb
from SO_RL.dqn_SO import DQN_SO
from SO_RL.experiment_SO import Experiment_SO
from SO_RL.replay_buffer_SO import ReplayBuffer_SO
from wrappers.cartpole_v1_wrapper import (CartPoleNoisyRewardWrapper,
                                          SparseCartpole,
                                          CartPoleV1AngleNegEnergyRewardWrapper,
                                          CartPoleV1AnglePosEnergyRewardWrapper)
from wrappers.mountaincar_discrete_wrapper import DiscreteMountainCarVelocityDistance
from wrappers.mo_wrapper import RescaledReward

from replay_buffer import ReplayBuffer
from experiment import Experiment
from config import default_params
from dqn import DQN
from RND import RNDUncertainty


def metric_fun(x):
    return np.average(x[int(len(x) * 9 / 10):]),


if os.environ.get("DESKTOP_SESSION") == "i3":
    matplotlib.use("tkagg")
else:
    matplotlib.use("Qt5agg")

# env = RescaledReward(SparseCartpole(CartPoleV1AngleNegEnergyRewardWrapper(gym.make("CartPole-v1"))), [1, 1/5])
env = RescaledReward(DiscreteMountainCarVelocityDistance(gym.make("MountainCar-v0")), [10, 1])

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
config_params['max_steps'] = int(2E5)
config_params['max_episodes'] = int(1e3)
config_params["grad_repeats"] = int(10)
config_params['render_step'] = 100
preference = np.array([1.0, 0.0], dtype=np.single)

multi_objective = True
tag = "MO_DQN" if multi_objective else "SO_DQN"

wandb.init(project="test-project", entity="idm-morl-bo", tags=[tag, env.spec.id], config=config_params)
preference_table = wandb.Table(columns=[i for i in range(env.numberPreferences)])
preference_table.add_data(*preference)
wandb.log({
    "Preference": preference_table
})

if multi_objective:
    model = torch.nn.Sequential(
        torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, env.action_space.n))

    learner = DQN(model, config_params, device, env)
    buffer = ReplayBuffer(env_params, buffer_size=int(1e5), device=device, k=config_params.get('k', 1))
    rnd = RNDUncertainty(config_params.get("uncertainty_scale"), env.observation_space.shape[0], device)
    experiment = Experiment(learner=learner, buffer=buffer, env=env, reward_dim=env_params["rewards"][0][0],
                            preference=preference, params=config_params, device=device, uncertainty=rnd)
else:
    model = torch.nn.Sequential(
        torch.nn.Linear(env_params['states'][0][0] + env_params["rewards"][0][0], 128), torch.nn.ReLU(),
        torch.nn.Linear(128, 512), torch.nn.ReLU(),
        torch.nn.Linear(512, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, env.action_space.n))

    learner = DQN_SO(model, config_params, device, env)
    buffer = ReplayBuffer_SO(env_params, buffer_size=int(1e5), device=device)
    rnd = RNDUncertainty(config_params.get("uncertainty_scale"), env_params['states'][0][0], device)
    experiment = Experiment_SO(learner=learner, buffer=buffer, env=env, params=config_params, device=device,
                               uncertainty=rnd)

experiment.run()

wandb.log({
    f"Experiment plot": experiment.fig,
    f"Episode length metric": metric_fun(experiment.episode_lengths)
})

wandb.run.log_code()
wandb.run.summary["Global reward metric"] = metric_fun(experiment.global_rewards)[0]
