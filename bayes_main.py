import gym
import torch
import matplotlib
from sklearn.gaussian_process.kernels import Matern

from wrappers.cartpole_v1_wrapper import CartPoleV1AngleEnergyRewardWrapper
from replay_buffer import ReplayBuffer
from config import default_params
from bayes_experiment import BayesExperiment

from BayesianOptimization.bayes_opt import BayesianOptimization, UtilityFunction

matplotlib.use("Qt5agg")

env = CartPoleV1AngleEnergyRewardWrapper(gym.make("CartPole-v1"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env_params = {
    "states": (env.observation_space.shape, torch.float32),
    "actions": ((1,), torch.long),
    "rewards": ((env.__dict__["numberPreferences"],), torch.float32),
    "preferences": ((env.__dict__["numberPreferences"],), torch.float32),
    "reward_names": (env.__dict__["reward_names"], torch.StringType),
    "dones": ((1,), torch.bool),
}

model = torch.nn.Sequential(
    torch.nn.Linear(env_params["states"][0][0] + env_params["rewards"][0][0], 215),
    torch.nn.ReLU(),
    torch.nn.Linear(215, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 1024),
    torch.nn.ReLU(),
    torch.nn.Linear(1024, 512),
    torch.nn.ReLU(),
    torch.nn.Linear(512, 215),
    torch.nn.ReLU(),
    torch.nn.Linear(215, env.action_space.n),
)

# These configuration parameters need to be changed depending on the environment
config_params = default_params()
config_params["render_step"] = 0  # Change this to something like 100 if you want to render the environment

# Change these two to tweak the intrinsic rewards with RND. The higher the uncertainty scale,
# the more important intrinsic rewards will be. Depends from environment to environment, requires tweaking.
config_params["intrinsic_reward"] = False
config_params["uncertainty_scale"] = 0

# These parameters refer to the DDQN agent. Again, dependent on the environment.
config_params["k"] = 5
config_params["max_episodes"] = int(5e2)
config_params["grad_repeats"] = int(1)
config_params['max_steps'] = 1e5

# These parameters affect the bayesian optimization process
alpha = 0.1
length_scale = 0.1
xi = 0.1
kappa = 2.5
nu = 2.5

optimizer = BayesianOptimization(
    f=None,
    kernel=Matern(length_scale_bounds="fixed", length_scale=length_scale, nu=nu),
    pbounds={"x": (0, 1)},
    verbose=2,
    random_state=1,
)
utility = UtilityFunction(kind="ucb", kappa=kappa, xi=xi)
optimizer.set_gp_params(alpha=alpha)

buffer = ReplayBuffer(
    env_params, buffer_size=int(1e5), device=device, k=config_params.get("k", 1)
)

number_of_bayes_episodes = 10

bayes_experiment = BayesExperiment(
    optimizer=optimizer,
    utility=utility,
    model=model,
    buffer=buffer,
    config_params=config_params,
    device=device,
    env=env,
    env_params=env_params
)

bayes_experiment.run(number_of_bayes_episodes)
