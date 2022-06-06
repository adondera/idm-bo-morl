import matplotlib.pyplot as plt
import numpy as np

from dqn import DQN
from experiment import Experiment
from RND import RNDUncertainty


from n_sphere import n_sphere
import torch
from math import pi


def reduce_dim(x: np.array or torch.tensor):
    """
    Project preference x to (n-1)-dimensional space
    """
    if type(x) == torch.Tensor:
        x = x.numpy()
    spherical_proj = n_sphere.convert_spherical(x)
    return spherical_proj[1:]


def increase_dim(x: dict or np.array):
    """
    Recover n-dimensional preference from (n-1)-dimensional preference
    """
    angles = x
    if type(x) == dict:
        angles = np.array(list(x.values()))
    # 1.0 is the norm/radius, x.values() are the angles
    l = torch.tensor(np.concatenate(([1.0], angles)), dtype=torch.float32)
    rectangular_proj = n_sphere.convert_rectangular(l)
    return torch.nn.functional.normalize(rectangular_proj, p=1.0, dim=0).numpy()


class BayesExperiment:
    def __init__(
        self,
        optimizer,
        utility,
        model,
        buffer,
        config_params,
        device,
        env,
        env_params,
        pbounds=(-pi, pi),
        metric_fun=lambda x: np.average(x[int(len(x) * 9 / 10) :]),
    ):
        self.optimizer = optimizer
        self.utility = utility
        self.model = model
        self.buffer = buffer
        self.config_params = config_params
        self.device = device
        self.env = env
        self.env_params = env_params
        self.global_rewards = []
        self.uncertainty_scale = config_params.get("uncertainty_scale", 0)
        self.pbounds = pbounds
        self.metric_fun = metric_fun

        self.alpha = 0.1

        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 5))

    def plot_bo(self, f=None):
        x = np.linspace(self.pbounds[0], self.pbounds[1], 4000)
        mean, sigma = self.optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
        self.ax.clear()
        if f:
            plt.plot(x, f(x))
        self.ax.plot(x, mean)
        self.ax.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
        self.ax.scatter(
            self.optimizer.space.params.flatten(),
            self.optimizer.space.target,
            c="red",
            s=50,
            zorder=10,
        )
        plt.draw()

    def run(self, number_of_experiments):
        for _ in range(number_of_experiments):
            learner = DQN(self.model, self.config_params, self.device, self.env)
            rnd = RNDUncertainty(
                self.uncertainty_scale,
                input_dim=self.env_params["states"][0][0],
                device=self.device,
            )
            next_preference_proj = self.optimizer.suggest(self.utility)
            # TODO make preferences work with dim!=2
            print(next_preference_proj)
            next_preference = increase_dim(next_preference_proj)
            print("Next preference to probe is:", next_preference)
            experiment = Experiment(
                learner=learner,
                buffer=self.buffer,
                env=self.env,
                reward_dim=self.env_params["rewards"][0][0],
                preference=next_preference,
                params=self.config_params,
                device=self.device,
                uncertainty=rnd,
            )
            experiment.run()
            global_rewards_experiment = experiment.global_rewards

            metric = self.metric_fun(global_rewards_experiment)
            self.global_rewards.append(metric)
            self.optimizer.register(params=next_preference_proj, target=metric)
            self.optimizer.suggest(self.utility)
            self.plot_bo()
