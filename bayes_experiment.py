from unicodedata import numeric
import matplotlib.pyplot as plt
import numpy as np
import wandb

from dqn import DQN
from experiment import Experiment
from RND import RNDUncertainty

from n_sphere import n_sphere
import torch
from math import pi
import scipy.stats

from spherical_coords import reduce_dim, increase_dim


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
            dirichlet_alpha: np.array = None,
            metric_fun=lambda x: np.average(x[int(len(x) * 9 / 10):]),
    ):
        self.optimizer = optimizer
        self.utility = utility
        self.model = model
        self.buffer = buffer
        self.config_params = config_params
        self.config_params_original = config_params.copy()
        self.device = device
        self.env = env
        self.env_params = env_params
        self.global_rewards = []
        self.uncertainty_scale = config_params.get("uncertainty_scale", 0)
        self.pbounds = pbounds
        self.metric_fun = metric_fun
        self.discarded_rewards = ([], [])

        self.numberPreferences = int(env_params["preferences"][0][0])

        if not isinstance(dirichlet_alpha, np.ndarray) or len(dirichlet_alpha) != self.numberPreferences:
            self.dirichlet_alpha = np.repeat(2.0, self.numberPreferences)
        else:
            self.dirichlet_alpha = dirichlet_alpha

        # TODO make into param
        self.prior = scipy.stats.dirichlet(alpha=self.dirichlet_alpha)

        wandb.init(project="test-project", entity="idm-morl-bo",
                   tags=["Bayes"] + self.env.tags,
                   config=self.config_params)
        wandb.run.log_code()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 5))

    def plot_bo(self, f=None):
        x = np.linspace(self.pbounds[0], self.pbounds[1], 4000)
        mean, sigma = self.optimizer._gp.predict(x.reshape(-1, 1), return_std=True)
        self.ax.clear()
        if f:
            plt.plot(x, f(x))
        self.ax.plot(x, mean)
        # acq_max = self.optimizer
        self.ax.fill_between(x, mean + sigma, mean - sigma, alpha=0.1)
        self.ax.scatter(
            self.optimizer.space.params.flatten(),
            self.optimizer.space.target,
            c="red",
            s=50,
            zorder=10,
        )
        self.ax.scatter(self.discarded_rewards[0], self.discarded_rewards[1], c="red", marker='X')
        plt.draw()

    def run(self, number_of_experiments=None):
        preference_table = wandb.Table(columns=[i for i in range(self.env.numberPreferences)])
        if number_of_experiments is None:
            number_of_experiments = self.config_params.get("number_BO_experiments", 20)

        discarded_experiments = self.config_params.get("discarded_experiments", int(number_of_experiments / 10))
        discarded_experiments_length_factor = self.config_params.get("discarded_experiments_length_factor", 1.0)
        prior_only_experiments = self.config_params.get("prior_only_experiments", int(number_of_experiments / 5))
        print(f"Running {discarded_experiments}")
        for experiment_id in range(number_of_experiments):

            prior_only_sample = experiment_id < prior_only_experiments
            discard_sample = experiment_id < discarded_experiments
            length_factor = discarded_experiments_length_factor if discard_sample else 1.0

            self.config_params["max_episodes"] = int(self.config_params_original["max_episodes"] * length_factor)
            self.config_params["max_steps"] = int(self.config_params_original["max_steps"] * length_factor)

            learner = DQN(self.model, self.config_params, self.device, self.env)
            rnd = RNDUncertainty(
                self.uncertainty_scale,
                state_dim=self.env_params["states"][0][0],
                preference_dim=self.config_params["preference_dim"],
                device=self.device,
            )

            if prior_only_sample:
                next_preference = self.prior.rvs(size=1).squeeze()
                next_preference_proj = reduce_dim(next_preference)
            else:
                next_preference_proj = self.optimizer.suggest(self.utility)

            next_preference = increase_dim(next_preference_proj)
            print("Next preference to probe is:", next_preference, " spherical: ", next_preference_proj)
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
            metric = self.evaluate(learner.model, preference=next_preference)
            if not discard_sample:
                self.global_rewards.append(metric)
                self.optimizer.register(params=next_preference_proj, target=metric)
            else:
                self.discarded_rewards[0].append(next_preference_proj)
                self.discarded_rewards[1].append(metric)
                print("Discarding sample: ", next_preference_proj, metric)

            self.optimizer.suggest(self.utility)
            self.plot_bo()
            preference_table.add_data(*next_preference.tolist())
            wandb.log({
                "Target": metric,
                f"Experiment {experiment_id} plot": experiment.fig,
                "BO plot": wandb.Image(self.fig)
            })
            # self.optimizer.suggest(self.utility)
        measured_max = self.optimizer.max
        measured_max = measured_max["target"], measured_max["params"], increase_dim(measured_max["params"])
        # TODO print max of the GP mean
        # gp_max = self.optimizer.space.target.argmax()
        print(
            f"The maximum is: {measured_max[0]}, preference={measured_max[2]} (spherical: {list(measured_max[1].values())})")
        wandb.log({
            "Preferences": preference_table
        })

        wandb.run.summary["Global reward metric"] = measured_max[0]

    # Run an episode by evaluating the greedy policy learned by the agent
    # The policy is deterministic, hence only 1 episode is required to evaluate it
    def evaluate(self, model, preference, num_episodes=10):
        config = self.config_params.copy()
        config["epsilon_start"] = 0
        config["epsilon_finish"] = 0
        global_rewards = []
        new_learner = DQN(model, config, self.device, self.env)
        experiment = Experiment(
            learner=new_learner,
            buffer=None,
            env=self.env,
            reward_dim=self.env_params["rewards"][0][0],
            preference=preference,
            params=self.config_params,
            device=self.device,
            uncertainty=None,
        )
        for _ in range(num_episodes):
            plt.close(experiment.fig)
            results = experiment._run_episode()
            global_rewards.append(results["global_reward"])
        return np.average(global_rewards)
