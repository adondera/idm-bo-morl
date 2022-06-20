import torch
import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d


class Experiment_SO:
    def __init__(self, learner, buffer, env, params, device, uncertainty=None):
        self.learner = learner
        self.buffer = buffer
        self.env = env
        self.device = device
        self.losses = []
        self.episode_lengths = []
        self.global_rewards = []
        self.max_episodes = params.get('max_episodes', int(1E6))
        self.max_steps = params.get('max_steps', int(1E9))
        self.grad_repeats = params.get('grad_repeats', 1)
        self.batch_size = params.get('batch_size', 1024)
        self.epi_len = params.get('max_episode_length', 500)
        self.render_step = params.get('render_step', 100)

        # Plot setup
        self.plot_frequency = params.get('plot_frequency', 100)
        self.plot_train_samples = params.get('plot_train_samples', True)
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(6, 10))
        plt.ion()
        plt.draw()

        # Uncertainty
        self.uncertainty = uncertainty
        self.intrinsic_reward = params.get('intrinsic_reward', True) and uncertainty is not None

    def _learn_from_episode(self, episode):
        self.buffer.store_episode(episode)
        self.uncertainty.observe(episode[3], torch.ones(episode[3].shape))
        if self.buffer.current_size >= self.batch_size:
            total_loss = 0
            for i in range(self.grad_repeats):
                sampled_batch, idxs = self.buffer.sample(self.batch_size)
                if self.intrinsic_reward:
                    unc = self.uncertainty(
                        sampled_batch['next_states'],
                        torch.ones(sampled_batch['next_states'].shape)
                    ).unsqueeze(dim=-1)
                    sampled_batch['rewards'] += unc
                    print('%.02g' % (sampled_batch['rewards'].mean().item() + 1), end=' ')
                total_loss += self.learner.train(sampled_batch)
                # returned the averaged loss
            return total_loss / self.grad_repeats
        else:
            return None

    def _run_episode(self, render=False):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state = self.env.reset()
        total_steps = 0
        globalReward = 0
        for t in range(self.epi_len):
            action = self.learner.choose_action(torch.from_numpy(state))
            next_state, (r, z), done, info = self.env.step(action)
            # Store transitions in replay buffer
            states.append(state)
            actions.append(action)
            rewards.append(z)
            next_states.append(next_state)
            dones.append(done)

            state = next_state

            if render:
                self.env.render()
                time.sleep(0.01)

            globalReward += z
            total_steps += 1

            if done:
                break

        episode_batch = list(
            map(lambda x: torch.tensor(x).to(self.device), [states, actions, rewards, next_states])
        )
        return {
            "batch": episode_batch,
            "env_steps": total_steps,
            "global_reward": globalReward,
        }

    def run(self):
        env_steps = 0
        for e in tqdm.tqdm(range(self.max_episodes)):
            render = self.render_step > 0 and (e + 1) % self.render_step == 0
            episode = self._run_episode(render)
            env_steps += episode['env_steps']
            loss = self._learn_from_episode(episode['batch'])
            if loss is not None:
                self.losses.append(loss)
            self.episode_lengths.append(episode['env_steps'])
            self.global_rewards.append(episode['global_reward'])
            if self.plot_frequency is not None and (e + 1) % self.plot_frequency == 0 \
                    and len(self.losses) > 2:
                if self.plot_train_samples:
                    self.plot(env_steps)
                else:
                    self.plot()
            if env_steps >= self.max_steps:
                break

    def evaluate(self, num_episodes=10):
        env_steps = 0
        # for e in tqdm.tqdm():
        global_rewards = []
        with torch.no_grad():
            for _ in range(num_episodes):
                episode = self._run_episode(render=False)
                env_steps += episode["env_steps"]
                global_rewards.append(episode["global_reward"])
        return np.average(global_rewards)

    def plot(self, current_steps=None):
        current_steps = current_steps if current_steps is not None else len(self.episode_lengths)
        self.ax1.clear()
        self.ax1.plot(
            np.linspace(1, current_steps, num=len(self.losses)),
            uniform_filter1d(self.losses, size=100, output=float),
            color='blue',
            label='Average loss')
        self.ax1.legend()

        self.ax2.clear()
        self.ax2.plot(
            np.linspace(1, current_steps, num=len(self.global_rewards)),
            uniform_filter1d(self.global_rewards, size=100, output=float),
            color='black',
            label='Cumulative global reward')
        self.ax2.legend()

        self.ax3.clear()
        self.ax3.plot(
            np.linspace(1, current_steps, num=len(self.episode_lengths)),
            uniform_filter1d(self.episode_lengths, size=100, output=float),
            color='yellow',
            label='Episode length')
        self.ax3.legend()

        plt.draw()
        plt.pause(0.02)
