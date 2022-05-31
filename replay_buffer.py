import threading
import skopt
import torch
from numpy import *
import nobo

"""
The replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer:
    def __init__(self, env_params, buffer_size, device, k=1):
        self.device = device
        self.env_params = env_params
        self.size = buffer_size
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {
            'states': torch.empty((self.size, *self.env_params['states'][0]), dtype=torch.float32, device=self.device),
            'actions': torch.empty((self.size, *self.env_params['actions'][0]), dtype=torch.long, device=self.device),
            'rewards': torch.empty((self.size, *self.env_params['rewards'][0]), dtype=torch.float32,
                                   device=self.device),
            'preferences': torch.empty((self.size, *self.env_params['preferences'][0]),
                                       dtype=torch.float32, device=self.device),
            'next_states': torch.empty((self.size, *self.env_params['states'][0]), dtype=torch.float32,
                                       device=self.device),
            'dones': torch.empty((self.size, 1), dtype=torch.bool, device=self.device),
        }
        # thread lock
        self.lock = threading.Lock()
        self.position = 0
        self.k = k
        self.norm = torch.zeros(self.env_params['rewards'][0], device=device)

    def bayes_opt(self, output_space):
        bo = nobo.Optimizer([skopt.utils.Real(0, 1), skopt.utils.Real(0, 1)])
        for i in range(50):
            x = bo.ask(verbose=False)
            y = 0.1 * mean(output_space) + 0.9 * output_space[-1]
            bo.tell(x, y)
        x, lo, y, hi = bo.get_best()
        return torch.tensor([x])

    # def bayes_opt(self,input_space,output_space):
    #     bo=nobo.Optimizer(input_space)
    #     for i in range(50):
    #         x = bo.ask(verbose=False)
    #         y = 0.1*mean(output_space[:input_space.index(x)])+0.9*output_space[input_space.index(x)]
    #         bo.tell(x, y)
    #     x, lo, y, hi = bo.get_best()
    #     return torch.tensor([x])

    # Store the episode
    def store_episode(self, episode_batch, criterion):
        mb_state, mb_action, mb_reward, mb_preference, mb_next_state = episode_batch
        batch_size = mb_state.shape[0]

        # TODO: Is it better to add the sampled preferences in the experience buffer when storing or when sampling?
        # Adds a new randomly sampled normally distributed preference vector for each transition
        if len(criterion) <= 10:
            mb_new_preferences = torch.abs(torch.randn(batch_size * self.k, mb_preference.shape[1], device=self.device))
            mb_new_preferences = torch.nn.functional.normalize(mb_new_preferences, p=1, dim=1)
        else:
            mb_new_preferences = self.bayes_opt(criterion)
        # mb_new_preferences = torch.rand(batch_size * self.k, mb_preference.shape[1], device=self.device)

        with self.lock:
            idxs = self._get_storage_idx(inc=(self.k + 1) * batch_size)
            # store the informations
            self.buffers['states'][idxs] = torch.repeat_interleave(mb_state, self.k + 1, 0)
            self.buffers['actions'][idxs] = torch.repeat_interleave(mb_action, self.k + 1, 0).reshape(
                (self.k + 1) * batch_size, 1)
            rewards = torch.repeat_interleave(mb_reward.to(torch.float32), self.k + 1, 0)
            self.buffers['rewards'][idxs] = rewards
            print("new:", mb_new_preferences.shape)
            self.buffers['preferences'][idxs] = torch.cat((mb_preference, mb_new_preferences))
            self.buffers['next_states'][idxs] = torch.repeat_interleave(mb_next_state, self.k + 1, 0)
            self.n_transitions_stored += batch_size * (self.k + 1)
            # update the normalization constant
            self.norm = self.norm * 0.5 + 0.5 * torch.mean(rewards, dim=0)

    # Sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        idxs = torch.randint(0, self.current_size, (batch_size,))
        # Sample transitions
        transitions = {key: temp_buffers[key][idxs] for key in temp_buffers.keys()}
        # multiply by the normlization constant to normalize rewards
        transitions['rewards'] = transitions['rewards'] / abs(self.norm)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        idxs = torch.arange(self.position, self.position + inc) % self.size
        self.position += inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idxs[0]
        return idxs
