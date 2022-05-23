import threading
import torch

"""
The replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer:
    def __init__(self, env_params, buffer_size):
        self.env_params = env_params
        self.size = buffer_size
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        # create the buffer to store info
        self.buffers = {'state': torch.empty((self.size, *self.env_params['state'][0]), dtype=torch.float32),
                        'action': torch.empty((self.size, *self.env_params['action'][0]), dtype=torch.long),
                        'reward': torch.empty((self.size, *self.env_params['reward'][0]), dtype=torch.double),
                        'preference': torch.empty((self.size, *self.env_params['preference'][0]),
                                                  dtype=torch.float32),
                        'next_state': torch.empty((self.size, *self.env_params['state'][0]), dtype=torch.float32),
                        }
        # thread lock
        self.lock = threading.Lock()

    # store the episode
    def store_episode(self, episode_batch):
        mb_state, mb_action, mb_reward, mb_preference, mb_next_state = episode_batch
        batch_size = mb_state.shape[0]
        # TODO: Is it better to add the sampled preferences in the experience buffer when storing or when sampling?
        mb_new_preferences = torch.randn(batch_size, mb_preference.shape[1])
        with self.lock:
            idxs = self._get_storage_idx(inc=2 * batch_size)
            # store the informations
            self.buffers['state'][idxs] =  torch.repeat_interleave(mb_state, 2, 0)
            self.buffers['action'][idxs] = torch.repeat_interleave(mb_action, 2, 0).reshape(2 * batch_size, 1)
            self.buffers['reward'][idxs] = torch.repeat_interleave(mb_reward, 2, 0)
            self.buffers['preference'][idxs] = torch.cat((mb_preference, mb_new_preferences))
            self.buffers['next_state'][idxs] = torch.repeat_interleave(mb_next_state, 2, 0)
            self.n_transitions_stored += batch_size * 2

    # sample the data from the replay buffer
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        idxs = torch.randint(0, self.current_size, (batch_size,))
        # sample transitions
        transitions = {key: temp_buffers[key][idxs] for key in temp_buffers.keys()}
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size + inc <= self.size:
            idx = torch.arange(self.current_size, self.current_size + inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = torch.arange(self.current_size, self.size)
            idx_b = torch.randint(0, self.current_size, overflow)
            idx = torch.cat([idx_a, idx_b])
        else:
            idx = torch.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idx[0]
        return idx
