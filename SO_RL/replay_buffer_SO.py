import threading
import torch

"""
The replay buffer here is basically from the openai baselines code
"""


class ReplayBuffer_SO:
    def __init__(self, env_params, buffer_size, device):
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
            'next_states': torch.empty((self.size, *self.env_params['states'][0]), dtype=torch.float32,
                                       device=self.device),
            'dones': torch.empty((self.size, 1), dtype=torch.bool, device=self.device),
        }
        # thread lock
        self.lock = threading.Lock()
        self.position = 0
        self.k = 0

    # Store the episode
    def store_episode(self, episode_batch):
        mb_state, mb_action, mb_reward, mb_next_state = episode_batch
        batch_size = mb_state.shape[0]
        # Adds a new randomly sampled normally distributed preference vector for each transition
        with self.lock:
            # self.k is set to 0 => only 1 repeat stored
            idxs = self._get_storage_idx(inc=(self.k + 1) * batch_size)
            # store the informations
            self.buffers['states'][idxs] = torch.repeat_interleave(mb_state, self.k + 1, 0)
            self.buffers['actions'][idxs] = torch.repeat_interleave(mb_action, self.k + 1, 0).reshape(
                (self.k + 1) * batch_size, 1)
            rewards = torch.repeat_interleave(mb_reward.to(torch.float32), self.k + 1, 0)
            self.buffers['rewards'][idxs] = rewards.unsqueeze(dim=1)
            self.buffers['next_states'][idxs] = torch.repeat_interleave(mb_next_state, self.k + 1, 0)
            self.n_transitions_stored += batch_size * (self.k + 1)

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
        transitions['rewards'] = transitions['rewards'] # / abs(self.norm)
        return transitions, idxs

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        idxs = torch.arange(self.position, self.position + inc) % self.size
        self.position += inc
        self.current_size = min(self.size, self.current_size + inc)
        if inc == 1:
            idx = idxs[0]
        return idxs
