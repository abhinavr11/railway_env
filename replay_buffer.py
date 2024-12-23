import random
from collections import deque, namedtuple
import torch

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, device='cpu'):
        """
        Initializes the ReplayBuffer.

        Args:
            buffer_size (int): Maximum number of transitions to store.
            batch_size (int): Number of transitions to sample per batch.
            device (str): Device to store tensors ('cpu' or 'cuda').
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device

        self.memory = deque(maxlen=buffer_size)
        self.transition = namedtuple('Transition',
                                     field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.

        Args:
            state (torch.Tensor): Current state.
            action (Any): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state after action.
            done (bool): Whether the episode ended.
        """
   
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        self.memory.append(self.transition(state, action, reward, next_state, done))

    def sample(self):
        """
        Samples a batch of transitions from the buffer.

        Returns:
            Tuple of tensors: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.memory, self.batch_size)
    
        batch = self.transition(*zip(*batch))


        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)

        actions = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float, device=self.device).unsqueeze(1)
        dones = torch.tensor(batch.done, dtype=torch.float, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of internal memory.
        """
        return len(self.memory)

    def clear(self):
        """
        Clears all transitions from the buffer.
        """
        self.memory.clear()

    def save(self, filepath):
        """
        Saves the buffer to a file.

        Args:
            filepath (str): Path to save the buffer.
        """
        torch.save(self.memory, filepath)

    def load(self, filepath):
        """
        Loads the buffer from a file.

        Args:
            filepath (str): Path to load the buffer from.
        """
        self.memory = torch.load(filepath, map_location=self.device)
