import numpy as np
import torch
import random
from collections import namedtuple, deque

class Replay_Buffer(object):
    """Replay buffer to store past experiences that the agent can then use for training data"""
    
    def __init__(self, buffer_size, batch_size, seed, device=None):
        """
        Args:
            buffer_size: Integer indicating the max number of experiences to store
            batch_size: Integer indicating the size of batches to sample
            seed: Integer to use as random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def add_experience(self, states, actions, rewards, next_states, dones):
        """Adds experience(s) into the replay buffer"""
        if isinstance(dones, list):
            assert not isinstance(dones[0], list), "A done shouldn't be a list"
            experiences = [self.experience(state, action, reward, next_state, done)
                         for state, action, reward, next_state, done in
                         zip(states, actions, rewards, next_states, dones)]
            self.memory.extend(experiences)
        else:
            experience = self.experience(states, actions, rewards, next_states, dones)
            self.memory.append(experience)
   
    def sample(self, num_experiences=None):
        """Draws a random sample of experience from the replay buffer"""
        experiences = self.pick_experiences(num_experiences)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def pick_experiences(self, num_experiences=None):
        if num_experiences is not None:
            batch_size = num_experiences
        else:
            batch_size = self.batch_size
        return random.sample(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)
