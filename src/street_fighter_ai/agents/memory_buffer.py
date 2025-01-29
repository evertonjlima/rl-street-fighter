from collections import deque
import random
import numpy as np

class SequenceReplayBuffer:
    """
    Stores full episodes as lists of transitions. During sampling,
    we pick an episode at random, then pick a random subsequence
    of length 'seq_len'.
    """
    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.current_episode = []

    def start_episode(self):
        """
        Call at the beginning of a new episode.
        Resets the internal list of transitions.
        """
        self.current_episode = []

    def store_transition(self, state, action, reward, next_state, done):
        """
        Append a single transition for the current episode.
        """
        self.current_episode.append((state, action, reward, next_state, done))
        # If done, push entire episode into the buffer
        if done:
            self.buffer.append(self.current_episode)
            self.current_episode = []

    def sample(self, batch_size, seq_len):
        """
        Returns batches of shape [batch_size, seq_len, ...] for states, actions, etc.
        We randomly pick episodes, then randomly pick subsequences within them.
        """
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # Randomly select 'batch_size' episodes
        episodes = random.choices(self.buffer, k=batch_size)

        for ep in episodes:
            # Make sure ep has enough transitions
            if len(ep) < seq_len:
                # If not, just skip or pad (here we skip for simplicity)
                start_idx = 0
                end_idx = len(ep)
            else:
                start_idx = random.randint(0, len(ep) - seq_len)
                end_idx = start_idx + seq_len

            seq = ep[start_idx:end_idx]

            # Unpack sequences
            s, a, r, ns, d = zip(*seq)
            batch_states.append(s)
            batch_actions.append(a)
            batch_rewards.append(r)
            batch_next_states.append(ns)
            batch_dones.append(d)

        # Convert to numpy arrays (batch_size, seq_len, ...)
        # States might be e.g. (C, H, W) each, so shape is (batch_size, seq_len, C, H, W)
        return (
            np.array(batch_states),
            np.array(batch_actions),
            np.array(batch_rewards, dtype=np.float32),
            np.array(batch_next_states),
            np.array(batch_dones, dtype=np.float32),
        )

    def __len__(self):
        """
        Number of episodes currently in the buffer
        (not the number of transitions).
        """
        return len(self.buffer)

