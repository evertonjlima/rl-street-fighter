import random
from collections import deque

import numpy as np


class SequenceReplayBuffer:
    """
    Stores entire episodes as lists of transitions. We keep track of
    the total transitions so we don't exceed 'max_transitions'.
    During sampling, we pick an episode at random, then pick a random
    subrange of length 'seq_len'.

    This version also ensures states are float32 of a consistent shape
    (e.g. (C,H,W)) to avoid 'object' dtype issues.
    """

    def __init__(self, state_shape=(4, 96, 96), capacity=30000):
        """
        Args:
            state_shape (tuple): e.g. (C, H, W)
            max_transitions (int): total transitions capacity
        """
        self.state_shape = state_shape  # (C,H,W)
        self.max_transitions = capacity

        self.episodes = deque()
        self.num_transitions = 0

        # Temporary list for the current (unfinished) episode
        self.current_episode = []

    def start_episode(self):
        """
        Called at the beginning of a new episode. If there's
        an unfinished current_episode (not done), you can store it
        or discard it. For simplicity, we'll discard partial episodes
        if the environment ended abruptly. Adjust as needed.
        """
        if len(self.current_episode) > 0:
            self._store_episode(self.current_episode)

        self.current_episode = []

    def store_transition(self, state, action, reward, next_state, done):
        """
        Convert 'state' and 'next_state' to float32 arrays of shape self.state_shape.
        Then append to the current episode. If 'done', store the full episode.
        """
        if np.array(state).shape != self.state_shape:
            print("Storing state of shape:", np.array(state).shape)
            return 0

        # Ensure consistent shape & dtype
        state_arr = np.array(state, dtype=np.float32).reshape(self.state_shape)
        next_state_arr = np.array(next_state, dtype=np.float32).reshape(
            self.state_shape
        )

        self.current_episode.append((state_arr, action, reward, next_state_arr, done))

        if done:
            self._store_episode(self.current_episode)
            self.current_episode = []

    def _store_episode(self, episode):
        """
        Add an entire episode to the buffer.
        If we exceed max_transitions, pop older episodes until we're under limit.
        """
        ep_len = len(episode)
        self.episodes.append(episode)
        self.num_transitions += ep_len

        # Evict oldest episodes if over capacity
        while self.num_transitions > self.max_transitions and len(self.episodes) > 0:
            oldest_ep = self.episodes.popleft()
            self.num_transitions -= len(oldest_ep)

    def sample(self, batch_size, seq_len):
        """
        Returns a batch of shape:
           states:       (batch_size, seq_len, C, H, W)
           actions:      (batch_size, seq_len)
           rewards:      (batch_size, seq_len)
           next_states:  (batch_size, seq_len, C, H, W)
           dones:        (batch_size, seq_len)

        For each sampled episode, if the available sequence is shorter than seq_len,
        pad the sequence with zeros along the time dimension.
        """
        if len(self.episodes) == 0:
            print("no episodes! returning empty ...")
            return (
                np.zeros((0, seq_len) + self.state_shape, dtype=np.float32),
                np.zeros((0, seq_len), dtype=np.int64),
                np.zeros((0, seq_len), dtype=np.float32),
                np.zeros((0, seq_len) + self.state_shape, dtype=np.float32),
                np.zeros((0, seq_len), dtype=np.float32),
            )

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # Randomly pick episodes (with replacement if needed)
        chosen_episodes = random.choices(self.episodes, k=batch_size)

        for ep in chosen_episodes:
            ep_len = len(ep)
            if ep_len < seq_len:
                # Use the entire episode and pad the rest with zeros.
                start_idx = 0
                end_idx = ep_len
                pad_length = seq_len - ep_len
            else:
                start_idx = random.randint(0, ep_len - seq_len)
                end_idx = start_idx + seq_len
                pad_length = 0

            seq = ep[start_idx:end_idx]
            # Unpack transitions
            states_list, actions_list, rewards_list, next_states_list, dones_list = zip(
                *seq
            )

            # Convert states and next_states to arrays (assumed shape: self.state_shape)
            states_np = np.stack(
                states_list, axis=0
            )  # shape: [current_seq_len, C, H, W]
            next_states_np = np.stack(next_states_list, axis=0)

            # Pad if necessary along the time dimension (first dimension)
            if pad_length > 0:
                # Create zero padding with the same shape as a single time step
                state_pad = np.zeros((pad_length,) + self.state_shape, dtype=np.float32)
                next_state_pad = np.zeros(
                    (pad_length,) + self.state_shape, dtype=np.float32
                )
                # For actions, rewards, dones, pad with zeros (or a default value)
                action_pad = np.zeros((pad_length,), dtype=np.int64)
                reward_pad = np.zeros((pad_length,), dtype=np.float32)
                done_pad = np.zeros((pad_length,), dtype=np.float32)

                states_np = np.concatenate([states_np, state_pad], axis=0)
                actions_np = np.concatenate(
                    [np.array(actions_list, dtype=np.int64), action_pad], axis=0
                )
                rewards_np = np.concatenate(
                    [np.array(rewards_list, dtype=np.float32), reward_pad], axis=0
                )
                next_states_np = np.concatenate(
                    [next_states_np, next_state_pad], axis=0
                )
                dones_np = np.concatenate(
                    [np.array(dones_list, dtype=np.float32), done_pad], axis=0
                )
            else:
                actions_np = np.array(actions_list, dtype=np.int64)
                rewards_np = np.array(rewards_list, dtype=np.float32)
                dones_np = np.array(dones_list, dtype=np.float32)

            batch_states.append(states_np)
            batch_actions.append(actions_np)
            batch_rewards.append(rewards_np)
            batch_next_states.append(next_states_np)
            batch_dones.append(dones_np)

        # Stack along the batch dimension -> (batch_size, seq_len, ...)
        batch_states = np.stack(batch_states, axis=0)
        batch_actions = np.stack(batch_actions, axis=0)
        batch_rewards = np.stack(batch_rewards, axis=0)
        batch_next_states = np.stack(batch_next_states, axis=0)
        batch_dones = np.stack(batch_dones, axis=0)

        return (
            batch_states,  # shape: [batch_size, seq_len, C, H, W]
            batch_actions,  # shape: [batch_size, seq_len]
            batch_rewards,  # shape: [batch_size, seq_len]
            batch_next_states,  # shape: [batch_size, seq_len, C, H, W]
            batch_dones,  # shape: [batch_size, seq_len]
        )

    def _sample(self, batch_size, seq_len):
        """
        Returns a batch of shape:
           states:       (batch_size, seq_len, C, H, W)
           actions:      (batch_size, seq_len)
           rewards:      (batch_size, seq_len)
           next_states:  (batch_size, seq_len, C, H, W)
           dones:        (batch_size, seq_len)

        We randomly pick 'batch_size' episodes, then pick a subsequence of length 'seq_len' from each.
        If an episode is shorter than seq_len, we use its entire length or skip it (your choice).
        """
        if len(self.episodes) == 0:
            # Return empty arrays if no data
            print("no episodes! returning empty ...")
            return (
                np.zeros((0, seq_len) + self.state_shape, dtype=np.float32),
                np.zeros((0, seq_len), dtype=np.int64),
                np.zeros((0, seq_len), dtype=np.float32),
                np.zeros((0, seq_len) + self.state_shape, dtype=np.float32),
                np.zeros((0, seq_len), dtype=np.float32),
            )

        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        # Randomly pick episodes (with replacement if needed)
        chosen_episodes = random.choices(self.episodes, k=batch_size)

        for ep in chosen_episodes:
            ep_len = len(ep)
            if ep_len < seq_len:
                # If too short, either skip or just take entire ep
                start_idx = 0
                end_idx = ep_len
            else:
                start_idx = random.randint(0, ep_len - seq_len)
                end_idx = start_idx + seq_len

            seq = ep[start_idx:end_idx]

            # Unpack each transition
            states_list, actions_list, rewards_list, next_states_list, dones_list = zip(
                *seq
            )

            # Now each is a tuple of arrays
            # Convert them to stacked np.float32 arrays
            states_np = np.stack(states_list, axis=0)  # shape: [seq_len, C,H,W]
            actions_np = np.array(actions_list, dtype=np.int64)  # shape: [seq_len]
            rewards_np = np.array(rewards_list, dtype=np.float32)
            next_states_np = np.stack(
                next_states_list, axis=0
            )  # shape: [seq_len, C,H,W]
            dones_np = np.array(dones_list, dtype=np.float32)

            batch_states.append(states_np)
            batch_actions.append(actions_np)
            batch_rewards.append(rewards_np)
            batch_next_states.append(next_states_np)
            batch_dones.append(dones_np)

        # Now stack along the batch dimension => (batch_size, seq_len, C,H,W)
        batch_states = np.stack(batch_states, axis=0)  # float32
        batch_actions = np.stack(batch_actions, axis=0)  # int64
        batch_rewards = np.stack(batch_rewards, axis=0)  # float32
        batch_next_states = np.stack(batch_next_states, axis=0)
        batch_dones = np.stack(batch_dones, axis=0)

        return (
            batch_states,  # (batch_size, seq_len, C, H, W)
            batch_actions,  # (batch_size, seq_len)
            batch_rewards,  # (batch_size, seq_len)
            batch_next_states,  # (batch_size, seq_len, C, H, W)
            batch_dones,  # (batch_size, seq_len)
        )

    def __len__(self):
        """
        Number of episodes stored (not transitions).
        """
        return len(self.episodes)
