"""Implementation of Deep Q-Learning Network for Reinforcement Learning"""
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.base import BaseAgent


class ReplayBuffer:
    def __init__(self, capacity=30_000):
        """
        capacity: Max number of transitions to store. Old ones are removed first.
        """
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Add a single transition to the buffer.
        """
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        Returns separate lists (or arrays) for states, actions, rewards, next_states, dones.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.memory)


class DQNetwork(nn.Module):
    def __init__(self, action_dim=15, in_channels=4, kernel_size=5, stride=2):
        super(DQNetwork, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, stride=stride)

        self.fc1 = nn.Linear(16 * 47 * 61, 256)
        self.fc2 = nn.Linear(256, action_dim)

    def prepare_input(self, x: np.ndarray):
        """Convert retro gym rgb output to pytorch format"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        return x

    def forward(self, x: np.ndarray):
        x = self.prepare_input(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values


class DQNAgent(BaseAgent):
    """
    DQN Agent Handles the RL logic;
    * Initialization: Hyperparameters, networks, optimizers, fitting, etc...
    * Action Selection: epsilon-greedy policy, callable policy.
    * Storing Experiences: Replaying buffer or memory.
    * Training Loop: Need to sample from memory and update network.
    * Load/Score: In order to persist model weights.


    """

    def __init__(
        self,
        state_shape,
        action_dim,
        gamma=0.99,
        lr=0.001,
        batch_size=16,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=1e-6,
        target_update_freq=24_000,
        replay_capacity=35_000,
        device=None,
    ):
        """
        state_shape:     Shape of the environment state (channels, height, width).
        action_dim:      Number of discrete actions.
        gamma:           Discount factor.
        lr:              Learning rate.
        batch_size:      Mini-batch size for training.
        epsilon_start:   Initial epsilon for exploration.
        epsilon_end:     Minimum epsilon.
        epsilon_decay:   How fast to decay epsilon each step.
        target_update_freq: How many steps between target network updates.
        replay_capacity: Max size of the replay buffer.
        device:          'cpu' or 'cuda'. If None, auto-detect.
        """

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size

        # Epsilon parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize networks
        self.policy_network = DQNetwork(
            action_dim=action_dim, in_channels=state_shape[0]
        ).to(self.device)
        self.target_network = DQNetwork(
            action_dim=action_dim, in_channels=state_shape[0]
        ).to(self.device)

        # Copy weights to target network initially
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # target net in eval mode by default

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=replay_capacity)

        # Counters
        self.learn_step_counter = 0  # Counts how many times we've trained
        self.target_update_freq = (
            target_update_freq  # Update target net after this many training steps
        )

    def act(self, state):
        """
        Epsilon-greedy action selection.
        state: A single state, typically a NumPy array (H, W, C) or (C, H, W).
        Returns: int (the chosen action index).
        """
        # Convert state to Tensor. We'll assume state is (C, H, W) or we permute outside if needed.
        # If your state is channels-last, do state = state.transpose(2, 0, 1) first.
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        # shape: [1, C, H, W]

        # Epsilon-greedy
        if np.random.rand() < self.epsilon:
            # Explore
            action_index = np.random.randint(self.action_dim)
        else:
            # Exploit
            with torch.no_grad():
                q_values = self.policy_network(state_t)  # shape: [1, action_dim]
                action_index = q_values.argmax(dim=1).item()

        return action_index

    def remember(self, state, action, reward, next_state, done):
        """
        Store the transition in replay buffer.
        state, next_state can be NumPy arrays or Tensors (though typically NumPy).
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update_target_network(self):
        """
        Copy weights from policy_network to target_network.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def decay_epsilon(self):
        """
        Decays epsilon toward epsilon_end after each step or episode.
        """
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end

    def replay(self):
        """
        Sample a batch from replay buffer and update the policy network.
        Also handles updating the target network at intervals.
        """
        # Check if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough data to train

        # Sample from replay
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to Tensors
        # Each item in states, etc. is a single state or next_state
        # We'll stack them along dim=0 to form a batch
        state_batch = []
        next_state_batch = []
        for s, ns in zip(states, next_states):
            # If your states are channel-last, you might need s = s.transpose(2,0,1)
            state_batch.append(torch.from_numpy(s).float().unsqueeze(0))
            next_state_batch.append(torch.from_numpy(ns).float().unsqueeze(0))

        state_batch = torch.cat(state_batch).to(
            self.device
        )  # shape: [batch_size, C, H, W]
        next_state_batch = torch.cat(next_state_batch).to(
            self.device
        )  # shape: [batch_size, C, H, W]

        action_batch = torch.tensor(
            actions, dtype=torch.long, device=self.device
        )  # [batch_size]
        reward_batch = torch.tensor(
            rewards, dtype=torch.float32, device=self.device
        )  # [batch_size]
        done_batch = torch.tensor(
            dones, dtype=torch.float32, device=self.device
        )  # [batch_size]

        # ---- Q(s, a) from the policy network
        # Q-values for all actions [batch_size, action_dim]
        policy_q_vals = self.policy_network(state_batch)
        # We only want Q for the chosen action in the batch
        # Gather along dim=1 using the action index
        q_s_a = policy_q_vals.gather(1, action_batch.unsqueeze(1)).squeeze(
            1
        )  # shape: [batch_size]

        # ---- Compute target Q
        # Q(s', a') from target network
        with torch.no_grad():
            next_q_vals = self.target_network(
                next_state_batch
            )  # [batch_size, action_dim]
            # We want the max across actions
            max_next_q_vals = next_q_vals.max(dim=1)[0]  # shape: [batch_size]

        # If done, there's no future reward. So we multiply max_next_q_vals by (1 - done)
        # done_batch is 1.0 if done, 0.0 otherwise
        target = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q_vals

        # ---- Loss: MSE or Huber
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_s_a, target)

        # ---- Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1

        # Update target network periodically
        if self.learn_step_counter % self.target_update_freq == 0:
            print("updating target network ...")
            self.update_target_network()

    def save(self, filename="dqn_checkpoint.pth"):
        """
        Save model and optimizer states.
        """
        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(checkpoint, filename)

    def load(self, filename="dqn_checkpoint.pth"):
        """
        Load model and optimizer states.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["learn_step_counter"]
