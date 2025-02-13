import torch
import torch.nn as nn
import torch.optim as optim
from agents.base import BaseAgent
from agents.memory_buffer import SequenceReplayBuffer
from agents.recurrent import RecurrentDQNetwork
from rich.console import Console

console = Console()


class RecurrentDQNAgent(BaseAgent):
    def __init__(
        self,
        state_shape,
        action_dim,
        gamma=0.99,
        lr=0.001,
        tau=0.001,
        batch_size=16,
        seq_len=32,  # how many consecutive steps per batch
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=1e-6,
        target_update_freq=10000,
        replay_capacity=35000,
        max_grad_norm=1,
        device=None,
    ):
        """
        - state_shape: (C, H, W)
        - action_dim:  discrete action size
        - seq_len:     how many consecutive frames to sample for the LSTM
        """

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.tau = tau
        self.max_grad_norm = max_grad_norm

        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if device is None:
            self.device = torch.device("mps" if torch.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Networks
        self.policy_network = RecurrentDQNetwork(
            action_dim=action_dim, in_channels=state_shape[0]
        ).to(self.device)

        self.target_network = RecurrentDQNetwork(
            action_dim=action_dim, in_channels=state_shape[0]
        ).to(self.device)

        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        self.hidden = None

        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)

        # Replay buffer
        self.replay_buffer = SequenceReplayBuffer(capacity=replay_capacity)

        self.learn_step_counter = 0
        self.target_update_freq = target_update_freq

    def act(
        self,
        state,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    ):
        """
        Convert Q-values to a 'logits' distribution, apply temperature,
        top-k and top-p filtering, then sample an action from that
        distribution. If we're below epsilon, pick a random action
        (epsilon-greedy fallback).

        state:     [C, H, W] numpy array (or however your environment provides it)
        top_k:     int, restricts sampling to the top_k logit values
        top_p:     float, restricts sampling to the smallest set of logits whose
                   cumulative probability <= top_p
        temperature: float, scales the logits; <1 => more greedy, >1 => more uniform

        Returns:
           action_index: int
        """
        import numpy as np
        import torch

        # Convert 'state' to shape: [1, 1, C, H, W] for your recurrent network
        state_t = (
            torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Epsilon-greedy remains if you want random exploration
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                q_values, self.hidden = self.policy_network(state_t, self.hidden)
                # q_values shape: [batch=1, seq_len=1, action_dim]
                # Squeeze to [action_dim]
                logits = q_values.squeeze(0).squeeze(0)  # => shape [action_dim]

            # ---- 1) Temperature ----
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)

            # ---- 2) Convert logits to probabilities via softmax ----
            probs = torch.softmax(logits, dim=-1)

            # ---- 3) Apply top-k / top-p filtering ----
            # If top_k > 0 => keep only top k
            if top_k > 0:
                values_to_keep, indices_to_keep = torch.topk(probs, k=top_k)
                # Create a mask for all positions that are not in the top_k
                mask = torch.ones_like(probs, dtype=torch.bool)
                mask[indices_to_keep] = False  # unmask the top_k
                # Zero out everything not in top_k
                probs[mask] = 0.0

            # If top_p < 1.0 => keep smallest set of actions with cumulative prob <= top_p
            if top_p < 1.0:
                # Sort probabilities descending
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                # Find cutoff where cumulative probs exceed top_p
                cutoff_index = torch.searchsorted(cumulative_probs, top_p, right=True)
                # Zero out anything above that cutoff
                if cutoff_index < len(sorted_probs):
                    sorted_probs[cutoff_index + 1 :] = 0.0
                # Now unsort them back to original positions
                probs.fill_(0.0)
                probs[sorted_indices] = sorted_probs

            # Re-normalize if we've zeroed out positions
            probs_sum = probs.sum()
            if probs_sum > 0:
                probs = probs / probs_sum
            else:
                # If everything got zeroed (extreme top_k/p), fallback to uniform
                probs = torch.ones_like(probs) / len(probs)

            # ---- 4) Sample from the distribution ----
            action_index = torch.multinomial(probs, 1).item()

        return action_index

    def remember(self, state, action, reward, next_state, done):
        """
        Store transitions in the sequence-based replay buffer.
        Typically, you start an episode with replay_buffer.start_episode()
        Then call remember(...) for each step.
        """
        self.replay_buffer.store_transition(state, action, reward, next_state, done)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_end:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_end

    def replay(self, burn_in=4):
        """
        Sample sequences from the replay buffer with a total length of (seq_len + burn_in).
        The first 'burn_in' steps are used to compute the LSTM's hidden state (burn-in phase),
        and the loss is computed on the subsequent 'seq_len' steps.

        Args:
            burn_in (int): Number of initial timesteps used for burn-in.
        """
        total_seq_len = self.seq_len + burn_in  # total number of steps to sample

        # Ensure the replay buffer has at least one complete episode.
        if len(self.replay_buffer.episodes) < 1:
            return

        if total_seq_len <= burn_in:
            raise ValueError("total_seq_len must be greater than burn_in")

        # Sample a batch of sequences from the replay buffer.
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size=self.batch_size, seq_len=total_seq_len
        )

        # Convert the sampled arrays to Tensors on the correct device.
        states_t = torch.from_numpy(states).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        # --------------------------------------------------
        # Compute initial hidden state using burn-in steps.
        # --------------------------------------------------
        # Use the first 'burn_in' timesteps to compute the hidden state.
        _, hidden_policy = self.policy_network(states_t[:, :burn_in], None)
        _, hidden_target = self.target_network(next_states_t[:, :burn_in], None)

        # --------------------------------------------------
        # Compute Q-values for the remaining steps after burn-in.
        # --------------------------------------------------
        # The remaining steps (indices burn_in: total_seq_len) have length = self.seq_len.
        q_values_seq, _ = self.policy_network(states_t[:, burn_in:], hidden_policy)
        next_q_values_seq, _ = self.target_network(
            next_states_t[:, burn_in:], hidden_target
        )
        # q_values_seq shape: [batch_size, self.seq_len, action_dim]

        # Adjust actions, rewards, and dones to match the training portion.
        actions_t = actions_t[:, burn_in:]
        rewards_t = rewards_t[:, burn_in:]
        dones_t = dones_t[:, burn_in:]

        # --------------------------------------------------
        # Compute Q-values for the selected actions.
        # --------------------------------------------------
        q_s_a = q_values_seq.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        # q_s_a shape: [batch_size, self.seq_len]

        # --------------------------------------------------
        # Compute target Q-values from the target network.
        # --------------------------------------------------
        with torch.no_grad():
            max_next_q_values, _ = next_q_values_seq.max(dim=2)
        targets = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q_values

        # --------------------------------------------------
        # Compute loss, backpropagate, and update network weights.
        # --------------------------------------------------
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_s_a, targets)
        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        self.learn_step_counter += 1

        # hard updates
        # if self.learn_step_counter % self.target_update_freq == 0:
        #    self.target_network.load_state_dict(self.policy_network.state_dict())
        self._soft_update()

    def _soft_update(self):
        for target_param, policy_param in zip(
            self.target_network.parameters(), self.policy_network.parameters()
        ):
            # Blend the weights
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def _replay(self):
        """
        Sample sequences from replay buffer, feed them through the RecurrentDQNetwork,
        and update Q-values.
        """
        # Need enough episodes in buffer to do a sample
        if len(self.replay_buffer.episodes) < 1:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size=self.batch_size, seq_len=self.seq_len
        )
        # shapes:
        # states: [batch_size, seq_len, C, H, W]
        # actions: [batch_size, seq_len]
        # rewards: [batch_size, seq_len]
        # next_states: [batch_size, seq_len, C, H, W]
        # dones: [batch_size, seq_len]

        # Convert to Tensors
        states_t = torch.from_numpy(states).float().to(self.device)
        next_states_t = torch.from_numpy(next_states).float().to(self.device)
        actions_t = torch.from_numpy(actions).long().to(self.device)
        rewards_t = torch.from_numpy(rewards).float().to(self.device)
        dones_t = torch.from_numpy(dones).float().to(self.device)

        # 1) Get Q-values from policy_network for entire sequence
        #    Output shape: [batch_size, seq_len, action_dim]
        q_values_seq, _ = self.policy_network(states_t)  # no initial hidden => zeros
        # Gather the Q-value for each (batch, seq_len, action)
        # actions_t has shape [batch_size, seq_len]
        # We want q_values_seq[batch, seq, actions_t[batch,seq]]
        q_s_a = q_values_seq.gather(2, actions_t.unsqueeze(-1)).squeeze(-1)
        # => shape: [batch_size, seq_len]

        # 2) Target Q from target_network on next_states
        with torch.no_grad():
            next_q_values_seq, _ = self.target_network(next_states_t)
            # next_q_values_seq: [batch_size, seq_len, action_dim]
            max_next_q_values, _ = next_q_values_seq.max(
                dim=2
            )  # shape [batch_size, seq_len]

        # 3) Bellman update
        #    We'll treat each time-step in the sequence as its own Q-learning target
        #    If done=1 => no future reward
        targets = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q_values

        # 4) Loss
        loss_fn = nn.SmoothL1Loss()
        loss = loss_fn(q_s_a, targets)

        # 5) Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_step_counter += 1
        # Periodically update target net
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def reset(self):
        self.hidden = None
        self.replay_buffer.start_episode()

    def _get_state(self):
        return {"epsilon": self.epsilon, "learn_step_counter": self.learn_step_counter}

    def save(self, filename="dqn_checkpoint.pth"):
        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(checkpoint, filename)

    def load(self, filename="dqn_checkpoint.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.learn_step_counter = checkpoint["learn_step_counter"]
