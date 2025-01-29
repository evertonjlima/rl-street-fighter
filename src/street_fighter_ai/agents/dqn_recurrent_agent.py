import torch
import torch.optim as optim
import numpy as np

from agents.base import BaseAgent
from agents.recurrent import RecurrentDQNetwork
from agents.memory_buffer import SequenceReplayBuffer

class RecurrentDQNAgent(BaseAgent):
    def __init__(
        self,
        state_shape,
        action_dim,
        gamma=0.99,
        lr=0.001,
        batch_size=16,
        seq_len=8,        # how many consecutive steps per batch
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=1e-6,
        temperature=1.0,  # temperature for sampling actions
        target_update_freq=24000,
        replay_capacity=35000,
        device=None,
    ):
        """
        - state_shape: (C, H, W)
        - action_dim:  discrete action size
        - seq_len:     how many consecutive frames to sample for the LSTM
        - temperature: scales Q-values before sampling an action
        """

        self.state_shape = state_shape
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Epsilon for exploration (optional if you rely on temperature)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

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

    def act(self, state):
        """
        Choose an action using a temperature-scaled softmax of Q-values, 
        or fallback to epsilon-greedy. Typically, you'd do *one step at a time*:
        - state shape: [C, H, W] (single frame or stacked frames)
        - hidden: optional (h, c) from previous LSTM step if you're doing truly 
          recurrent inference frame-by-frame.

        Returns: action_index, new_hidden
        """

        # Flatten to shape: [1, 1, C, H, W] so policy_network forward can handle seq_len=1
        state_t = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0).to(self.device)

        # Epsilon-greedy remains if you want random exploration
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(self.action_dim)
            self.hidden = hidden
        else:
            with torch.no_grad():
                q_values, self.hidden = self.policy_network(state_t, self.hidden)
                action_index = q_values.argmax(dim=1).item()

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

    def replay(self):
        """
        Sample sequences from replay buffer, feed them through the RecurrentDQNetwork,
        and update Q-values.
        """
        # Need enough episodes in buffer to do a sample
        if len(self.replay_buffer) < 1:
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
            max_next_q_values, _ = next_q_values_seq.max(dim=2)  # shape [batch_size, seq_len]

        # 3) Bellman update
        #    We'll treat each time-step in the sequence as its own Q-learning target
        #    If done=1 => no future reward
        targets = rewards_t + (1.0 - dones_t) * self.gamma * max_next_q_values

        # 4) Loss
        loss_fn = nn.MSELoss()
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
        self.replay_buffer.start_episode()

    def save(self, filename="dqn_checkpoint.pth"):
        checkpoint = {
            "policy_network": self.policy_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "temperature": self.temperature,
            "learn_step_counter": self.learn_step_counter,
        }
        torch.save(checkpoint, filename)

    def load(self, filename="dqn_checkpoint.pth"):
        checkpoint = torch.load(filename, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.temperature = checkpoint.get("temperature", 1.0)
        self.learn_step_counter = checkpoint["learn_step_counter"]

