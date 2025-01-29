import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentDQNetwork(nn.Module):
    """
    CNN -> Flatten -> LSTM -> Final FC layer -> Q-values
    """

    def __init__(
        self,
        action_dim=15,
        in_channels=4,    # stacked frames or single frame + LSTM
        kernel_size=5,
        stride=2,
        hidden_size=256,
        num_lstm_layers=1
    ):
        super(RecurrentDQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=kernel_size, stride=stride)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=kernel_size, stride=stride)

        # Flatten to feed into LSTM
        self.cnn_out_size = 16 * 47 * 61

        # Fully-connected layer to reduce dimension before LSTM
        self.fc1 = nn.Linear(self.cnn_out_size, 256)

        # LSTM: input_size=256, hidden_size=hidden_size
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True
        )

        # Final layer from hidden state to Q-values
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, x, hidden=None):
        """
        x shape: [batch_size, seq_len, in_channels, H, W]
        hidden: (h_0, c_0) for LSTM (optional)
        Returns (q_values, (h_n, c_n)):

         - q_values shape: [batch_size, seq_len, action_dim]
         - (h_n, c_n) are the final hidden states
        """

        bsz, seq_len, C, H, W = x.shape

        # 1) Flatten the sequence dimension into batch for CNN
        x = x.view(bsz * seq_len, C, H, W)

        # 2) CNN feature extraction
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # shape: [bsz*seq_len, cnn_out_size]

        x = F.relu(self.fc1(x))   # => [bsz*seq_len, 256]

        # 3) Reshape for LSTM: [bsz, seq_len, 256]
        x = x.view(bsz, seq_len, 256)

        # 4) LSTM over the time dimension
        x, hidden_out = self.lstm(x, hidden)
        # x => [batch_size, seq_len, hidden_size]

        # 5) Map hidden_size -> action_dim
        # We want Q-values at each time step => pass x through fc2
        q_values = self.fc2(x)  # shape [bsz, seq_len, action_dim]

        return q_values, hidden_out

