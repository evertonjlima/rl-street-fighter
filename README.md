# Street Fighter II Reinforcement Learning Project

This project uses **OpenAI’s Retro Gym** to train a reinforcement learning (RL) agent to play **Street Fighter II**. The agent observes game frames, processes them via a Deep Q-Network (DQN), and outputs actions (button presses) in an attempt to defeat various in-game opponents.

---

## Overview

- **Goal**: Have an RL agent learn **basic or advanced fighting game strategies** (punches, kicks, special moves, etc.) to reduce the opponent’s health bar and eventually win each match.
- **Method**: Deep Q-Learning (DQN) with:
  - A **Convolutional Neural Network** (CNN) architecture to process stacked frames as input.
  - A **replay buffer** to store and sample past experiences for efficient learning.
  - Custom **reward shaping** to encourage landing hits, discourage taking damage, and speed up meaningful exploration.

The project includes custom code for:
1. **Action Space Mapping** (how game inputs are converted to button presses).
2. **Reward Function** (incorporating health lost, health dealt, time penalties, etc.).
3. **DQN Implementation** (neural network, replay buffer, and training logic).
4. **Exploration Enhancements** (e.g., using macros or special moves to avoid purely random exploration).

---

## About OpenAI Retro Gym

- **Retro Gym**: A library for **reinforcement learning research** on classic video games by enabling easy integration of console emulators.
- **Official Repository**: [github.com/openai/retro](https://github.com/openai/retro)
- **Documentation**: [gym-retro on PyPI](https://pypi.org/project/gym-retro/)

Retro Gym wraps old console/arcade titles in a **Gym-like** interface, allowing you to:
- **Reset** the environment to a starting state.
- **Step** forward by providing actions (button presses).
- **Observe** the screen frames and other info like score or health.
- **Get rewards/done signals** if you define a custom or built-in reward function.

---

## Current Approach

1. **Deep Q-Network (DQN)**
   - **Neural Network** with convolutional layers to handle visual input (stacked frames) and fully connected layers to output Q-values for each possible action (or set of actions).
   - **Target Network** updated periodically to stabilize training.
   - **Replay Buffer** (Experience Replay) that samples mini-batches of past transitions to break correlation and improve data efficiency.

2. **Action Space**
   - We define a custom **input mapping** that translates discrete indices (or multi-binary arrays) into Street Fighter II button presses (e.g., *light punch, heavy kick, forward, crouch, special macros,* etc.).

3. **Reward Shaping**
   - Rewards for **damaging the opponent** (positive).
   - Penalties for **taking damage** (negative).
   - Optional **score-based** component.
   - A **time penalty** to avoid stalling or overly cautious play.
   - Potential **bonus** for winning the round.

4. **Exploration**
   - Early training often uses an **epsilon-greedy** policy, where the agent chooses random actions (or macros) to ensure coverage of the action space.
   - We can enhance exploration by **injecting special moves** or using “macro actions” (like **quarter-circle forward + punch**).

5. **Training**
   - The agent interacts with the environment frame by frame, storing transitions in the replay buffer.
   - Periodically samples a mini-batch from the buffer, calculates **TD error**, and performs a gradient descent step on the network.
   - Over time, **epsilon** is decayed to shift from exploration to exploitation.

---

## Project Structure

- **`agents/base.py`**: Defines a `BaseAgent` interface (or parent class) for RL agents.
- **`agents/dqn_agent.py`**: Contains the DQN agent class (`DQNAgent`) with core RL logic—policy network, target network, replay buffer, training loop, etc.
- **`action_mapping.py`**: Custom function(s) to map discrete indices or macros to 12-bit arrays for Street Fighter II button presses.
- **`train.py`** (example script): Shows how to instantiate the environment, create the agent, and run episodes.
- **`README.md`**: The file you’re reading now.

---

## Getting Started

1. **Install Dependencies**
   - Python 3.7+
   - `torch` (PyTorch)
   - `gym-retro`
   - `numpy`
   - Others listed in `requirements.txt` (if provided).

2. **Obtain ROMs**
   - Retro Gym requires you to **provide your own game ROM** for Street Fighter II. Follow instructions in the [gym-retro docs](https://pypi.org/project/gym-retro/) on how to import ROMs.

3. **Run Training**
   ```bash
   python train.py
   ```
   - This script typically runs training episodes, logs results, and occasionally saves model checkpoints.

4. **Checkpoints**
   - Model weights, optimizer states, and training counters are often saved after certain intervals (e.g., every N episodes or time steps).

---

## Limitations & Future Work

- **Complexity**: Street Fighter is a challenging environment. Mastery often requires thousands of episodes and/or advanced algorithms (like PPO, hierarchical RL, or imitation learning).
- **Compute**: On a single laptop, training can be **slow**. Frame skipping, smaller action spaces, or shorter round times can accelerate it.
- **Reward Shaping**: Balancing the reward function is crucial. Over-penalizing or over-rewarding certain actions can lead to suboptimal behaviors.

---

## Contributing

Feel free to open issues or submit pull requests if you’d like to:
- Add new action mappings or macros.
- Experiment with different reward functions.
- Improve the network architecture or training scripts.

---

## References & Acknowledgments

- **OpenAI Retro Gym**: [github.com/openai/retro](https://github.com/openai/retro)
- **OpenAI Gym**: [gym.openai.com](https://gym.openai.com/)
- **Deep Q-Network** original paper: ["Playing Atari with Deep Reinforcement Learning" (Mnih et al.)](https://arxiv.org/abs/1312.5602)

---

**Enjoy training your own Street Fighter II agent!** If you have questions or suggestions, please reach out or open an issue.
