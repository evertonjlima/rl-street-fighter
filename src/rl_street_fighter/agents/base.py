"""Abstract BaseAgent class and general description"""
import numpy as np


class BaseAgent:
    def __init__(self, action_space):
        """
        Base class for RL agents.

        Args:
            action_space (gym.Space): The action space of the environment.
        """
        self.action_space = action_space

    def act(self, observation: np.ndarray):
        """
        Select an action given the current observation.

        Args:
            observation (array-like): The current observation from the environment.

        Returns:
            array-like: The action to take.
        """
        raise NotImplementedError("The `act` method must be implemented by subclasses.")


class RndAgent(BaseAgent):
    def __init__(self, action_space):
        """
        Agent that performs random actions

        Args:
            action_space (gym.Space): The action space of the environment.
        """
        self.action_space = action_space

    def act(self, observation: np.ndarray):
        """
        Ignores observation and returns an random action

        Returns:
        array-like: The action to take.
        """

        return np.random.randint(0, 2, self.action_space.shape)
