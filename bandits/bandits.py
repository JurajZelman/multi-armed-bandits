"""Bandit algorithms."""

from abc import ABC, abstractmethod

import numpy as np


class Bandit(ABC):
    """Base class for bandit algorithms."""

    @abstractmethod
    def __init__(self, k: int, label: str) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit.
        """

    @abstractmethod
    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """

    @abstractmethod
    def update(self, arm: int, reward: float) -> None:
        """
        Update the bandit with the reward for an arm.

        Args:
            arm: The index of the arm.
            reward: The reward for the arm.
        """


class GreedyBandit(Bandit):
    """Greedy bandit algorithm."""

    def __init__(self, k: int, label: str, seed: int = 42) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        # Select argmax of rewards but break ties randomly
        max_reward = np.max(self.rew_est)
        max_indices = np.where(self.rew_est == max_reward)[0]
        return self.rng.choice(max_indices)

    def update(self, arm: int, reward: float) -> None:
        """
        Update the bandit with the reward for an arm.

        Args:
            arm: The index of the arm.
            reward: The reward for the arm.
        """
        self.counts[arm] += 1
        self.rew_est[arm] += (reward - self.rew_est[arm]) / self.counts[arm]
        self.rewards.append(reward)


class EpsilonGreedyBandit(Bandit):
    """Epsilon greedy bandit algorithm."""

    def __init__(
        self, k: int, label: str, epsilon: float, seed: int = 42
    ) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            epsilon: The probability of exploration.
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.k)
        max_reward = np.max(self.rew_est)
        max_indices = np.where(self.rew_est == max_reward)[0]
        return self.rng.choice(max_indices)

    def update(self, arm: int, reward: float) -> None:
        """
        Update the bandit with the reward for an arm.

        Args:
            arm: The index of the arm.
            reward: The reward for the arm.
        """
        self.counts[arm] += 1
        self.rew_est[arm] += (reward - self.rew_est[arm]) / self.counts[arm]
        self.rewards.append(reward)


class UpperConfidenceBoundBandit(Bandit):
    """Upper confidence bound bandit algorithm."""

    def __init__(self, k: int, label: str, c: float, seed: int = 42) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            c: The exploration parameter.
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.c = c
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        t = np.sum(self.counts) + 1
        ucb = self.rew_est + self.c * np.sqrt(np.log(t) / (self.counts + 1e-5))
        return np.argmax(ucb)

    def update(self, arm: int, reward: float) -> None:
        """
        Update the bandit with the reward for an arm.

        Args:
            arm: The index of the arm.
            reward: The reward for the arm.
        """
        self.counts[arm] += 1
        self.rew_est[arm] += (reward - self.rew_est[arm]) / self.counts[arm]
        self.rewards.append(reward)
