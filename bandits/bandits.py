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

    def __init__(
        self, k: int, label: str, hold: int = 0, seed: int = 42
    ) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            hold: Number of times to hold the selected arm before exploring.
                Default is `0` (no holding).
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.hold = hold
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

        self.hold_count = 0
        self.hold_arm = None

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        # Hold the arm for a number of times before exploring
        if self.hold_arm is not None and self.hold_count < self.hold:
            self.hold_count += 1
            return self.hold_arm

        # Explore new arms
        else:
            self.hold_count = 1
            max_reward = np.max(self.rew_est)
            max_indices = np.where(self.rew_est == max_reward)[0]
            self.hold_arm = self.rng.choice(max_indices)
            return self.hold_arm

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
        self, k: int, label: str, epsilon: float, hold: int = 0, seed: int = 42
    ) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            hold: Number of times to hold the selected arm before exploring.
                Default is `0` (no holding).
            epsilon: The probability of exploration.
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.epsilon = epsilon
        self.hold = hold
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

        self.hold_count = 0
        self.hold_arm = None

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        if self.hold_arm is not None and self.hold_count < self.hold:
            self.hold_count += 1
            return self.hold_arm
        else:
            self.hold_count = 1
            if self.rng.random() < self.epsilon:
                self.hold_arm = self.rng.integers(self.k)
                return self.hold_arm
            max_reward = np.max(self.rew_est)
            max_indices = np.where(self.rew_est == max_reward)[0]
            self.hold_arm = self.rng.choice(max_indices)
            return self.hold_arm

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

    def __init__(
        self, k: int, label: str, c: float, hold: int = 0, seed: int = 42
    ) -> None:
        """
        Initialize the bandit.

        Args:
            k: Number of arms.
            label: The label for the bandit used for plotting.
            c: The exploration parameter.
            hold: Number of times to hold the selected arm before exploring.
                Default is `0` (no holding).
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.k = k
        self.label = label
        self.c = c
        self.hold = hold
        self.rng = np.random.default_rng(seed=seed)
        self.rew_est = np.zeros(k)
        self.counts = np.zeros(k)
        self.rewards = []

        self.hold_count = 0
        self.hold_arm = None

    def select_arm(self) -> int:
        """
        Select an arm.

        Returns:
            The index of the arm to select.
        """
        if self.hold_arm is not None and self.hold_count < self.hold:
            self.hold_count += 1
            return self.hold_arm
        else:
            t = np.sum(self.counts) + 1
            ucb = self.rew_est + self.c * np.sqrt(
                np.log(t) / (self.counts + 1e-5)
            )
            max_reward = np.max(ucb)
            max_indices = np.where(ucb == max_reward)[0]
            self.hold_arm = self.rng.choice(max_indices)
            return self.hold_arm

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
