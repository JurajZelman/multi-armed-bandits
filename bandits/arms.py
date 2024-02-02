"""Arms (samplers) for the bandit problem."""

from abc import ABC, abstractmethod

import numpy as np


class Arm(ABC):
    """Base class for bandit arms."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the arm."""

    @abstractmethod
    def sample(self) -> float:
        """
        Sample the arm.

        Returns:
            The reward for the arm.
        """


class NormalArm(Arm):
    """An arm with a normal distribution."""

    def __init__(self, mu: float, sigma: float, seed: int = 42) -> None:
        """
        Initialize the arm.

        Args:
            mu: The mean of the normal distribution.
            sigma: The standard deviation of the normal distribution.
            seed: The random seed for reproducibility. Default is `42`.
        """
        self.mu = mu
        self.sigma = sigma
        self.rng = np.random.default_rng(seed=seed)

    def sample(self) -> float:
        """
        Sample the arm.

        Returns:
            The reward for the arm.
        """
        return self.rng.normal(self.mu, self.sigma)
