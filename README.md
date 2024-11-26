# Multi-armed bandits

## Introduction

The following is a simple implementation of the **multi-armed bandit problem**. The problem is as follows. You are faced with $k$ distributions (or *arms*) which are not known to you. As an example, these could be several variations of *high-frequency trading algorithms* with unknown returns. You can sample from these distributions, and you want to *maximize your reward*, i.e. maximize your average return over time. Currently, the distributions are assumed to be *stationary*.

In this implementation, I additionally include a *hold* parameter for smoother optimization, which forces the algorithm to hold the current action for a certain number of steps. This could be desired for example in the context of the trading algorithms, where you want to test a certain algorithm for a certain number of steps before switching to another one.

The bandit simulation can be found in [`main.ipynb`](/main.ipynb) notebook.

## Bandits

There are several variations of bandit algorithms, see [Sutton, Barto (2015)](https://inst.eecs.berkeley.edu//~cs188/sp20/assets/files/SuttonBartoIPRLBook2ndEd.pdf). The following are implemented in [`bandits/bandits.py`](/bandits/bandits.py):

- **Greedy bandit algorithm** - The simplest bandit algorithm which always chooses the arm with the highest estimated reward, i.e.
    $$A_t = \arg \max_a Q_t(a)$$
    where $Q_t(a)$ is the estimated reward of arm $a$ at time $t$.
- **Epsilon-greedy bandit algorithm** - A simple bandit algorithm which chooses the arm with the highest estimated reward with probability $1-\epsilon$, and chooses a random arm with probability epsilon, i.e. for a given epsilon $\epsilon > 0$,
    $$A_t = \arg \max_a Q_t(a) \quad \text{with} \ \ p=1-\epsilon$$
    $$A_t = \text{random arm} \quad \text{with} \ \ p=\epsilon.$$
- **Upper confidence bound (UCB) bandit algorithm** - A bandit algorithm which chooses the arm with the highest estimated reward plus a bonus term which depends on the number of times the arm has been sampled. This bonus term is equal to the square root of the logarithm of the total number of steps divided by the number of times the arm has been sampled, i.e. $$A_t = \arg \max_a \left[Q_t(a) + c\sqrt{\frac{\log(t)}{N(a)}} \right],$$
    where $c$ is a constant which determines the exploration-exploitation trade-off.
