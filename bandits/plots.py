"""Plotting functions for bandit algorithms."""

import matplotlib.pyplot as plt


def set_plot_style() -> None:
    """Set the plotting style."""
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update(
        {"axes.prop_cycle": plt.cycler("color", plt.cm.tab10.colors)}
    )
    # Change to computer modern font and increase font size
    plt.rcParams.update({"font.family": "cmr10", "font.size": 12})
    plt.rcParams.update({"axes.formatter.use_mathtext": True})

    SMALL_SIZE = 16
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20

    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
