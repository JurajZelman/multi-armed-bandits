"""Plotting functions for bandit algorithms."""

import matplotlib.pyplot as plt


def set_plot_style() -> None:
    """Set the plotting style."""
    plt.style.use("seaborn-v0_8")
    plt.rcParams.update({"axes.prop_cycle": plt.cycler("color", plt.cm.tab10.colors)})
    # Change to computer modern font and increase font size
    plt.rcParams.update({"font.family": "cmr10", "font.size": 12})
    plt.rcParams.update({"axes.formatter.use_mathtext": True})

    small_size = 16
    medium_size = 18
    bigger_size = 20

    plt.rc("font", size=small_size)  # controls default text sizes
    plt.rc("axes", titlesize=small_size)  # fontsize of the axes title
    plt.rc("axes", labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=small_size)  # fontsize of the tick labels
    plt.rc("legend", fontsize=small_size)  # legend fontsize
    plt.rc("figure", titlesize=bigger_size)  # fontsize of the figure title
