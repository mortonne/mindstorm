"""Plot statistics using Seaborn."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_swarm_bar(
    data=None,
    x=None,
    y=None,
    hue=None,
    dark=None,
    light=None,
    dodge=False,
    capsize=0.425,
    point_kind='swarm',
    legend=True,
    width=None,
    ax=None,
):
    """
    Make a bar plot with individual points and error bars.

    Parameters
    ----------
    data : pandas.DataFrame
        Long data frame with variables indicating at least x and y values.

    x : str
        Column in data to determine bar x-positions.

    y : str
        Column in data to determine values to plot.

    hue : str, optional
        Column in data to determine bar and point hues.

    dark : str, optional
        Specification for Seaborn palette; used to plot points.

    light : str, optional
        Specification for Seaborn palette; used to plot bars.

    dodge : bool, optional
        If true, hues will be plotted at different x-values. Used to
        create grouped bar plots.

    capsize : float, optional
        Size of the error bar caps.

    point_kind : {'swarm', 'strip'}, optional
        Method for plotting the individual data points in each bin.

    legend : bool, optional
        If true, and the hue input is set, include a legend on the plot.

    width : float, optional
        Width of the bars.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if not specified, will use
        the current Axes.

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes that the plot was drawn onto.
    """
    if dark is None:
        dark = 'ch:rot=-.5, light=.7, dark=.3, gamma=.6'
    if light is None:
        light = 'ch:rot=-.5, light=.7, dark=.3, gamma=.2'
    data = data.reset_index()

    if ax is None:
        ax = plt.gca()

    # plot individual points
    point_prop = {
        'size': 3,
        'linewidth': 0.1,
        'edgecolor': 'k',
        'alpha': 0.7,
        'zorder': 3,
        'dodge': dodge,
        'palette': dark,
        'clip_on': False,
    }
    if point_kind == 'swarm':
        sns.swarmplot(
            data=data, x=x, y=y, hue=hue, ax=ax, **point_prop,
        )
    elif point_kind == 'strip':
        sns.stripplot(
            data=data, x=x, y=y, hue=hue, ax=ax, **point_prop,
        )
    else:
        raise ValueError(f'Invalid point plot kind: {point_kind}')

    # plot error bars for the mean
    sns.barplot(
        data=data,
        x=x,
        y=y,
        hue=hue,
        ax=ax,
        dodge=dodge,
        color='k',
        palette=light,
        errwidth=0.8,
        capsize=capsize,
        edgecolor='k',
        linewidth=0.75,
        errcolor='k',
    )

    # remove overall xlabel and increase size of x-tick labels
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize='large')

    # fix ordering of plot elements
    plt.setp(ax.lines, zorder=100, linewidth=1.25, label=None)
    plt.setp(ax.collections, zorder=100, label=None)

    # modify the width of the bar plot
    if width is not None:
        bar_x = []
        for patch in ax.patches:
            # set width
            center = patch.get_x() + patch.get_width() / 2
            patch.set_width(width)

            # adjust the x position
            patch_x = center - width / 2
            patch.set_x(patch_x)
            bar_x.append(patch_x)
        # set the border around the bars to be the same as the bar spacing
        spacing = 1 - width
        xmin = np.min(bar_x) - spacing
        xmax = np.max(bar_x) + width + spacing
        ax.set_xlim(xmin, xmax)

    # delete legend (redundant with the x-tick labels)
    lg = ax.get_legend()
    if lg is not None:
        lg.remove()
        if hue is not None and legend:
            # refresh the legend to remove the swarm points
            ax.legend()
    return ax
