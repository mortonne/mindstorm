"""Plot statistics using Seaborn."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def add_swarm_bar_sig(ax, sig_ind, y_offset=None):
    """Add significance stats to swarm bar plot."""
    # center of each bar in left/right order
    bx = np.array(sorted([p.get_x() + (p.get_width() / 2) for p in ax.patches]))

    # highest point in each point plot in left/right order
    collections = [c for c in ax.collections if c.get_offsets().shape[0] > 1]
    cy = np.array([np.max(c.get_offsets()[:, 1]) for c in collections])
    cx = np.array([np.mean(c.get_offsets()[:, 0]) for c in collections])
    cy = cy[np.argsort(cx)]

    # define y point relative to the highest point
    if y_offset is None:
        y_lim = ax.get_ylim()
        y_offset = (y_lim[1] - y_lim[0]) * .15

    # plot significance markers
    ax.plot(
        bx[sig_ind],
        cy[sig_ind] + y_offset,
        markersize=8,
        color='k',
        linestyle='',
        marker=(5, 2, 0),
    )


def plot_swarm_bar(
    data=None,
    x=None,
    y=None,
    hue=None,
    dark=None,
    light=None,
    dodge=False,
    point_kind='swarm',
    legend=True,
    width=None,
    sig_ind=None,
    ax=None,
    point_kws={},
    bar_kws={},
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

    point_kind : {'swarm', 'strip'}, optional
        Method for plotting the individual data points in each bin.

    legend : bool, optional
        If true, and the hue input is set, include a legend on the plot.

    width : float, optional
        Width of the bars.

    sig_ind : list of int, optional
        Indices of bars (from left to right) to label as significant.

    ax : matplotlib.axes.Axes, optional
        Axes object to draw the plot onto; if not specified, will use
        the current Axes.

    point_kws : dict, optional
        Options when plotting points using seaborn.swarmplot or
        seaborn.stripplot.

    bar_kws : dict, optional
        Options when plotting bars using seaborn.barplot.

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
    point_prop.update(point_kws)
    if point_kind == 'swarm':
        sns.swarmplot(data=data, x=x, y=y, hue=hue, ax=ax, **point_prop)
    elif point_kind == 'strip':
        sns.stripplot(data=data, x=x, y=y, hue=hue, ax=ax, **point_prop)
    else:
        raise ValueError(f'Invalid point plot kind: {point_kind}')

    # plot error bars for the mean
    bar_prop = {
        'color': 'k',
        'errwidth': 0.8,
        'capsize': 0.425,
        'edgecolor': 'k',
        'linewidth': 0.75,
        'errcolor': 'k',
        'dodge': dodge,
        'palette': light,
    }
    bar_prop.update(bar_kws)
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **bar_prop)

    # remove overall xlabel and increase size of x-tick labels
    ax.set_xlabel('')
    ax.tick_params(axis='x', labelsize='large')

    # fix ordering of plot elements
    plt.setp(ax.lines, zorder=100, linewidth=1.25, label=None, clip_on=False)
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

    # add significance markers
    if sig_ind is not None:
        add_swarm_bar_sig(ax, sig_ind)
    return ax


def plot_sig(x, y, spacing, line_kws={}, marker_kws={}, ax=None):
    """Plot brackets with a significance indicator."""
    if ax is None:
        ax = plt.gca()

    # bar centers
    x1, x2 = x

    # largest y-values
    y1, y2 = y

    # line and annotation y-values
    yl1 = max(y1, y2) + spacing
    yl2 = yl1 + spacing
    yl3 = yl2 + spacing

    # brackets connecting bars
    line_prop = {'linewidth': 0.75, 'color': 'k'}
    line_prop.update(line_kws)
    ax.plot([x1, x1, x2, x2], [yl1, yl2, yl2, yl1], **line_prop)

    # significance marker
    marker_prop = {'marker': (5, 2, 0), 'markersize': 8, 'color': 'k'}
    marker_prop.update(marker_kws)
    ax.plot((x1 + x2) / 2, yl3, **marker_prop)
