"""Utilities for plotting dissimilarity matrices."""

import math
import numpy as np
from scipy import stats
from scipy import spatial
import scipy.spatial.distance as sd
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


def plot_pattern(mat, ax=None):
    """Plot a matrix as a simple grayscale image with no axes."""
    if ax is None:
        ax = plt.gca()
    h = ax.matshow(mat, cmap="gray")
    ax.set_axis_off()
    return h


def plot_dsm(
        dsm, rank=False, prange=(1, 99), vlim=None, cmap="viridis", checks=True,
        ax=None
):
    """Plot pairwise dissimilarity values as a matrix."""
    if ax is None:
        ax = plt.gca()

    if rank:
        dsm = sd.squareform(stats.rankdata(sd.squareform(dsm, checks=checks)))

    if vlim is None:
        vlim = np.percentile(sd.squareform(dsm, checks=checks), prange)

    h = ax.pcolor(dsm, vmin=vlim[0], vmax=vlim[1], cmap=cmap)
    ax.set_aspect(1)
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_axis_off()
    return h


def embed(dsm, embedding=None):
    """Calculate a two-dimensional MDS embedding for a dissimilarity matrix."""
    if embedding is None:
        embedding = manifold.MDS(n_components=2, dissimilarity="precomputed")
    X = embedding.fit_transform(dsm)
    x, y = X.T
    return x, y


def embed_align(dsm_set, reference=0, embedding=None):
    """Calculate embedding for dissmilarity matrices and align them."""
    embed_set = [np.vstack(embed(dsm, embedding)).T for dsm in dsm_set]
    ref_embed = embed_set[reference]
    aligned = [None] * len(dsm_set)
    for i, mat in enumerate(embed_set):
        if i == reference:
            continue
        pro1, pro2, disparity = spatial.procrustes(ref_embed, mat)
        if aligned[reference] is None:
            aligned[reference] = pro1
        aligned[i] = pro2
    return aligned


def pad_image(image, shape, value):
    """Pad an image to match a given shape."""
    # center align to fill difference in shape
    full_shape = np.hstack((shape, image.shape[-1]))
    width = full_shape - image.shape
    pad_width = [(math.floor(w / 2), math.ceil(w / 2)) for w in width]
    padded = np.pad(image, pad_width, constant_values=value)
    return padded


def image_matrix(images, shape, background=1):
    """Concatenate stimulus images into a matrix."""
    # get the maximum image size
    image_shapes = np.array([i.shape for i in images])
    max_size = np.max(image_shapes, 0)

    # compile the stimulus matrix
    ind = 0
    mat_rows = []
    for i in range(shape[0]):
        mat_cols = []
        for j in range(shape[1]):
            # pad the image as needed to make everything constant size
            image = images[ind]
            padded = pad_image(image, max_size[:2], background)
            mat_cols.append(padded)
            ind += 1
        mat_rows.append(np.hstack(mat_cols))
    return np.vstack(mat_rows)


def pool_image_matrix(pool, images, shape, **kwargs):
    """Plot an image matrix for a pool."""
    items = pool["stim"].to_list()
    pool_images = [images[item] for item in items]
    mat = image_matrix(pool_images, shape, **kwargs)
    return mat


def image_scatter(x, y, images, ax=None, zoom=1.0, frameon=False):
    """Make a scatter plot with images instead of points."""
    if ax is None:
        ax = plt.gca()

    artists = []
    x, y = np.atleast_1d(x, y)
    for (x0, y0, im) in zip(x, y, images):
        im0 = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(im0, (x0, y0), xycoords="data", frameon=frameon, pad=0)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists


def plot_mds(dsm, images, ind=None, embedding=None, zoom=1.0, ax=None):
    """Plot multidimensional scaling of a dissimilarity matrix."""
    if ind is not None:
        if embedding == "precomputed":
            dsm = dsm[:, ind]
        else:
            dsm = dsm[ind, ind]
        images = images[ind]

    if embedding == "precomputed":
        x, y = dsm
    else:
        x, y = embed(dsm, embedding)

    if np.ptp(x) >= np.ptp(y):
        d1, d2 = x, y
    else:
        d1, d2 = y, x
    artists = image_scatter(d1, d2, images, ax=ax, zoom=zoom)
    return artists


def plot_mds_aligned(
    model_set, model_names, images, embedding=None, fig=None, col_wrap=3
):
    """Plot MDS aligned over multiple models."""
    mset = [model_set[name] for name in model_names]
    aligned = embed_align(mset, embedding)

    if fig is None:
        fig = plt.figure(figsize=(14, 7))

    n_row = int(np.ceil(len(aligned) / col_wrap))
    for i, mat in enumerate(aligned):
        ax = fig.add_subplot(n_row, col_wrap, i + 1)
        plot_mds(mat.T, images, zoom=0.06, ax=ax, embedding="precomputed")
        ax.set_axis_off()
        ax.set_aspect("equal")
        ax.set_title(model_names[i])
    return fig


def plot_models_dsm(model_dsm, fig=None, prange=(1, 99), col_wrap=None):
    """Plot multiple dissimilarity matrices."""
    if fig is None:
        fig = plt.gcf()

    n_model = len(model_dsm)
    if col_wrap is None:
        n_row = 1
        n_col = n_model
    else:
        n_col = col_wrap
        n_row = math.ceil(n_model / col_wrap)

    i = 0
    for name, dsm in model_dsm.items():
        ax = fig.add_subplot(n_row, n_col, i + 1)
        plot_dsm(dsm, ax=ax, prange=prange)
        ax.set_title(name)
        i += 1


def plot_rep_as_dsm(
    data=None, distance="correlation", ax=None, prange=(1, 99), color=None, **plot_kws
):
    """Convert representation to dissimilarity and plot."""
    mat = data.filter(like="dim").to_numpy()
    dsm = sd.squareform(sd.pdist(mat, distance))
    plot_dsm(dsm, ax=ax, prange=prange, **plot_kws)


def plot_rep_as_mds(
    data=None, distance="correlation", ax=None, images=None, color=None, zoom=1,
    embedding=None
):
    """Run dimensionality reduction and plot images."""
    mat = data.filter(like="dim").to_numpy()
    dsm = sd.squareform(sd.pdist(mat, distance))
    x, y = embed(dsm, embedding)
    im_list = [images[stim] for stim in data.stim.to_list()]
    image_scatter(x, y, im_list, ax=ax, zoom=zoom)
