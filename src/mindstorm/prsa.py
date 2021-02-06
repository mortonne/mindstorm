"""Display task searchlight analysis in brainiak."""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
import scipy.optimize as optim
from scipy.spatial.distance import pdist, cdist, squareform


def labels2rdm(labels):
    """Create an RDM from a vector with categorical labels."""
    n = len(labels[0])
    # get pairs where all labels are equal
    conj = np.ones((n, n), dtype=bool)
    for label in labels:
        iseq = label[:, np.newaxis] == label.flatten()
        conj = np.logical_and(conj, iseq)

    # make into an RDM
    mat = 1 - np.asarray(conj, "float")

    mat[np.diag_indices(mat.shape[0])] = 0
    return mat


def pair_and(x, y=None):
    """Pairs where conditions are both true."""
    if y is None:
        y = x
    return np.asarray(x)[:, None] & np.asarray(y)[:, None].T


def pair_eq(x, y=None):
    """Pairs where conditions are equal."""
    if y is None:
        y = x
    return np.asarray(x)[:, None] == np.asarray(y)[:, None].T


def pair_neq(x, y=None):
    """Pairs where conditions are not equal."""
    if y is None:
        y = x
    return np.asarray(x)[:, None] != np.asarray(y)[:, None].T


def perm_z(stat_perm):
    """Calculate a z-statistic from permutation results."""
    if np.any(np.isnan(stat_perm) | np.isinf(stat_perm)):
        warnings.warn("Invalid permutation statistic.", RuntimeWarning)
        return np.nan
    elif len(np.unique(stat_perm)) == 1:
        warnings.warn("Permutation statistic does not vary.", RuntimeWarning)
        return np.nan

    p = np.mean(stat_perm >= stat_perm[0])
    e = 1.0 / len(stat_perm)
    if p > (1 - e):
        p = 1 - e
    return stats.norm.ppf(1 - p)


def init_pRSA(n_perm, model_rdms, rank=True):
    """Prepare model RDMs for partial RSA.

    Parameters
    ----------
    n_perm : int
        Number of model permutations to test to establish a null
        distribution for each partial correlation.

    model_rdms : list of numpy arrays
        Representational dissimilarity matrix for each of a set of
        models to test. Each matrix should be [trials x trials],
        giving the dissimilarity between trials predicted by that model.

    rank : bool
        If true, RDMs will be rank transformed.

    Returns
    -------
    rsa_def : dict
        Information needed to run the partial RSA. Includes:
        rsa_def['model_mats'] : list of numpy arrays
            Design matrix for each model. This includes rank-transformed
            representational dissimilarity vectors for all of the other
            models.
        rsa_def['model_resid'] : list of numpy arrays
            Residual vectors for each model after rank transforming and
            controlling for all the other models.
    """

    n_model = len(model_rdms)
    n_item = model_rdms[0].shape[0]

    # generate random indices to use across all models
    rand_ind = [
        np.random.choice(np.arange(n_item), n_item, False) for i in range(n_perm)
    ]
    rand_ind.insert(0, np.arange(n_item))

    # put models in vector format and rank-order them
    if rank:
        model_vecs = np.asarray(
            [stats.rankdata(squareform(rdm)) for rdm in model_rdms]
        ).T
    else:
        model_vecs = np.asarray([squareform(rdm) for rdm in model_rdms]).T

    # make control design matrices for regression
    model_mats = []
    for i in range(n_model):
        other = [j for j in range(n_model) if j is not i]
        control_vecs = model_vecs[:, other]
        intercept = np.ones((control_vecs.shape[0], 1))
        mat = np.hstack((control_vecs, intercept))
        model_mats.append(mat)

    # residualize each model against the others
    model_resid = []
    for i in range(n_model):
        resid = []
        for ind in rand_ind:
            rand_rdm = model_rdms[i][np.ix_(ind, ind)]
            rand_model = squareform(rand_rdm)
            if rank:
                rand_model = stats.rankdata(rand_model)
            beta = optim.nnls(model_mats[i], rand_model)[0]
            res = rand_model - model_mats[i].dot(beta)
            resid.append(res)
        model_resid.append(np.asarray(resid))

    return {"model_mats": model_mats, "model_resid": model_resid}


def perm_partial(data_vec, model_mat, model_resid):
    """Calculate partial correlation for multiple permutations."""

    # regress to get residuals from data similarity vector
    beta = optim.nnls(model_mat, data_vec)[0]
    data_resid = data_vec - model_mat.dot(beta)

    # correlate with the residualized (random) model
    xmat = data_resid.reshape((1, len(data_resid)))
    stat = 1 - cdist(xmat, model_resid, "correlation").squeeze()
    if np.any(np.isnan(stat)):
        raise ValueError("statistic is undefined.")
    return stat


def call_pRSA(subj, mask, sl_rad, bcast_var, rank=True):
    """Function to pass when running partial RSA searchlight.

    Parameters
    ----------
    subj : list of numpy arrays
        Each element contains a 4D array of functional data for one
        subject.

    mask : numpy array
        A 3D array with the current mask for the searchlight.

    sl_rad : int
        Searchlight radius.

    bcast_var : dict
        The dict created by init_pRSA.

    rank : bool
        If true, rank transform data.

    Returns
    -------
    stat_all : tuple
        The z-statistic for each of the partial correlations, compared
        to the null permutation distribution.
    """

    # data representational dissimilarity vector
    data = subj[0][mask, :].T
    data_vec = pdist(data, "correlation")
    if rank:
        data_vec = stats.rankdata(data_vec)

    # unpack global data
    model_mats = bcast_var["model_mats"]
    model_resid = bcast_var["model_resid"]

    # calculate z-statistic for each partial correlation
    stat_all = []
    for i in range(len(model_mats)):
        stat = perm_partial(data_vec, model_mats[i], model_resid[i])
        stat_all.append(perm_z(stat))

    return tuple(stat_all)


def sign_perm(mat, n_perm, beta):
    """Test for significantly positive values with FDR correction."""
    # generate random sign flips
    n_samp, n_test = mat.shape
    rand_sign = np.hstack(
        (np.ones((n_samp, 1)), np.random.choice([-1, 1], (n_samp, n_perm)))
    )

    # apply random sign to all conditions
    mat_perm = mat[:, :, None] * rand_sign[:, None, :]

    # ttest vs zero for permuted and actual stats
    perm_t, perm_p = stats.ttest_1samp(mat_perm, 0, axis=0)
    t, p = stats.ttest_1samp(mat_perm[:, :, 0], 0, axis=0)

    # convert to right-tailed test
    perm_p /= 2
    perm_p[perm_t < 0] = 1 - perm_p[perm_t < 0]
    p /= 2
    p[t < 0] = 1 - p[t < 0]

    # q-value at each actual p-value
    ERstar = np.zeros(p.shape)
    r = np.zeros(p.shape)
    rstarb = np.zeros(p.shape)
    m = n_test
    q = np.zeros(p.shape)
    for i, a in enumerate(p):
        # expected value of number of rejections under the null
        Rstar = np.sum(perm_p <= a, 0)
        ERstar[i] = np.mean(Rstar)

        # 1 - beta quantile of Rstar
        rstarb[i] = np.quantile(Rstar, 1 - beta)

        # number of actual rejections
        r[i] = np.sum(p <= a)

        # FDR local estimator
        if (r[i] - rstarb[i]) >= (a * m):
            q[i] = ERstar[i] / (ERstar[i] + r[i] - (a * m))
        else:
            q[i] = np.mean(Rstar >= 1)

    # package results
    std = np.std(mat, axis=0, ddof=1)
    mean = np.mean(mat, axis=0)
    sem = stats.sem(mat, axis=0)
    df = pd.DataFrame({
        't': t, 'p': p, 'q': q, 'ERstar': ERstar, 'rstarb': rstarb, 'r': r,
        'mean': mean, 'std': std, 'sem': sem, 'd': np.abs(mean) / std
    })
    return df
