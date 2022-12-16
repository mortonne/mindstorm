"""General linear modeling of fMRI data."""

import os
import numpy as np
import pandas as pd
import nibabel as nib
import sklearn.linear_model as lm
from nilearn.glm import first_level
import click


def create_betaseries_design(events, trial_column, n_vol, tr, time_offset, high_pass=0):
    """Create a design matrix for betaseries estimation."""

    # check for required columns in events
    required_columns = ["onset", "duration", trial_column]
    for column in required_columns:
        if column not in events:
            raise ValueError(f"Must define {column} column in events.")

    # create dataframe with trial information
    trials = pd.DataFrame(
        {
            "onset": events["onset"],
            "duration": events["duration"],
            "trial_type": events[trial_column],
        }
    )

    # create a design matrix
    frame_times = time_offset + np.arange(n_vol) * tr
    design = first_level.make_first_level_design_matrix(
        frame_times, events=trials, high_pass=high_pass
    )
    return design


def create_confound_matrix(confounds, regressors, exclude_motion=False):
    """Prepare betaseries design matrix and confounds."""
    # create nuisance regressor matrix
    raw = confounds.get(regressors).to_numpy()
    nuisance = raw - np.nanmean(raw, 0)
    nuisance[np.isnan(nuisance)] = 0

    # exclude motion outliers
    if exclude_motion:
        outliers = confounds.filter(like="motion_outlier")
        if not outliers.empty:
            nuisance = np.hstack([nuisance, outliers.to_numpy()])
    return nuisance


def estimate_betaseries(data, design, confound=None):
    """
    Estimate beta images for a set of trials.

    Parameters
    ----------
    data : numpy.ndarray
        [timepoints x voxels] array of functional data to model.

    design : numpy.ndarray
        [timepoints x EVs] array. Each explanatory variable (EV) will
        be estimated using a separate model.

    confound : numpy.ndarray
        [timepoints x regressors] array with regressors of no interest,
        which will be included in each model.

    Returns
    -------
    beta : numpy.ndarray
        [EVs by voxels] array of beta estimates.
    """
    # TODO: handle hierarchical designs (separate trial classes)
    n_trial = design.shape[1]
    n_sample = data.shape[0]
    beta_maker = np.zeros((n_trial, n_sample))
    trial_evs = list(range(n_trial))
    for i, ev in enumerate(trial_evs):
        # this trial
        dm_trial = design[:, ev, np.newaxis]

        # other trials, summed together
        other_trial_evs = [x for x in trial_evs if x != ev]
        dm_otherevs = np.sum(design[:, other_trial_evs, np.newaxis], 1)

        # put together the design matrix
        if confound is not None:
            dm_full = np.hstack((dm_trial, dm_otherevs, confound))
        else:
            dm_full = np.hstack((dm_trial, dm_otherevs))
        s = dm_full.shape
        dm_full = dm_full - np.kron(np.ones(s), np.mean(dm_full, 0))[: s[0], : s[1]]
        dm_full = np.hstack((dm_full, np.ones((n_sample, 1))))

        # calculate beta-forming vector
        beta_maker_loop = np.linalg.pinv(dm_full)
        beta_maker[i, :] = beta_maker_loop[0, :]

    # this uses Jeanette Mumford's trick of extracting the beta-forming
    # vector for each trial and putting them together, which allows
    # estimation for all trials at once
    beta = np.dot(beta_maker, data)
    return beta


def run_betaseries(
    raw_dir,
    post_dir,
    mask,
    bold,
    subject,
    run,
    high_pass=0,
    space="T1w",
    mask_dir="func",
    mask_thresh=None,
    exclude_motion=False,
):
    """Estimate betaseries for one run."""
    # TODO: rewrite for general inputs
    # TODO: consider adding smoothing

    tr = 2
    subj_raw = os.path.join(raw_dir, f"sub-{subject}", "func")
    subj_post = os.path.join(post_dir, f"sub-{subject}", "func")

    # task events
    events_file = os.path.join(
        subj_raw, f"sub-{subject}_task-struct_run-{run}_events.tsv"
    )
    if not os.path.exists(events_file):
        raise IOError(f"Events do not exist: {events_file}")

    # ROI/brain mask
    if mask_dir == "func":
        mask_file = get_func_mask(post_dir, subject, "struct", run, space, desc=mask)
    else:
        mask_file = get_anat_mask(post_dir, subject, space, label=mask)
    if not os.path.exists(mask_file):
        raise IOError(f"Mask file does not exist: {mask_file}")

    # BOLD scan
    bold_file = os.path.join(
        subj_post,
        f"sub-{subject}_task-struct_run-{run}_space-{space}_desc-{bold}_bold.nii.gz",
    )
    if not os.path.exists(bold_file):
        raise IOError(f"BOLD file does not exist: {bold_file}")

    # confounds file
    conf_file = os.path.join(
        subj_post, f"sub-{subject}_task-struct_run-{run}_desc-confounds_timeseries.tsv"
    )
    if not os.path.exists(conf_file):
        raise IOError(f"Confounds file does not exist: {conf_file}")

    # create nuisance regressor matrix
    mat, confound = prepare_betaseries_design(
        events_file, conf_file, tr, high_pass, exclude_motion
    )

    # load functional data
    bold_vol = nib.load(bold_file)
    mask_vol = nib.load(mask_file)
    bold_img = bold_vol.get_fdata()
    if mask_thresh is None:
        mask_img = mask_vol.get_fdata().astype(bool)
    else:
        mask_img = mask_vol.get_fdata() > mask_thresh
    data = bold_img[mask_img].T

    # estimate each beta image
    beta = estimate_betaseries(data, mat, confound)

    # estimate model residuals (for smoothness calculation)
    model = lm.LinearRegression()
    design = np.hstack([mat, confound])
    model.fit(design, data)
    resid = data - model.predict(design)
    return beta, resid


@click.command()
@click.argument("bold_file", type=click.Path(exists=True))
@click.argument("tr", type=float)
@click.argument("events_file", type=click.Path(exists=True))
@click.argument("events_id", type=str)
@click.argument("mask_file", type=click.Path(exists=True))
@click.argument("confound_file", type=click.Path(exists=True))
@click.argument("betaseries_file", type=click.Path())
@click.option("--events-category", type=str, help="Field with event category")
@click.option("--hp-filter", help="Highpass filter in Hz", type=float)
@click.option("--smooth", help="Smoothing kernel FWHM", type=float)
@click.option("--confound-measures", help="List of confound measures to include")
@click.option(
    "--confound-measures-file", help="File with list of confound measures to include"
)
def betaseries(
    bold_file,
    events_file,
    events_id,
    mask_file,
    confound_file,
    betaseries_file,
    events_category,
    hp_filter,
    smooth,
    confound_measures,
    confound_measures_file,
):
    # TODO: run functions to estimate model
    # TODO: write out images
    # TODO: add logging
    conf = pd.read_table(confound_file)
    if confound_measures is not None:
        conf_mat = conf[confound_measures]
    elif confound_measures_file is not None:
        # load from file and filter from there
        try:
            with open(confound_measures_file) as f:
                confound_measures = f.readlines()[0].split(",")
        except IOError:
            raise IOError
    else:
        conf_mat = conf
