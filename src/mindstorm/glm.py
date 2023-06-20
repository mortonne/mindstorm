"""General linear modeling of fMRI data."""

from pathlib import Path
import logging
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib as mpl
import matplotlib.pyplot as plt
from nilearn.glm import first_level
from nipype.interfaces import afni
import click


def create_simple_design(events, trial_column, n_vol, tr, time_offset, high_pass=0):
    """
    Create a simple design matrix.

    Create a linear model design matrix for a "simple" design, i.e.,
    one that only has one event regressor to be modeled.

    Parameters
    ----------
    events : pandas.DataFrame
        BIDS-compliant events table with onset and duration columns.

    trial_column : str
        Column of events to convolve with a hemodynamic response function.

    n_vol : int
        Number of volumes in the timeseries to be modeled.

    tr : float
        Repetition time, i.e., the sampling rate of the timeseries to
        be modeled, in seconds.

    time_offset : float
        Start time of the first timeseries sample, in seconds.

    high_pass : float
        Cutoff frequency for the high-pass filter in Hz.

    Returns
    -------
    design : pandas.DataFrame
        Design matrix with a convolved event regressor and high-pass filter
        nuisance regressors.
    """

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


def create_confound_matrix(
    confounds, regressors=None, censor_motion=False, censor_motion_range=(-1, 2)
):
    """
    Prepare confound matrix from fMRIPrep output.

    Parameters
    ----------
    confounds : pandas.DataFrame
        Table of confound timeseries output by fMRIPrep.

    regressors : list of str
        Confound regressors to include. Each regressor will be
        mean-centered.

    censor_motion : bool
        If true, high-motion volumes will be censored (i.e., a
        regressor will be added for each censored volumne to regress
        out any activation specific to that timepoint).

    censor_motion_range : tuple of int
        First and last volumes to censor, relative to high-motion
        volumes at time zero (e.g., -1 indicates volume before high-
        motion volume, 2 indicates two timepoints after).

    Returns
    -------
    nuisance : numpy.ndarray
        Matrix of nuisance regressor timeseries.

    nuisance_names : list of str
        Name of each included nuisance regressor.
    """
    nuisance_names = []

    # create nuisance regressor matrix
    if regressors is not None:
        raw = confounds.get(regressors).to_numpy()
        nuisance = raw - np.nanmean(raw, 0)
        nuisance[np.isnan(nuisance)] = 0
        nuisance_names.extend(regressors)
    else:
        nuisance = None

    # exclude motion outliers
    if censor_motion:
        outliers = confounds.filter(like="motion_outlier")
        if not outliers.empty:
            # get high-motion times
            motion_ind = np.where(outliers)[0]

            # add offsets to create window around those times
            offsets = np.arange(censor_motion_range[0], censor_motion_range[1] + 1)

            # get unique and valid samples to censor
            censor_ind = np.unique(motion_ind[:, np.newaxis] + offsets)
            n_sample = outliers.shape[0]
            censor_ind = censor_ind[(censor_ind >= 0) & (censor_ind < n_sample)]

            # assign separate regressor for each censored time point
            n_censor = len(censor_ind)
            mat = np.zeros((n_sample, n_censor))
            mat[censor_ind, np.arange(n_censor)] = 1
            if nuisance is not None:
                nuisance = np.hstack([nuisance, mat])
            else:
                nuisance = mat
            censor_names = [f"motion_{i + 1:03}" for i in range(mat.shape[1])]
            nuisance_names.extend(censor_names)
    return nuisance, nuisance_names


def estimate_betaseries(data, design, confound=None, category=None):
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

    category : numpy.ndarray
        [EVs] array with the category label of each EV.

    Returns
    -------
    beta : numpy.ndarray
        [EVs by voxels] array of beta estimates.
    """
    n_trial = design.shape[1]
    n_sample = data.shape[0]
    beta_maker = np.zeros((n_trial, n_sample))
    trial_evs = list(range(n_trial))
    for i, ev in enumerate(trial_evs):
        # this trial
        dm_trial = design[:, ev, np.newaxis]

        # other trials, summed together
        if category is not None:
            ucat = np.unique(category)
            dm_otherevs = np.zeros((design.shape[0], len(ucat)))
            for j, cat in enumerate(ucat):
                other_evs = [
                    x for x, c in zip(trial_evs, category) if x != ev and c == cat
                ]
                dm_otherevs[:, j] = np.sum(design[:, other_evs], 1)
        else:
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
    events_path,
    events_field,
    tr,
    time_offset,
    func_path,
    mask_name,
    mask_file,
    out_dir,
    subject,
    task,
    run,
    space,
    events_category=None,
    nuisance=None,
    nuisance_names=None,
    high_pass=0,
    smooth_fwhm=None,
    sort_field=None,
):
    if sort_field is None:
        sort_field = events_field

    # load events
    events = pd.read_table(events_path, dtype={"group": "str"})

    # add an EV name column (necessary to avoid name restrictions in nilearn)
    evs = events.sort_values(sort_field)[events_field].unique()
    n_ev = len(evs)
    ev_inds = np.arange(n_ev)
    ev_names = [f"ev_{e + 1:03}" for e in ev_inds]
    events["ev"] = events[events_field]
    events["ev_index"] = events[events_field].map(dict(zip(evs, ev_names)))

    # create design matrix
    img = nib.load(func_path)
    n_vol = img.header["dim"][4]
    design = create_simple_design(
        events, "ev_index", n_vol, tr, time_offset, high_pass=high_pass
    )
    design_header = design.columns.to_list()[:-1]

    # get corresponding events; keep any fields that are consistent across
    # presentations
    n = events.groupby("ev_index").apply(lambda x: x.nunique()).reset_index(drop=True)
    consistent = (n == 1).all()
    consistent_fields = consistent[consistent].index
    ev_events = (
        events.groupby("ev_index")
        .first()
        .reset_index()
        .get(consistent_fields)
        .drop(columns=["ev", "ev_index"])
    )
    if events_category is not None:
        category = ev_events[events_category]
    else:
        category = None

    # create confound matrix
    mat = design.iloc[:, :n_ev].to_numpy()
    if nuisance is not None:
        confound = np.hstack((design.iloc[:, n_ev:-1].to_numpy(), nuisance))
        design_header.extend(nuisance_names)
    else:
        confound = design.iloc[:, n_ev:-1]
    full_design = pd.DataFrame(np.hstack([mat, confound]), columns=design_header)

    # prepare output directory
    sub_path = Path(out_dir) / f"sub-{subject}" / "func"
    sub_path.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-{subject}_task-{task}_run-{run}"

    # smooth functional data within mask
    if smooth_fwhm is not None:
        smooth_path = (
            sub_path
            / f"{prefix}_space-{space}_label-{mask_name}_desc-smooth_bold.nii.gz"
        )
        logging.getLogger("nipype.interface").setLevel(0)
        bim = afni.BlurInMask()
        bim.inputs.in_file = func_path
        bim.inputs.mask = mask_file
        bim.inputs.fwhm = smooth_fwhm
        bim.inputs.out_file = smooth_path
        bim.inputs.options = "-overwrite"
        bim.run()
        func_path = smooth_path

    # load functional data
    bold_vol = nib.load(func_path)
    mask_vol = nib.load(mask_file)
    bold_img = bold_vol.get_fdata()
    mask_img = mask_vol.get_fdata().astype(bool)
    data = bold_img[mask_img].T

    # estimate betaseries
    beta = estimate_betaseries(data, mat, confound=confound, category=category)
    out_data = np.zeros([*mask_img.shape, beta.shape[0]])
    out_data[mask_img, :] = beta.T

    # save betaseries image
    beta_path = sub_path / f"{prefix}_space-{space}_label-{mask_name}_betaseries.nii.gz"
    new_img = nib.Nifti1Image(out_data, mask_vol.affine, mask_vol.header)
    nib.save(new_img, beta_path)

    # save mask
    mask_path = sub_path / f"{prefix}_space-{space}_label-{mask_name}_mask.nii.gz"
    new_img = nib.Nifti1Image(mask_img, mask_vol.affine, mask_vol.header)
    nib.save(new_img, mask_path)

    # save design
    design_path = sub_path / f"{prefix}_desc-design_timeseries.tsv"
    full_design.to_csv(design_path, sep="\t", index=False, na_rep="n/a")
    mpl.use("Agg")
    fig, ax = plt.subplots()
    ax.axis("off")
    full_mat = full_design.to_numpy()
    full_min = np.min(full_mat, axis=0)
    full_max = np.max(full_mat, axis=0)
    ax.matshow((full_mat - full_min) / (full_max - full_min))
    fig.savefig(design_path.with_suffix(".png"), pad_inches=0, dpi=600)

    # save corresponding events
    evs_path = sub_path / f"{prefix}_desc-events_timeseries.tsv"
    ev_events.to_csv(evs_path, sep="\t", index=False, na_rep="n/a")


@click.command()
@click.argument("bold_file", type=click.Path(exists=True))
@click.argument("tr", type=float)
@click.argument("time_offset", type=float)
@click.argument("mask_name", type=str)
@click.argument("mask_file", type=click.Path(exists=True))
@click.argument("events_file", type=click.Path(exists=True))
@click.argument("events_field", type=str)
@click.argument("out_dir", type=click.Path())
@click.argument("subject", type=str)
@click.argument("task", type=str)
@click.argument("run", type=str)
@click.argument("space", type=str)
@click.option(
    "--events-category",
    type=str,
    help="Field with event category (categories are modeled as separate regressors)",
)
@click.option(
    "--sort-field",
    type=str,
    help="Colon-separated list of fields to sort by for output order",
)
@click.option("--high-pass", help="Highpass filter in Hz", type=float)
@click.option("--smooth", help="Smoothing kernel FWHM", type=float)
@click.option(
    "--confound-file", help="Path to confound matrix file", type=click.Path(exists=True)
)
def betaseries(
    bold_file,
    tr,
    time_offset,
    mask_name,
    mask_file,
    events_file,
    events_field,
    out_dir,
    subject,
    task,
    run,
    space,
    events_category,
    sort_field,
    high_pass,
    smooth,
    confound_file,
):
    """
    Create a betaseries image estimating stimulus activity for a functional run.

    This script is designed to be flexible in where data are located, but assumes
    that confound regressors are pre-made and stored in a text file.

    \b
    Required inputs
    ---------------
    bold_file
        Path to NIfTI file with functional timeseries data.
    tr
        Repetition time in s.
    time_offset
        Time in s that frames were collected, relative to time zero.
    mask_name
        Name of the mask to use when selecting voxels to analyze.
    mask_file
        Path to a binary mask indicating voxels to analyze.
    events_file
        Path to a tab- or comma-separated values file with task events. Must
        include onset and duration fields (see BIDS specification for task events).
    events_field
        Column of the events file to use to indicate individual EVs (explanatory
        variables) to model.
    out_dir
        Path to main output directory.
    subject
        Subject ID.
    task
        Task code.
    run
        Run number.
    space
        Space in which the data should be analyzed (must have been a space output
        by fMRIPrep).
    """
    if sort_field is None:
        sort_field = events_field
    else:
        sort_field = sort_field.split(":")

    # load nuisance regressors
    if confound_file is not None:
        nuisance = np.loadtxt(confound_file)
        nuisance_names = [f"confound_{i + 1:03}" for i in range(nuisance.shape[1])]
    else:
        nuisance = None
        nuisance_names = []

    # run betaseries estimation and save results
    run_betaseries(
        events_file,
        events_field,
        tr,
        time_offset,
        bold_file,
        mask_name,
        mask_file,
        out_dir,
        subject,
        task,
        run,
        space,
        events_category=events_category,
        nuisance=nuisance,
        nuisance_names=nuisance_names,
        high_pass=high_pass,
        smooth_fwhm=smooth,
        sort_field=sort_field,
    )


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("fmriprep_dir", type=click.Path(exists=True))
@click.argument("out_dir", type=click.Path())
@click.argument("subject", type=str)
@click.argument("task", type=str)
@click.argument("run", type=str)
@click.argument("space", type=str)
@click.argument("mask_name", type=str)
@click.argument("mask_file", type=click.Path(exists=True))
@click.argument("events_field", type=str)
@click.option(
    "--events-category",
    type=str,
    help="Field with event category (categories are modeled as separate regressors)",
)
@click.option(
    "--sort-field",
    type=str,
    help="Colon-separated list of fields to sort by for output order",
)
@click.option("--high-pass", help="Highpass filter in Hz", type=float, default=0)
@click.option("--smooth", help="Smoothing kernel FWHM", type=float)
@click.option(
    "--confound-measures", help="Colon-separated list of confound measures to include"
)
@click.option(
    "--censor-motion/--no-censor-motion",
    default=False,
    help="Censor times around high-motion frames (default is no censoring)",
)
@click.option(
    "--censor-motion-range",
    type=int,
    nargs=2,
    default=(-1, 2),
    help="Range of frames to exclude around high motion (default is (-1, 2))",
)
def betaseries_bids(
    data_dir,
    fmriprep_dir,
    out_dir,
    subject,
    task,
    run,
    space,
    mask_name,
    mask_file,
    events_field,
    events_category,
    sort_field,
    high_pass,
    smooth,
    confound_measures,
    censor_motion,
    censor_motion_range,
):
    """
    Create a betaseries image estimating stimulus activity for a functional run.

    This script is designed to use BIDS-formatted data and data preprocessed
    using fMRIPrep.

    \b
    Required inputs
    ---------------
    data_dir
        Path to a BIDS-compliant dataset with task events.
    fmriprep_dir
        Path to fMRIPrep output.
    out_dir
        Path to main output directory.
    subject
        Subject ID.
    task
        Task code.
    run
        Run number.
    space
        Space in which the data should be analyzed (must have been a space output
        by fMRIPrep).
    mask_name
        Name of the mask to use when selecting voxels to analyze.
    mask_file
        Path to a binary mask indicating voxels to analyze.
    events_field
        Column of the events file to use to indicate individual EVs (explanatory
        variables) to model.
    """
    data_dir = Path(data_dir)
    fmriprep_dir = Path(fmriprep_dir)
    if sort_field is None:
        sort_field = events_field
    else:
        sort_field = sort_field.split(":")
    if confound_measures is not None:
        confound_measures = confound_measures.split(":")

    # task events, functional data, and confounds
    events_path = (
        data_dir
        / f"sub-{subject}"
        / "func"
        / f"sub-{subject}_task-{task}_run-{run}_events.tsv"
    )
    func_path = (
        fmriprep_dir
        / f"sub-{subject}"
        / "func"
        / f"sub-{subject}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
    )
    confound_path = (
        fmriprep_dir
        / f"sub-{subject}"
        / "func"
        / f"sub-{subject}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
    )

    # load functional timeseries information
    func_json = func_path.with_suffix("").with_suffix(".json")
    with open(func_json, "r") as f:
        func_param = json.load(f)
    tr = func_param["RepetitionTime"]
    time_offset = func_param["StartTime"]

    # create nuisance regressor matrix
    confounds = pd.read_table(confound_path)
    nuisance, nuisance_names = create_confound_matrix(
        confounds, confound_measures, censor_motion, censor_motion_range
    )

    # run betaseries estimation and save results
    run_betaseries(
        events_path,
        events_field,
        tr,
        time_offset,
        func_path,
        mask_name,
        mask_file,
        out_dir,
        subject,
        task,
        run,
        space,
        events_category=events_category,
        nuisance=nuisance,
        nuisance_names=nuisance_names,
        high_pass=high_pass,
        smooth_fwhm=smooth,
        sort_field=sort_field,
    )
