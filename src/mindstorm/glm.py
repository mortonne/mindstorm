"""General linear modeling of fMRI data."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import nibabel as nib
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
    events_path,
    events_field,
    n_vol,
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
    nuisance=None,
    sort_field=None,
):
    if sort_field is None:
        sort_field = events_field

    # load events
    events = pd.read_table(events_path)

    # add an EV name column (necessary to avoid name restrictions in nilearn)
    evs = events.sort_values(sort_field)[events_field].unique()
    n_ev = len(evs)
    ev_inds = np.arange(n_ev)
    ev_names = [f"ev{e:03d}" for e in ev_inds]
    events['ev'] = events[events_field]
    events['ev_index'] = events[events_field].map(dict(zip(evs, ev_names)))

    # create design matrix
    design = create_betaseries_design(
        events, "ev_index", n_vol, tr, time_offset, high_pass=0
    )

    # get corresponding events; keep any fields that are consistent across
    # presentations
    n = events.groupby('ev_index').apply(lambda x: x.nunique()).reset_index(drop=True)
    consistent = (n == 1).all()
    consistent_fields = consistent[consistent].index
    ev_events = (
        events.groupby('ev_index')
        .first()
        .reset_index()
        .get(consistent_fields)
        .drop(columns=['ev', 'ev_index'])
    )

    # create confound matrix
    mat = design.iloc[:, :n_ev].to_numpy()
    confound = np.hstack((design.iloc[:, n_ev:-1].to_numpy(), nuisance))

    # load functional data
    bold_vol = nib.load(func_path)
    mask_vol = nib.load(mask_file)
    bold_img = bold_vol.get_fdata()
    mask_img = mask_vol.get_fdata().astype(bool)
    data = bold_img[mask_img].T

    # estimate betaseries
    beta = estimate_betaseries(data, mat, confound=confound)
    out_data = np.zeros([*mask_img.shape, beta.shape[0]])
    out_data[mask_img, :] = beta.T

    # save betaseries image
    sub_dir = out_dir / f"sub-{subject}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sub-{subject}_task-{task}_run-{run}"
    beta_path = sub_dir / f"{prefix}_space-{space}_label-{mask_name}_betaseries.nii.gz"
    new_img = nib.Nifti1Image(out_data, mask_vol.affine, mask_vol.header)
    nib.save(new_img, beta_path)

    # save mask
    mask_path = sub_dir / f"{prefix}_space-{space}_label-{mask_name}_mask.nii.gz"
    new_img = nib.Nifti1Image(mask_img, mask_vol.affine, mask_vol.header)
    nib.save(new_img, mask_path)

    # save corresponding events
    evs_path = sub_dir / f"{prefix}_desc-events_timeseries.tsv"
    ev_events.to_csv(evs_path, sep="\t", index=False, na_rep="n/a")


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
@click.option("--events-category", type=str, help="Field with event category")
@click.option(
    "--sort-field",
    type=str,
    help="Colon-separated list of fields to sort by for output order",
)
@click.option("--hp-filter", help="Highpass filter in Hz", type=float, default=0)
@click.option("--smooth", help="Smoothing kernel FWHM", type=float)
@click.option("--confound-measures", help="List of confound measures to include")
@click.option(
    "--exclude-motion",
    type=int,
    nargs=2,
    help="Range of frames to exclude around high motion",
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
    hp_filter,
    smooth,
    confound_measures,
    exclude_motion,
):
    data_dir = Path(data_dir)
    fmriprep_dir = Path(fmriprep_dir)
    out_dir = Path(out_dir)
    if sort_field is None:
        sort_field = events_field
    else:
        sort_field = sort_field.split(':')
    if confound_measures is not None:
        confound_measures = confound_measures.split(':')

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
    img = nib.load(func_path)
    n_vol = img.header["dim"][4]
    func_json = func_path.with_suffix("").with_suffix(".json")
    with open(func_json, "r") as f:
        func_param = json.load(f)
    tr = func_param["RepetitionTime"]
    time_offset = func_param["StartTime"]

    # create nuisance regressor matrix
    confounds = pd.read_table(confound_path)
    nuisance = create_confound_matrix(confounds, confound_measures, exclude_motion)

    # run betaseries estimation and save results
    run_betaseries(
        events_path,
        events_field,
        n_vol,
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
        nuisance=nuisance,
        sort_field=sort_field,
    )
