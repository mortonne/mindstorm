#!/usr/bin/env python
#
# Remove runs that are shorter than the others.

import os
import argparse
import tempfile
import subprocess as sub
import numpy as np
import nibabel as nib
from bids import BIDSLayout


def rename_bold(all_files, included_files, dry_run=False):
    """Delete excluded files and rename included files."""
    example = included_files[0]
    parent_dir = os.path.dirname(example)
    subject = example.tags['subject'].value
    if 'session' in example.tags:
        session = example.tags['session'].value
    else:
        session = None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []
        n = 0
        for file in all_files:
            if file not in included_files:
                # delete
                cmd = f'rm {file.path}'
                if dry_run:
                    print(cmd)
                else:
                    sub.run(cmd, shell=True)
            else:
                # copy to temp file
                temp = os.path.join(temp_dir, f'file{n}.nii.gz')
                cmd = f'cp {file.path} {temp}'
                if dry_run:
                    print(cmd)
                else:
                    sub.run(cmd, shell=True)
                n += 1
                temp_files.append(temp)

        for i, (file, temp) in enumerate(zip(included_files, temp_files)):
            # set new file name
            run = i + 1
            task = file.tags['task'].value
            if session is not None:
                prefix = f'sub-{subject}_ses-{session}'
            else:
                prefix = f'sub-{subject}'
            filename = f'{prefix}_task-{task}_run-{run}_bold.nii.gz'

            # copy from the temp file to the new path
            new_file = os.path.join(parent_dir, filename)
            cmd = f'cp {temp} {new_file}'
            if dry_run:
                print(cmd)
            else:
                sub.run(cmd, shell=True)


def main(data_dir, dry_run=False):
    layout = BIDSLayout(data_dir)
    subjects = layout.get_subjects()
    for subject in subjects:
        bold_files = layout.get(
            subject=subject, datatype='func', extension='.nii.gz', suffix='bold'
        )
        n = np.zeros(len(bold_files))
        for i, bold_file in enumerate(bold_files):
            img = nib.load(bold_file)
            n[i] = img.header.get_data_shape()[3]

        max_n = np.max(n)
        if np.all(n == max_n):
            continue

        include = n == max_n
        included_files = [f for f, i in zip(bold_files, include) if i]
        rename_bold(bold_files, included_files, dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', help="Path to raw BIDS data")
    parser.add_argument(
        '--dry-run', action="store_true", help="Print commands without running"
    )
    args = parser.parse_args()
    main(args.data_dir, args.dry_run)
