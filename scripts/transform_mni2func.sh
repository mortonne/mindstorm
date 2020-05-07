#!/bin/bash
#
# Transform an image from template space to functional space.

if [[ $# -lt 3 ]]; then
    cat <<EOF
Usage:   transform_mni2func.sh [-n interp] [-a anat] input output subject
Example: transform_mni2func.sh -n NearestNeighbor contrast_group_mask.nii.gz bender_01_contrast_mask.nii.gz bender_01

After running reg_anat2mni.py, can use this to transform any image in
template space (MNI or custom) to functional space.

input
    Path to an image in the space of the template. May be a single volume
    or a timeseries.

output
    Path to an image to be written, in functional space.

subject
    Subject code. Used to look up existing transforms. Must also set the
    STUDYDIR environment variable as the base directory for the study
    before calling.

OPTIONS
-a
    Suffix of anatomical image series that was used as a reference when
    registration was done to the template. Should also be the image used
    as a target for registering the reference functional scan to the
    anatomical. If not specified, no suffix will be used.

-n
    Type of interpolation to use. May be: BSpline (default), Linear,
    NearestNeighbor, or any other types supported by antsApplyTransforms.

EOF
    exit 1
fi

interp=BSpline
refanat=""
while getopts ":a:n:" opt; do
    case $opt in
    a)
        refanat=$OPTARG
        ;;
    n)
        interp=$OPTARG
        ;;
    *)
        echo "Invalid option: $opt"
        exit 1
    esac
done
shift $((OPTIND-1))

input=$1
output=$2
subject=$3

if [[ ! -f $input ]]; then
    echo "Error: input volume missing."
    exit 1
fi

sdir=${STUDYDIR}/${subject}

# check for the functional reference
refvol=$sdir/BOLD/antsreg/data/refvol.nii.gz
if [[ ! -f $refvol ]]; then
    echo "Error: reference volume missing."
    exit 1
fi

# make sure the anat to func transformation is available in old ants format
anat2func=$sdir/anatomy/bbreg/transforms/highres-refvol
if [[ ! -f ${anat2func}.txt ]]; then
    if [[ ! -f ${anat2func}.mat ]]; then
        echo "Error: anatomical to functional registration missing."
        exit 1
    fi

    c3d_affine_tool -ref "$refvol" -src "$sdir/anatomy/orig${refanat}.nii.gz" \
        "${anat2func}.mat" -fsl2ras -oitk "${anat2func}.txt"
fi

# check anat to template transform files
temp2orig_warp=$sdir/anatomy/antsreg/transforms/orig-template_InverseWarp.nii.gz
orig2temp=$sdir/anatomy/antsreg/transforms/orig-template_Affine.txt
if [[ ! -f $temp2orig_warp ]]; then
    echo "Error: Warp file missing."
    exit 1
fi
if [[ ! -f $orig2temp ]]; then
    echo "Error: Affine file missing."
    exit 1
fi

# run transformation
ntp=$(fslval "$input" dim4)
if [[ $ntp -gt 1 ]]; then
    antsApplyTransforms -d 3 -e 3 -i "$input" -o "$output" -r "$refvol" \
        -n "$interp" -t "${anat2func}.txt" -t "[${orig2temp},1]" \
        -t "$temp2orig_warp"
else
    antsApplyTransforms -d 3 -i "$input" -o "$output" -r "$refvol" \
        -n "$interp" -t "${anat2func}.txt" -t "[${orig2temp},1]" \
        -t "$temp2orig_warp"
fi
