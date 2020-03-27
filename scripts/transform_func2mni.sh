#!/bin/bash

if [ $# -lt 3 ]; then
    cat <<EOF
Usage:   transform_func2mni.sh [-a anat] [-n interp] [-p postmask] input output subject
Example: transform_func2mni.sh bender_01_mat_item_w2v_faces.nii.gz bender_01_mat_item_w2v_faces_mni.nii.gz bender_01

After running reg_anat2mni.py, can use this to transform any image
in functional space to template space (MNI or custom).

input
    Path to an image in the space of the reference functional scan.
    May be a single volume or a timeseries.

output
    Filename for image to be written in template space.

subject
    Subject code. Used to look up existing transforms. Must also
    set the STUDYDIR environment variable as the base directory
    for the study before calling.

OPTIONS
-a anat
    Suffix of anatomical image series that was used as reference
    When registration was done to the template. Should also be the
    image used as a target for registering the reference functional
    scan to the anatomical. If not specified, no suffix will be
    used.

-n interp
    Type of interpolation to use. May be: Linear (default), BSpline,
    NearestNeighbor, or any other types supported by antsApplyTransforms.

-p postmask
    Post-mask to apply to the transformed image. This is useful
    to remove very small values outside the brain that occur when
    using B-spline interpolation.

EOF
    exit 1
fi

refanat=""
interp=Linear
postmask=""
while getopts ":a:mn:p:" opt; do
    case $opt in
	a)
	    refanat=$OPTARG
	    ;;
	n)
	    interp=$OPTARG
	    ;;
	p)
	    postmask=$OPTARG
	    ;;
    esac
done
shift $((OPTIND-1))

input=$1
output=$2
subject=$3

if [ ! -f $input ]; then
    echo "Error: input volume missing."
    exit 1
fi

sdir=${STUDYDIR}/${subject}
anat2func=$sdir/anatomy/bbreg/transforms/highres-refvol
refvol=$sdir/BOLD/antsreg/data/refvol.nii.gz
if [ ! -f $refvol ]; then
    echo "Error: reference volume missing."
    exit 1
fi

# after reg_anat2mni.py, images in this directory will be the in the
# space of the template, regardless of the template used
reference=$sdir/anatomy/antsreg/data/orig.nii.gz
if [ ! -f $reference ]; then
    echo "Error: anatomical reference image missing."
    exit 1
fi

# make sure anat to func transformation is available in old ants format
if [ ! -f ${anat2func}.txt ]; then
    if [ ! -f ${anat2func}.mat ]; then
	echo "Error: anatomical to functional registration missing."
	exit 1
    fi

    c3d_affine_tool -ref $refvol -src $sdir/anatomy/orig${refanat}.nii.gz ${anat2func}.mat -fsl2ras -oitk ${anat2func}.txt
fi

# transform input from functional to template space
orig2temp_warp=$sdir/anatomy/antsreg/transforms/orig-template_Warp.nii.gz
orig2temp=$sdir/anatomy/antsreg/transforms/orig-template_Affine.txt
if [ ! -f $orig2temp_warp ]; then
    echo "Error: Warp file missing."
    exit 1
fi
if [ ! -f $orig2temp ]; then
    echo "Error: Affine file missing."
    exit 1
fi

ntp=$(fslval $input dim4)
if [ $ntp -gt 1 ]; then
    antsApplyTransforms -d 3 -e 3 -i $input -o $output -r $reference -n $interp -t $orig2temp_warp -t $orig2temp -t [${anat2func}.txt,1]
else
    antsApplyTransforms -d 3 -i $input -o $output -r $reference -n $interp -t $orig2temp_warp -t $orig2temp -t [${anat2func}.txt,1]
fi

# mask voxels outside the template
if [ -n "$postmask" ]; then
    if [ ! -f $postmask ]; then
	echo "Error: mask not found: $postmask"
	exit 1
    fi
    fslmaths $output -mas $postmask $output
fi
