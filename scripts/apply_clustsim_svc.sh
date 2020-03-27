#!/bin/bash
#
# Run small volume correction on randomise permutation results.

if [[ $# -lt 3 ]]; then
    echo "Run small volume correction on randomise permutation results."
    echo
    echo "Usage:   apply_clustsim_svc.sh [-o] [-t template] filepath model maskname"
    echo "Example: apply_clustsim_svc.sh mvpa/cat_react_item_rt study_stim b_par"
    echo
    echo "Setting -o will overwrite existing results."
    echo
    echo "If specified, a link to the template image will be placed as"
    echo "bg_image.nii.gz."
    echo
    exit 1
fi

overwrite=false
template=""
while getopts ":ot:" opt; do
    case $opt in
        o)
            overwrite=true
            ;;
        t)
            template="$OPTARG"
            ;;
        *)
            echo "Invalid option: $opt"
            exit 1
    esac
done
shift $((OPTIND-1))

filepath=$1
model=$2
maskname=$3

# fixed voxelwise alpha because we just have a thresholded image, not
# a zscore image
alpha=0.01

clustsimdir=$STUDYDIR/batch/glm/$model/$maskname
if [[ ! -d $clustsimdir ]]; then
    echo "ClustSim dir does not exist: $clustsimdir"
    exit 1
fi

statdir=$STUDYDIR/batch/$filepath
if [[ ! -d $statdir ]]; then
    echo "Statistics image dir does not exist: $statdir"
    exit 1
fi
echo "Processing files in: $statdir"

resdir=$statdir/$maskname
mkdir -p "$resdir"
cd "$resdir" || exit

# get cluster extent based on the existing clustsim results, based on
# the residuals from the betaseries model that the searchlight was
# based on
cfile=$clustsimdir/clustsim.NN3_1sided.1D
clust_extent=$(grep "^ $alpha" < "$cfile" | awk '{ print $3 }')
echo "Minimum cluster extent: $clust_extent"
echo "$clust_extent" > clust_thresh

recalc=false
if [[ $overwrite = true ]]; then
    recalc=true
fi
if [[ $(imtest "$statdir/zstat_vox_p_tstat1") = 1 && $(imtest "$statdir/stat_thresh") = 0 ]]; then
   recalc=true
fi

if [[ $recalc = true ]]; then
    # have results from randomise, which have not been
    # thresholded. Apply the voxelwise alpha
    echo "Recalculating thresholded images..."

    # significant voxels
    thresh=$(python -c "print(1-$alpha)")
    fslmaths "$statdir/zstat_vox_p_tstat1" -thr "$thresh" -bin "$statdir/vox_mask"

    # corresponding z-stat image (for display)
    fslmaths "$statdir/zstat_vox_p_tstat1" -mul -1 -add 1 -ptoz "$statdir/zstat1"

    # thresholded z-stat
    fslmaths "$statdir/zstat1" -mas "$statdir/vox_mask" "$statdir/stat_thresh"

    # pull a copy of the template image
    if [[ $(imtest "$statdir/bg_image") = 0 && -n $template ]]; then
        ln -sf "$template" "$statdir/bg_image.nii.gz"
    fi

    # get uncorrected clusters of at least minimal size (useful for
    # defining masks, getting cluster size)
    "$FSLDIR/bin/cluster" -i "$statdir/vox_mask" -t 0.5 --minextent=100 \
        -o "$statdir/cluster_mask100" > "$statdir/cluster100.txt"
fi

imcp "$clustsimdir/mask" mask
imcp "$statdir/stat_thresh" cope1
fslmaths cope1 -mas mask thresh_cope1

# report corrected clusters
"$FSLDIR/bin/cluster" -i thresh_cope1 -c cope1 -t 0.0001 \
    --minextent="$clust_extent" --othresh=cluster_thresh_cope1 \
    -o cluster_mask_cope1 --connectivity=26 --mm \
    --olmax=lmax_cope1_std.txt --scalarname=Z > cluster_cope1_std.txt

range=$(fslstats cluster_thresh_cope1 -l 0.0001 -R 2>/dev/null)
low=$(echo "$range" | awk '{print $1}')
high=$(echo "$range" | awk '{print $2}')
echo "Rendering using zmin=$low zmax=$high"

imcp "$statdir/bg_image" example_func
overlay 1 0 example_func -a cluster_thresh_cope1 "$low" "$high" rendered_thresh_cope1
slicer rendered_thresh_cope1 -S 2 750 rendered_thresh_cope1.png
