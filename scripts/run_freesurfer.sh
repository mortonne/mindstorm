#!/bin/bash
#
# Streamline running FreeSurfer recon-all.

if [[ $# -lt 1 ]]; then
    cat <<EOF
Usage: run_freesurfer.sh [-cf] [-n threads] [-o outname] subject

Use highres anatomical image to run FreeSurfer standard reconstruction.
Highres is expected to be in $STUDYDIR/$subject/anatomy/highres.nii.gz.

Results are saved in $STUDYDIR/$subject/anatomy/$outname.

OPTIONS
-c
    Use a T2 coronal image to estimate hippocampal subfields. Coronal
    image expected to be in $STUDYDIR/$subject/anatomy/coronal.nii.gz.

-f
    Use a FLAIR image to refine the pial surface. Expected to be in
    $STUDYDIR/$subject/anatomy/coronal.nii.gz.

-n threads
    Number of threads to use for running recon-all.

-o outname
    Set the name of the output directory. Default is $subject.

EOF
    exit 1
fi

if [[ -u $STUDYDIR ]]; then
    echo "STUDYDIR unset; quitting."
    exit 1
fi

if [[ ! -d $STUDYDIR ]]; then
    echo "STUDYDIR does not exist; quitting."
    exit 1
fi

coronal=false
flair=false
while getopts ":o:n:cf:b" opt; do
    case $opt in
        o)
            outname=$OPTARG
            ;;
        n)
            threads=$OPTARG
            ;;
        c)
            coronal=true
            ;;
        f)
            flair=true
            ;;
        *)
            echo "Invalid option: $opt"
            exit 1
            ;;
    esac
done
shift $((OPTIND-1))

subject=$1
subjdir=$STUDYDIR/$subject

if [[ -z $outname ]]; then
    outname=$subject
fi

if [[ ! -f ${subjdir}/anatomy/highres.nii.gz ]]; then
    echo "ERROR: Highres file not found."
    exit 1
fi

# delete existing freesurfer results
if [[ -d ${subjdir}/anatomy/$outname ]]; then
    cd "$subjdir/anatomy" || exit 1
    rm -rf "$outname"
fi

source "$FREESURFER_HOME/SetUpFreeSurfer.sh"

opt=""
if [[ $flair = true ]]; then
    if [[ ! -e $subjdir/anatomy/flair.nii.gz ]]; then
        echo "Error: flair image not found."
        exit 1
    fi
    opt="$opt -FLAIR $subjdir/anatomy/flair.nii.gz -FLAIRpial"
fi

if [[ $coronal = true ]]; then
    if [[ ! -e $subjdir/anatomy/coronal.nii.gz ]]; then
        echo "Error: coronal image not found."
        exit 1
    fi
fi

recon-all \
    -s "$outname" \
    -sd "$subjdir/anatomy/" \
    -i "$subjdir/anatomy/highres.nii.gz" \
    -all -parallel -threads "$threads" $opt

if [[ $coronal = true ]]; then
    segmentHA_T2.sh \
        "$subject" \
        "$subjdir/anatomy/coronal.nii.gz" \
        T2 1 "$subjdir/anatomy"
fi
