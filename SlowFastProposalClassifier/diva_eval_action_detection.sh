# Evaluates action detection on EPIC based on BMN proposals
# for a given config (VSSL model).

# Usage: ./diva_eval_action_detection.sh -c <config_file>

parent="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $parent


# common variables
NUM_GPUS=4
batch_size=16
DATA_DIR=/ssd/pbagad/datasets/EPIC-KITCHENS-100/EPIC-KITCHENS/
ANNO_DIR=/ssd/pbagad/datasets/EPIC-KITCHENS-100/annotations/


# get inputs from the user
while getopts "c:" OPTION; do
    case $OPTION in
        c) cfg=$OPTARG;;
        *) exit 1 ;;
    esac
done

# check cfg is given
if [ "$cfg" ==  "" ];then
       echo "cfg is a required argument; Please use -c <relative path to config> to pass config file."
       echo "You can choose configs from:"
       ls $repo/configs/*
       exit
fi

echo "::::::::::::::: Running evaluation for $cfg :::::::::::::::"

# cfg=/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml


# configure output paths
expt_folder="${cfg%.yaml}"
IFS='/' read -r -a array <<< $expt_folder
expt_folder="${array[-2]}--${array[-1]}"
out_base_dir=/home/pbagad/expts/epic-kitchens-ssl/$expt_folder/
# mkdir -p $output_dir
logs_dir=$out_base_dir/logs/
# mkdir -p $logs_dir

# out_base_dir=/home/pbagad/expts/epic-kitchens-ssl/R2PLUS1D--32x112x112_R18_K400_LR0.0025/
ckpt_name=checkpoint_best.pyth
ckpt_path=$out_base_dir/checkpoints/$ckpt_name
outdir=$out_base_dir/bmn-proposal-classification-scores/
echo "Saving logs at $outdir"
mkdir -p $outdir


echo ":: Config file : $cfg"
echo ":: Output dir : $outdir"
echo ":: Checkpoint : $ckpt_path"
echo ":: Logs dir : $logs_dir"

if [ ! -f $ckpt_path ]; then
    echo ":: Checkpoint file not found at $ckpt_path"
    exit
else
    echo ":: Checkpoint file found at $ckpt_path"
fi

# run classifier on proposals
python run_net.py \
    --cfg $cfg \
    NUM_GPUS $NUM_GPUS \
    OUTPUT_DIR $outdir \
    EPICKITCHENS.VISUAL_DATA_DIR $DATA_DIR \
    EPICKITCHENS.ANNOTATIONS_DIR $ANNO_DIR \
    TRAIN.ENABLE False \
    TEST.ENABLE True \
    TEST.CHECKPOINT_FILE_PATH $ckpt_path \
    EPICKITCHENS.TEST_LIST /home/pbagad/projects/C2-Action-Detection-ssl/BMNProposalGenerator/output/ek100/result_proposal-validation.pkl \
    EPICKITCHENS.TEST_SPLIT validation \
    TEST.BATCH_SIZE $batch_size > $logs_dir/bmn_proposal_classifier_log.txt

# compute final metrics
python ../EvaluationCode/evaluate_detection_json_ek100.py  $outdir/action_detection_baseline_validation.json $ANNO_DIR/EPIC_100_validation.pkl > $outdir/metrics.txt