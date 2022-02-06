### Evaluates given model checkpoint on the generated proposals to classify actions

# common variables
NUM_GPUS=4
DATA_DIR=/ssd/pbagad/datasets/EPIC-KITCHENS-100/EPIC-KITCHENS/
ANNO_DIR=/ssd/pbagad/datasets/EPIC-KITCHENS-100/annotations/
batch_size=16


# echo "Evaluating for AVID-CMA"
# # setup
# cfg=/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/AVID_CMA/32x112x112_R18_K400_LR0.0025.yaml
# out_base_dir=/home/pbagad/expts/epic-kitchens-ssl/AVID_CMA--32x112x112_R18_K400_LR0.0025/
# ckpt_name=checkpoint_best.pyth
# ckpt_path=$out_base_dir/checkpoints/$ckpt_name
# outdir=$out_base_dir/bmn-proposal-classification-scores/
# echo "Saving logs at $outdir"
# mkdir -p $outdir
# # run classifier on proposals
# python run_net.py \
#     --cfg $cfg \
#     NUM_GPUS $NUM_GPUS \
#     OUTPUT_DIR $outdir \
#     EPICKITCHENS.VISUAL_DATA_DIR $DATA_DIR \
#     EPICKITCHENS.ANNOTATIONS_DIR $ANNO_DIR \
#     TRAIN.ENABLE False \
#     TEST.ENABLE True \
#     TEST.CHECKPOINT_FILE_PATH $ckpt_path \
#     EPICKITCHENS.TEST_LIST /home/pbagad/projects/C2-Action-Detection-ssl/BMNProposalGenerator/output/ek100/result_proposal-validation.pkl \
#     EPICKITCHENS.TEST_SPLIT validation \
#     TEST.BATCH_SIZE $batch_size > $outdir/log.txt
# # compute final metrics
# python ../EvaluationCode/evaluate_detection_json_ek100.py  $outdir/action_detection_baseline_validation.json $ANNO_DIR/EPIC_100_validation.pkl > $outdir/metrics.txt


echo "Evaluating for R2+1D"
cfg=/home/pbagad/projects/epic-kitchens-ssl/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml
out_base_dir=/home/pbagad/expts/epic-kitchens-ssl/R2PLUS1D--32x112x112_R18_K400_LR0.0025/
ckpt_name=checkpoint_best.pyth
ckpt_path=$out_base_dir/checkpoints/$ckpt_name
outdir=$out_base_dir/bmn-proposal-classification-scores/
echo "Saving logs at $outdir"
mkdir -p $outdir
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
    TEST.BATCH_SIZE $batch_size > $outdir/log.txt
# compute final metrics
python ../EvaluationCode/evaluate_detection_json_ek100.py  $outdir/action_detection_baseline_validation.json $ANNO_DIR/EPIC_100_validation.pkl > $outdir/metrics.txt