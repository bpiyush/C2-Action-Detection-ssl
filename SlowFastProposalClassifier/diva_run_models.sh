
REPO=/home/pbagad/projects/epic-kitchens-ssl/

conda activate slowfast
export PYTHONPATH=$REPO

# echo "R2Plus1D"
# cfg=$REPO/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_K400_LR0.0025.yaml
# bash diva_eval_action_detection.sh -c $cfg

# echo "AVID-CMA"
# cfg=$REP/configs/AVID_CMA/32x112x112_R18_K400_LR0.0025.yaml
# bash diva_eval_action_detection.sh -c $cfg

echo "CTP"
cfg=$REPO/configs/EPIC-KITCHENS/CTP/32x112x112_R2+1D-18_K400_LR0.0025.yaml
bash diva_eval_action_detection.sh -c $cfg

echo "RSPNet"
cfg=$REPO/configs/EPIC-KITCHENS/RSPNET/32x112x112_R18_K400_LR0.0025.yaml
bash diva_eval_action_detection.sh -c $cfg

echo "VideoMoCo"
cfg=$REPO/configs/EPIC-KITCHENS/VIDEOMOCO/32x112x112_VMOCO_R18_K400_LR0.0025.yaml
bash diva_eval_action_detection.sh -c $cfg
