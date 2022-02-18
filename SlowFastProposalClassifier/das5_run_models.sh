

REPO=/var/scratch/pbagad/projects/epic-kitchens-ssl/

conda activate slowfast
export PYTHONPATH=$REPO


echo "TCLR"
cfg=$REPO/configs/EPIC-KITCHENS/TCLR/tclr_32x112x112_R18_K400_LR0.0025_no_norm.yaml
bash das5_eval_action_detection.sh -c $cfg

echo "GDT"
cfg=$REPO/configs/EPIC-KITCHENS/GDT/das5_32x112x112_R2+1D_K400_LR0.0025.yaml
bash das5_eval_action_detection.sh -c $cfg

echo "SeLaVi"
cfg=$REPO/configs/EPIC-KITCHENS/SELAVI/selavi_32x112x112_R18_K400_LR0.0025.yaml
bash das5_eval_action_detection.sh -c $cfg

echo "Scratch"
cfg=$REPO/configs/EPIC-KITCHENS/R2PLUS1D/32x112x112_R18_scratch_LR0.0025.yaml
bash das5_eval_action_detection.sh -c $cfg

