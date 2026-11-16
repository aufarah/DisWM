export log_directory="/data_all/aufar/diswm/DisWM/logdir"
export dataset_directory="/data_all/aufar/diswm/DisWM/datadir/Cheetah_Run"
export WANDB_MODE=disabled

# For training Video Pred
python dreamer.py --configs defaults dmc2gym \
    --device 'cuda:0' --task dmc2gym_cheetah_run \
    --logdir $log_directory \
    --seed 0 --beta_vae_pretrain True --beta_vae True \
    --pretrain_datasets_path $dataset_directory

# For training WM
python dreamer.py --device 'cuda:0' --task dmc2gym_walker_walk \
    --logdir $log_directory \
    --pretrain_checkpoint_path ./checkpoints/dmc2gym_cheetah_run \
    --seed 0 --configs defaults dmc2gym --traverse True --beta_vae True \
    --method_name 'diswm' --cross_domain False \
    --distillation True --color_distractor True