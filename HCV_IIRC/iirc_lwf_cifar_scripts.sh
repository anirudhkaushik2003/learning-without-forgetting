python ./prior_main_ptm_training_icarl.py \
    --method learning_without_forgetting_ptm \
    --n_memories_per_class -1 \
    --group lwf_HCV_experiments \
    --save_each_task_model \
    --epochs_per_task 14 \
    --total_n_memories -1 \
    --use_best_model \
    --checkpoint_interval 5 \
    --wandb_project HCV_LwF_HCV
