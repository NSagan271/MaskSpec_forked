DATA_PATH_TRAIN='./batlab_audio/data/batlab_data_train_mp3.hdf'
DATA_PATH_VAL='./batlab_audio/data/batlab_data_eval_mp3.hdf'
DATA_PATH_TEST='./batlab_audio/data/batlab_data_test_mp3.hdf'
OUTPUT_DIR='./batlab_audio/output_dir'
LOG_DIR='./batlab_audio/log_dir'
NORM_FILE='./batlab_audio/mean_std_128.npy'

python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --use_env batlab_audio/run.py \
    --batch_size 32 \
    --data_path_train ${DATA_PATH_TRAIN} \
    --data_path_val ${DATA_PATH_VAL} \
    --data_path_test ${DATA_PATH_TEST} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --norm_file ${NORM_FILE} \
    --device cuda
