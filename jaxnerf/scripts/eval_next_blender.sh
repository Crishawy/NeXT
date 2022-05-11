SCENE=$1
DATA_ROOT=./data
MODEL=next
BACKBONE=$2
TRAIN_DIR=./logs/blender/$MODEL/$BACKBONE/$SCENE

python eval.py \
--train_dir $TRAIN_DIR \
--data_dir $DATA_ROOT/nerf_synthetic/$SCENE \
--dataset blender \
--batching single_image \
--factor 0 \
--num_coarse_samples 128 \
--num_fine_samples 128 \
--use_viewdirs true \
--white_bkgd true \
--randomized true \
--model $MODEL \
--backbone $BACKBONE \
--decay_type cosine \
--warmup_init_lr 0.000001 \
--lr_delay_steps 2500 \
--batch_size 4096 \
--chunk 1024 \
