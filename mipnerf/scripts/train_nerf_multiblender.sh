SCENE=$1
DATA_ROOT=./data
TRAIN_DIR=./logs/multiblender/nerf/$SCENE

python train.py \
--train_dir $TRAIN_DIR \
--data_dir $DATA_ROOT/nerf_synthetic_multiscale/$SCENE \
--gin_file ./configs/multiblender.gin

