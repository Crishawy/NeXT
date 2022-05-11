SCENE=$1
DATA_ROOT=./data
MODEL=$2
TRAIN_DIR=./logs/multiblender/$MODEL/$SCENE

echo $SCENE
echo $MODEL

python train.py \
--train_dir $TRAIN_DIR \
--data_dir $DATA_ROOT/nerf_synthetic_multiscale/$SCENE \
--gin_file ./configs/${MODEL}_multiblender.gin \
--decay_type cosine \
--chunk 4096

