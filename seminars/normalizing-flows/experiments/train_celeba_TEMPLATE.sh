export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=./experiments/model
DATASET_PATH=./datasets/celeba/celeba_valid.tfrecords
cd ..

python ./train_celeba.py \
    --batch_size 16 \
    --image_size 24 \
    --train_steps 100000 \
    --eval_steps 100 \
    --dataset_path  $DATASET_PATH \
    --model_dir $MODEL_PATH \
    --l2_reg 0.0001 \
    --sample_beta 1.0 \
    --save_secs 100 \
    --lr 0.005 \
    --decay_steps 20000 \
    --decay_rate 0.5 \
    --units_factor 4 \
    --units_width 0 \
    --num_blocks 2 \
    --skip_connection True \
    --num_steps 8 \
    --num_scales 3 \
    --selu_reg 0.0001 \
    --mode train