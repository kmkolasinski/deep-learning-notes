export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=./experiments/model
DATASET_PATH=./datasets/celeba/celeba_valid.tfrecords
cd ..

python ./train_celeba.py \
    --batch_size 16 \
    --image_size 24 \
    --train_steps 20000 \
    --eval_steps 100 \
    --dataset_path  $DATASET_PATH \
    --model_dir $MODEL_PATH \
    --l2_reg 0.0001 \
    --sample_beta 0.9 \
    --save_secs 100 \
    --lr 0.005 \
    --decay_steps 5000 \
    --decay_rate 0.25 \
    --width 64 \
    --num_bits 5 \
    --num_steps 8 \
    --num_scales 3 \
    --mode train