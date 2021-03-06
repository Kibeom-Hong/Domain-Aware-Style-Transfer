CUDA_VISIBLE_DEVICES=0 python main.py \
    --type train \
    --batch_size 8 \
    --imsize 256 \
    --cropsize 256 \
    --cencrop \
    --lr 1e-4 \
    --max_iter 100001 \
    --model_type baseline \
    --DA_comment Indicator_Test \
    --ST_comment Decoder_Test \
    --check_iter 100 \
    --is_da_train False \
    --is_st_train True \
    --DA_Net_trained_epoch 98765 \
    --num_workers 20