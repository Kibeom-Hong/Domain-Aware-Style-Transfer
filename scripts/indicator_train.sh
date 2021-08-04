CUDA_VISIBLE_DEVICES=0 python main.py \
    --type train \
    --batch_size 6 \
    --imsize 512 \
    --cropsize 512 \
    --lr 1e-4 \
    --max_iter 100001 \
    --model_type baseline \
    --DA_comment New_DA_Net_v1_no_mixup_no_dlow \
    --ST_comment test\
    --check_iter 250 \
    --is_da_train True \
    --is_st_train False \



#--DA_comment test_LL_shallow_beta_linear_11_lastlayers \