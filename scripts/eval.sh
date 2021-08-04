CUDA_VISIBLE_DEVICES=1 python main.py \
    --type eval \
    --batch_size 3 \
    --imsize 256 \
    --cropsize 256 \
    --cencrop \
    --model_type baseline \
    --DA_comment test_LL_deep_linear_lastlayers \
    --ST_comment test_t4_mixed_3skip_26500 \
    --is_da_train False \
    --is_st_train False \

#--DA_comment test_LL_shallow_beta_linear_11_lastlayers \