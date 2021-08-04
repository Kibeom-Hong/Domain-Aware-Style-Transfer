iters=( 70000 )
for value in "${iters[@]}"
do
    CUDA_VISIBLE_DEVICES=1 python main.py \
        --type calculation_time \
        --batch_size 1 \
        --model_type baseline \
        --DA_comment New_DA_Net_v1_mixup \
        --ST_comment test_mixed_3skip_avgpool_kernel8_f5 \
        --is_da_train False \
        --is_st_train False \
        --decoder_trained_epoch 10000 \
        --DA_Net_trained_epoch 98765 \
        --test_content './test_images/content/' \
        --test_p_reference './test_images/origin_p_reference/' \
        --test_a_reference '../pytorch-hed/origin_results/art_styles_real'
        
done