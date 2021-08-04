iters=( 1 )

for value in "${iters[@]}"
do
    CUDA_VISIBLE_DEVICES=0 python main.py \
        --train_type 'seg' \
        --type transfer \
        --batch_size 1 \
        --model_type baseline \
        --DA_comment New_DA_Net_v1_mixup \
        --ST_comment test_mixed_3skip_avgpool_kernel8_f7_weak_gan_loss_MultipleD_t2 \
        --is_da_train False \
        --is_st_train False \
        --decoder_trained_epoch 47500 \
        --DA_Net_trained_epoch 98765 \
        --test_content '../WCT2/examples/origin_content/' \
        --test_p_reference '../WCT2/examples/origin_p_reference/' \
        --test_content_segment '../WCT2/examples/origin_content_segment/' \
        --test_p_reference_segment '../WCT2/examples/origin_p_reference_segment/' \
        --test_a_reference '../WCT2/examples/multiple_seg_a/' \
        
done