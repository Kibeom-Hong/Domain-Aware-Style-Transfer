CUDA_VISIBLE_DEVICES=0 python main.py \
    --type interpolate \
    --batch_size 1 \
    --imsize 512 \
    --cropsize 512 \
    --cencrop \
    --model_type baseline \
    --DA_comment New_DA_Net_v1_mixup \
    --ST_comment test_mixed_3skip_avgpool_kernel8_f7_weak_gan_loss_MultipleD_t2 \
    --is_da_train False \
    --is_st_train False \
    --decoder_trained_epoch 47500 \
    --DA_Net_trained_epoch 98765 \
    --test_content './test_images/sample_content/' \
    --test_p_reference './test_images/p_reference_last/' \
    --test_a_reference './test_images/sample_a_reference/'
#--DA_comment test_LL_shallow_beta_linear_11_lastlayers \