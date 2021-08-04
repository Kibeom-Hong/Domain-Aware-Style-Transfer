CUDA_VISIBLE_DEVICES=0 python main.py \
    --type train \
    --batch_size 8 \
    --imsize 256 \
    --cropsize 256 \
    --cencrop \
    --lr 1e-4 \
    --max_iter 100001 \
    --model_type baseline \
    --DA_comment New_DA_Net_v1_mixup \
    --ST_comment test_mixed_3skip_avgpool_kernel8_f7_weak_gan_loss_MultipleD_t2_test\
    --check_iter 100 \
    --is_da_train False \
    --is_st_train True \
    --DA_Net_trained_epoch 98765 \
    --num_workers 20

    #test_mixed_3skip_avgpool_kernel8_f6_weak_gan_loss