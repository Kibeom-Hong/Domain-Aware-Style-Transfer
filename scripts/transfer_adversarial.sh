CUDA_VISIBLE_DEVICES=0 python main.py \
    --type transfer \
    --batch_size 1 \
    --model_type baseline \
    --DA_comment StyleIndicator \
    --ST_comment Decoder_adversarial \
    --is_da_train False \
    --is_st_train False \
    --test_content './test_images/content/' \
    --test_p_reference './test_images/origin_p_reference/' \
    --test_a_reference './test_images/origin_a_reference/'
        
