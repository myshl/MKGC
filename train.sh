python main.py -dataset DB15K \
                -batch_size 112 \
                -pretrained_model pretrained_model_path \
                -lr 5e-4 \
                -prompt_length 10 \
                -label_smoothing 0.1 \
                -embed_dim 256 \
                -k_w 16 \
                -k_h 16 \
                -epoch 30

                