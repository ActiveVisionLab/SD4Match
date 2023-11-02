cd ../..
accelerate launch --multi_gpu --num_processes=3 --mixed_precision=fp16 train_cpm.py \
  --dataset pfpascal \
  --config_file config/learnedToken.py \
  --captioner_config Pair-DINO-Feat-G25-C50 \
  --epochs=90 \
  --batch_size=3 \
  --init_lr=0.01 \
  --scheduler="constant" \
  --num_workers 4