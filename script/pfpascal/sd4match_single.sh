cd ../..
accelerate launch --multi_gpu --num_processes=3 --mixed_precision=fp16 train_simple_prompt.py \
  --dataset pfpascal \
  --config_file config/learnedToken.py \
  --prompt_option single \
  --learnable_seq_length 75 \
  --learn_hidden_state f \
  --epochs=90 \
  --batch_size=3 \
  --init_lr=0.01 \
  --scheduler="constant" \
  --num_workers 4