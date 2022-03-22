python main.py --model="GATGeometric" --dataset="PPI" --attention_heads 4 4 6 --num_layers=3 --num_hidden_features=256\
  --dropout_val=0.0 --concat_layers 1 1 0 --verbose --use_early_stopping --num_runs=10 --learning_rate=.01 --batch_size=1 \
  --l2_lambda=0.0