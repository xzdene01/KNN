#!bin/bash

nums_topics=(75 100 125 150 175 200)
nums_docs=(15000 20000 25000 30000 35000 40000)
vocab_sizes=(20000 30000 40000 50000)

python main.py \
    --cache_dir models/ \
    --save_path models/fastopic_all-MiniLM-L6-v2_10k.zip \
    --docs_path data/splits_reduced.jsonl \
    --embes_path data/splits_reduced_all-MiniLM-L6-v2 \
    --debug --batch_size 2000 --seed 42 \
    --num_topics 75 --num_docs 10000 --vocab_size 20000