#!bin/bash

vocab_sizes=(20000 30000 40000 50000)
nums_docs=(15000 20000 25000 30000 35000 40000)
nums_topics=(75 100 125 150 175 200)

exit

python main.py --cache_dir models/ --save_path models/fastopic_model.zip --docs_path data/splits_segment0.jsonl --embes_path data/embeddings_segment0.h5 --batch_size 16000 --vocal_size 20000 --num_docs 15000 --num_topics 75
