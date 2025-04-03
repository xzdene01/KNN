#!bin/bash

nums_topics=(75 100 150)
nums_docs=(15000 20000 30000)
vocab_sizes=(20000 40000)

embe_model="paraphrase-multilingual-MiniLM-L12-v2"
model_root="models/fastopic_${embe_model}"
log_root="logs/fastopic_${embe_model}"
eval_root="results/fastopic_${embe_model}"

for num_topics in "${nums_topics[@]}"; do
    for num_docs in "${nums_docs[@]}"; do
        for vocab_size in "${vocab_sizes[@]}"; do
            model_path="${model_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}.zip"
            log_path="${log_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}.log"
            eval_dir="${eval_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}"

            echo "========================================="
            echo "Running with num_topics=${num_topics}, num_docs=${num_docs}, vocab_size=${vocab_size}"
            echo "========================================="
            
            python main.py \
                --cache_dir models/ \
                --save_path "$model_path" \
                --docs_path data/splits_reduced.jsonl \
                --embes_path data/splits_reduced_${embe_model}.h5 \
                --embe_model "$embe_model" \
                --debug --batch_size 2000 --seed 42 \
                --num_topics "$num_topics" --num_docs "$num_docs" --vocab_size "$vocab_size" \
                --log_path "$log_path" --eval_dir "$eval_dir" --epochs 200
        done
    done
done