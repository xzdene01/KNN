#!bin/bash

# If you want to only test models, rename --save_path to --load_path and remove --log_path

nums_topics=(20 200)
nums_docs=(10000 20000)
vocab_sizes=(40000 0) # Setting this to 0 will use the whole available vocabulary

norm="" # For non-normalized embeddings, set to "" (empty string) otherwise set to "_norm"

embe_model="paraphrase-multilingual-MiniLM-L12-v2"
model_root="models/fastopic_${embe_model}${norm}"
log_root="logs/fastopic_${embe_model}${norm}"
eval_root="results/fastopic_${embe_model}${norm}"

if [ -n "$norm" ]; then
    normalize="--norm_embes"
fi

for num_topics in "${nums_topics[@]}"; do
    for num_docs in "${nums_docs[@]}"; do
        for vocab_size in "${vocab_sizes[@]}"; do
            model_path="${model_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}.zip"
            log_path="${log_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}.log"
            eval_dir="${eval_root}/fastopic_${num_topics}_${num_docs}_${vocab_size}"

            if [ "$vocab_size" -eq 0 ]; then
                vocab_cmd=""
            else
                vocab_cmd="--vocab_size $vocab_size"
            fi

            echo "========================================="
            echo "Running with num_topics=${num_topics}, num_docs=${num_docs}, vocab_size=${vocab_size}"
            echo "========================================="
            
            python main.py \
                --cache_dir models/ \
                --save_path "$model_path" \
                --docs_path data/splits_reduced.jsonl \
                --embes_path data/splits_reduced_${embe_model}${norm}.h5 \
                --test_docs_path data/reduced_dataset.csv \
                --test_embes_path data/reduced_dataset_${embe_model}${norm}.h5 \
                --embe_model "$embe_model" \
                --debug --batch_size 1000 --seed 42 \
                --num_topics "$num_topics" --num_docs "$num_docs" ${vocab_cmd} \
                --log_path "$log_path" --eval_dir "$eval_dir" --epochs 200 ${normalize}
        done
    done
done
