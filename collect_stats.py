import os
import json
import pandas as pd

def main():
    results_root = "results"
    results_subfolders = [
        "fastopic_paraphrase-multilingual-MiniLM-L12-v2",
        "fastopic_paraphrase-multilingual-MiniLM-L12-v2_norm",
    ]

    # Create empty dataframe to store results
    columns = [
        "embedder",
        "model",
        "topics",
        "docs",
        "vocab",
        "coherence",
        "diversity",
        "purity",
        "nmi",
        "accuracy",
        "f1-score",
    ]
    results_df = pd.DataFrame(columns=columns)

    for subfolder in results_subfolders:
        subfolder_path = os.path.join(results_root, subfolder)

        embedder = subfolder.split("_")[1]
        if "norm" in subfolder:
            embedder += "_norm"

        for model_path in os.listdir(subfolder_path):
            model, topics, docs, vocab = model_path.split("_")

            results_path = os.path.join(subfolder_path, model_path, "results.json")
            results = json.load(open(results_path, "r"))

            new_row = {
                "embedder": embedder,
                "model": model,
                "topics": topics,
                "docs": docs,
                "vocab": vocab,
                "coherence": results["coherence"],
                "diversity": results["topic_diversity"],
                "purity": results["purity"],
                "nmi": results["nmi"],
                "accuracy": results["accuracy"],
                "f1-score": results["f1-score"],
            }

            results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
    
    results_df.to_csv("results/combined_results.csv", index=False)


if __name__ == "__main__":
    main()