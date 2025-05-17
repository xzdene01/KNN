# Set cache dir for (transformer) models
import os
os.environ["HF_HOME"] = "./models"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json
import torch
import logging

from wrappers.fastopic_wrapper import FASTopicWrapper
from wrappers.lda_wrapper import LDAWrapper
from wrappers.eval_wrapper import *
from utils.argparser import get_args, get_log_level
from utils.seed_everything import seed_everything

# This will allow to load model even with new PyTorch; this is required by torchvision that is required by topmost :(
# This is a workaround that sets the 'weights_only' option of 'torch.load' to False
# !!! Through this workaround can be downloaded even unsafe model, use only on trusted models or downgrade to older PyTorch vesrion !!!
original_load = torch.load
def load_wrapper(*args, **kwargs):
    logging.warning("The 'weights_only' option of 'torch.load' has been forcibly disabled.")
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = load_wrapper


def main():
    args = get_args()
    logging.basicConfig(level=get_log_level(args), force=True)
    seed_everything(args)
    logging.debug(f"Arguments: {args.__dict__}")

    if args.model_type == 'fastopic':
        wrapper = FASTopicWrapper(args)

        logging.basicConfig(level=logging.WARNING, force=True)

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            with open(args.log_path, "w") as f:
                json_args = json.dumps(args.__dict__, indent=4)
                f.write(f"Arguments: {json_args}\n")
            print(f"Arguments saved to {args.log_path}.")

        if args.eval_dir:
            os.makedirs(args.eval_dir, exist_ok=True)

            if args.test_docs_path is not None:
                eval_wrapper = EvaluationWrapper(wrapper)
                results = eval_wrapper.evaluate()

                print(f"Evaluation results: {results}")
                with open(os.path.join(args.eval_dir, "results.json"), "w") as f:
                    json.dump(results, f, indent=4)
                print(f"Evaluation results saved to {args.eval_dir}.")

            wrapper.visualize_hierarchy(save_path=os.path.join(args.eval_dir, "hierarchy.png"))
            wrapper.visualize_weights(save_path=os.path.join(args.eval_dir, "weights.png"))
            print(f"Visualizations saved to {args.eval_dir}.")

    elif args.model_type == 'lda':
        wrapper = LDAWrapper(args)
        logging.basicConfig(level=logging.WARNING, force=True)

        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            with open(args.log_path, "w") as f:
                json_args = json.dumps(args.__dict__, indent=4)
                f.write(f"Arguments: {json_args}\n")
            print(f"Arguments saved to {args.log_path}.")

        if args.eval_dir:
            os.makedirs(args.eval_dir, exist_ok=True)

            eval_wrapper = LDAEvaluationWrapper(wrapper)
            results = eval_wrapper.evaluate()

            with open(os.path.join(args.eval_dir, "results_lda.json"), "w") as f:
                json.dump(results, f, indent=4)
    elif args.model_type == 'bertopic':
        logging.basicConfig(level=logging.INFO, force=True)
        
        if args.log_path:
            os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
            with open(args.log_path, "w") as f:
                json_args = json.dumps(args.__dict__, indent=4)
                f.write(f"Arguments: {json_args}\n")
            print(f"Arguments saved to {args.log_path}.")

        if args.eval_dir:
            os.makedirs(args.eval_dir, exist_ok=True)
            
            eval_wrapper = BERTopicEvalWrapper(args)
            results = eval_wrapper.evaluate()
            with open(os.path.join(args.eval_dir, "results_bertopic.json"), "w") as f:
                json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()
