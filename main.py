# Set cache dir for (transformer) models
import os
os.environ["HF_HOME"] = "./models"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
import logging

from wrappers.fastopic_wrapper import FASTopicWrapper
from utils.argparser import get_args, get_log_level
from utils.seed_everything import seed_everything

# This will allow to load model even with new PyTorch; this is required by torchvision that is required by topmost :(
# This is a workaround that sets the 'weights_only' option of 'torch.load' to False
# !!! Through this workaround can be downloaded even unsafe model, use only on trusted models or downgrade to older PyTorch vesrion !!!
original_load = torch.load
def load_wrapper(*args, **kwargs):
    logging.warning("The 'weights_only' option of 'torch.load' has been forcibly disabled." )
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)
torch.load = load_wrapper

def main():
    args = get_args()
    logging.basicConfig(level=get_log_level(args))
    seed_everything(args)
    logging.debug(f"Arguments: {args.__dict__}")

    wrapper = FASTopicWrapper(args)

    logging.basicConfig(level=logging.WARNING, force=True)
    print("Topic diversity:", wrapper.topic_diversity)
    # print("Topic coherence:", wrapper.topic_coherence)
    wrapper.visualize_hierarchy()
    wrapper.visualize_weights()


if __name__ == '__main__':
    main()