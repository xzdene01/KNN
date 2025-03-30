import torch
import logging

from wrappers.fastopic_wrapper import FASTopicWrapper
from utils.argparser import get_args

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

    wrapper = FASTopicWrapper(args)
    print("Topic diversity:", wrapper.topic_diversity)


if __name__ == '__main__':
    main()