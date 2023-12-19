import random
import numpy as np
import torch

RANDOM_SEED = 0


# def give_seed(seed=RANDOM_SEED):
#     # Python random
#     random.seed(seed)
#     # Numpy
#     np.random.seed(seed)
#     # Pytorch
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     # torch.set_deterministic(True)
#     # torch.use_deterministic_algorithms = True
#     torch.backends.cudnn.benchmark = False


def give_seed(seed=RANDOM_SEED):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.use_deterministic_algorithms = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


give_seed()
