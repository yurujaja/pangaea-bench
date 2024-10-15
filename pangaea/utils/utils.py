import os as os
import random
from pathlib import Path

import numpy as np
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# to make flops calculator work
def prepare_input(input_res):
    image = {}
    x1 = torch.FloatTensor(*input_res)
    # input_res[-2] = 2
    input_res = list(input_res)
    input_res[-3] = 2
    x2 = torch.FloatTensor(*tuple(input_res))
    image["optical"] = x1
    image["sar"] = x2
    return dict(img=image)


def get_best_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_best.pth"))
    )

def get_final_model_ckpt_path(exp_dir: str | Path) -> str:
    return os.path.join(
        exp_dir, next(f for f in os.listdir(exp_dir) if f.endswith("_final.pth"))
    )
