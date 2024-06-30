# -*- coding: utf-8 -*-

import numpy as np
import argparse
import importlib


def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")
	
    	
def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def load_class(class_name:str):
    try:
        components = class_name.split('.')
        module_string = '.'.join(components[:-1])
        class_string = components[-1]
        module = importlib.import_module(module_string)
        return getattr(module, class_string)
    
    except ImportError as ie:
        raise ImportError(f"Unable to load module '{module_string}': {ie}") from ie
    
    except AttributeError as ae:
        raise AttributeError(f"Module '{module_string}' does not contain a class named '{class_string}': {ae}") from ae
    
    except Exception as e:
        raise Exception(f"An unexpected error occurred while trying to load '{class_name}': {e}") from e