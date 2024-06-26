# -*- coding: utf-8 -*-
''' 
Authors: Yuru Jia, Valerio Marsocci
'''

import os
import importlib

def make_dataset(dataset_config):
    components = dataset_config['dataset'].split('.')
    module_string = '.'.join(components)
    class_string = components[-1]
    module = importlib.import_module(module_string)
    class_object = getattr(module, class_string)
    dataset = class_object()

    if hasattr(dataset, 'get_splits') and callable(dataset.get_splits):
        return dataset.get_splits(dataset_config)
    else:
        raise TypeError(f"Please make sure your dataset {dataset_config['dataset']} implements a get_splits method.")
