import yaml

def load_specific_config(args, key):
    if args.get(key):
        with open(args[key], "r") as file:
            return yaml.safe_load(file)
    elif train_config.get(key):
        with open(train_config[key], "r") as file:
            return yaml.safe_load(file)
    else:
        raise ValueError(f"No configuration found for {key}")

def load_config(args):
    cfg_path = args["run_config"]
    with open(cfg_path, "r") as file:
        train_config = yaml.safe_load(file)

    encoder_config = load_specific_config(args, "encoder_config")
    dataset_config = load_specific_config(args, "dataset_config") 
    task_config = load_specific_config(args, "task_config")

    return train_config, encoder_config, dataset_config, task_config