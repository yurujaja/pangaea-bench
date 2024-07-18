import yaml


def load_specific_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_config(args):
    #args = vars(args)
    #cfg_path = args["run_config"]
    #with open(cfg_path, "r") as file:
    #    train_config = yaml.safe_load(file)
    encoder_config = load_specific_config(args.encoder_config)#load_specific_config(args, "encoder_config", train_config=train_config)
    dataset_config = load_specific_config(args.dataset_config)#load_specific_config(args, "dataset_config", train_config=train_config)
    segmentor_config = load_specific_config(args.segmentor_config)#load_specific_config(args, "task_config", train_config=train_config)

    segmentor_config['num_classes'] = dataset_config['num_classes']
    segmentor_config['in_channels'] = encoder_config['embed_dim']

    # the encoder can handle any number of input channels, e.g., DOFA
    if not encoder_config.get("input_bands"):
        encoder_config["input_bands"] = dataset_config['bands']

    # Add task_config parameters from dataset
    # if dataset_config.get("num_classes"):
    #     task_config["num_classes"] = dataset_config["num_classes"]
    #
    # # Validate config
    # if dataset_config.get("img_size") and encoder_config["encoder_model_args"].get("img_size"):
    #     if dataset_config["img_size"] != encoder_config["encoder_model_args"]["img_size"]:
    #         print(f"Warning: dataset img_size {dataset_config['img_size']} and encoder img_size {encoder_config['encoder_model_args']['img_size']} do not match. {encoder_config['encoder_model_args']['img_size']} is used.")
    #     task_config["img_size"] = encoder_config["encoder_model_args"]["img_size"]
    #
    # if not dataset_config["multi_temporal"] and task_config["head_args"]["num_frames"] > 1:
    #     raise ValueError("task head num_frame > 1 is only supported for multi_temporal datasets.")

    return encoder_config, dataset_config, segmentor_config