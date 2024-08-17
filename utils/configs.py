import yaml
import os

def load_specific_config(path):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_config(args):
    #args = vars(args)
    #cfg_path = args["run_config"]
    #with open(cfg_path, "r") as file:
    #    train_config = yaml.safe_load(file)
    if args.test_only and os.path.exists(os.path.join(args.test_only, 'configs')):
        encoder_config = load_specific_config(os.path.join(args.test_only, 'configs', 'encoder_config.yaml'))
        dataset_config = load_specific_config(os.path.join(args.test_only, 'configs', 'dataset_config.yaml'))
        segmentor_config = load_specific_config(os.path.join(args.test_only, 'configs', 'segmentor_config.yaml'))
        
    elif os.path.exists(args.encoder_config) and \
        os.path.exists(args.dataset_config) and \
        os.path.exists(args.segmentor_config):

        encoder_config = load_specific_config(args.encoder_config) 
        dataset_config = load_specific_config(args.dataset_config) 
        segmentor_config = load_specific_config(args.segmentor_config)
   
        segmentor_config['num_classes'] = dataset_config['num_classes']
        segmentor_config['in_channels'] = encoder_config['embed_dim']
        segmentor_config['multi_temporal'] = encoder_config['multi_temporal'] = dataset_config['multi_temporal']
       
        # the encoder can handle any number of input channels, e.g., DOFA
        if not encoder_config.get("input_bands"):
            encoder_config["input_bands"] = dataset_config['bands']

        segmentor_config["loss"]['distribution'] = dataset_config['distribution']

    else:
        raise ValueError("Missing necessary configuration files")
   
    return encoder_config, dataset_config, segmentor_config


def write_config(config, path):
    with open(path, "w") as file:
        yaml.dump(config, file)