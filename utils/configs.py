import yaml
import pathlib
import argparse
import sys
from omegaconf import OmegaConf
from collections import defaultdict
from typing import Any, Dict, Optional


def load_configs(parser:argparse.ArgumentParser) -> OmegaConf:
    cli_provided, cli_defaults = omegaconf_from_argparse(parser)
    all_cli = OmegaConf.merge(cli_defaults, cli_provided)

    if all_cli.eval_dir:
        # Just load the dumped config file if we are evaluating
        eval_config_path = pathlib.Path(all_cli.eval_dir) / 'configs'
        file_cfg = OmegaConf.load(eval_config_path/"config.yaml")
        cfg = OmegaConf.merge(cli_defaults, file_cfg, cli_provided)

    elif all_cli.config:
        file_cfg = OmegaConf.load(cli_provided.config)
        default_cfg = OmegaConf.load("configs/run/default.yaml")

        # Generate a config with enough info to load other config files
        bootstrap_cfg = OmegaConf.merge(cli_defaults, default_cfg, file_cfg, cli_provided)

        encoder_cfg = OmegaConf.load(bootstrap_cfg.encoder_config_path) 
        dataset_cfg = OmegaConf.load(bootstrap_cfg.dataset_config_path) 
        segmentor_cfg = OmegaConf.load(bootstrap_cfg.segmentor_config_path)
        augmentation_cfg = OmegaConf.load(bootstrap_cfg.augmentation_config_path)
   
        # Set some invariants based on other configs
        segmentor_cfg['num_classes'] = dataset_cfg['num_classes']
        segmentor_cfg['in_channels'] = encoder_cfg['embed_dim']
        segmentor_cfg['multi_temporal'] = encoder_cfg['multi_temporal'] = dataset_cfg['multi_temporal']
       
        # the encoder can handle any number of input channels, e.g., DOFA
        if not encoder_cfg.get("input_bands"):
            encoder_cfg["input_bands"] = dataset_cfg['bands']

        segmentor_cfg["loss"]['distribution'] = dataset_cfg['distribution']
        segmentor_cfg["loss"]['ignore_index'] = dataset_cfg['ignore_index']

        file_cfg["encoder"] = encoder_cfg
        file_cfg["dataset"] = dataset_cfg
        file_cfg["segmentor"] = segmentor_cfg
        file_cfg["augmentation"] = augmentation_cfg

        # Assemble final config file
        cfg = OmegaConf.merge(cli_defaults, default_cfg, file_cfg, cli_provided)

    else:
        raise ValueError("Either the --config, or the --eval_dir argument is required.")
   
    return cfg


def omegaconf_from_argparse(parser: argparse.ArgumentParser):
    """Parse CLI arguments into two omegaconf configs, one for default arguments, one for provided ones.
    These can be merged with config files to have the correct priority (CLI > config > defults).

    From https://github.com/omry/omegaconf/issues/569

    Args:
        parser (argparse.ArgumentParser): Argument parser to use for parsing

    Returns:
        Tuple(omegaconf.OmegaConf, omegaconf.OmegaConf): CLI provided and default arguments
    """
    dest_to_arg = {v.dest: k for k, v in parser._option_string_actions.items()}

    all_args = vars(parser.parse_args())
    provided_args = {}
    default_args = {}
    for k, v in all_args.items():
        if dest_to_arg[k] in sys.argv:
            provided_args[k] = v
        else:
            default_args[k] = v

    provided = OmegaConf.create(_nest(provided_args))
    defaults = OmegaConf.create(_nest(default_args))

    return provided, defaults


def _nest(
    d: Dict[str, Any], separator: str = ".", include_none: bool = False
) -> Optional[Dict[str, Any]]:
    """_nest Recursive function to nest a dictionary on keys with . (dots)

    Parse documentation into a hierarchical dict. Keys should be separated by dots (e.g. "model.hidden") to go down into the hierarchy
    From https://github.com/omry/omegaconf/issues/569
    
    Args:
        d (Dict[str, Any]): dictionary containing flat config values
        separator (str): Separator to nest dictionary
        include_none (bool): If true includes none values in final dict

    Returns:
        Dict[str, Any]: Hierarchical config dictionary

    Examples:
        >>> _nest({{"model.hidden": 20, "optimizer.lr": 1e-3}})
        {"model": {"hidden": 20}, "optimizer": {"lr": 1e-3}}
    """
    nested: Dict[str, Any] = defaultdict(dict)

    for key, val in d.items():
        if separator in key:
            splitkeys = key.split(separator)
            inner = _nest({separator.join(splitkeys[1:]): val})

            if inner is not None:
                nested[splitkeys[0]].update(inner)
        else:
            if val is not None:
                nested[key] = val

            if val is None and include_none:
                nested[key] = val

    return dict(nested) if nested else None


def ensure_compatible_configs(cfg:OmegaConf) -> OmegaConf:
    # SpectralGPT_Encoder can handle multi-temporal input, but in change detection, we encode each time step separately,
    # to then compute the change from the different feature representations. 
    if cfg.encoder.encoder_name == "SpectralGPT_Encoder" and cfg.segmentor.task_name == "change-detection":
        cfg.encoder.multi_temporal=1
    return cfg