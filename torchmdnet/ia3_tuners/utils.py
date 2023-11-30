import re


class IA3Config:
    def __init__(self, args):
        # Adapter Config
        self.init_ia3_weights = True
        self.is_feedforward = False
        self.inference_mode = False
        self.target_modules = [
            "k_proj",
            "v_proj",
            "o_proj",
            "dk_proj",
            "dv_proj",
            "vec_proj"
        ]

        # self.trainable_param_names = ".*layer_norm.*|.*adapter.*"


def get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def check_target_module_exists(config, key: str):
    """A helper method to check if the passed module's key name matches any of the target modules in the adapter_config.

    Args:
        config (`LoraConfig` | `LycorisConfig`): A config to match target modules from
        key (`str`): A key to search any matches in config

    Returns:
        `bool` | `re.Match[str]` | `None`: True of match object if key matches any target modules from config, False or
        None if no match found
    """
    if isinstance(config.target_modules, str):
        target_module_found = re.fullmatch(config.target_modules, key)
    else:
        target_module_found = key in config.target_modules or any(
            key.endswith(f".{target_key}") for target_key in config.target_modules
        )
    return target_module_found
