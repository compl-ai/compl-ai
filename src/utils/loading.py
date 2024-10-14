#    Copyright 2024 SRI Lab @ ETH Zurich, LatticeFlow AI, INSAIT
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from functools import partial

import yaml
from pydantic import ValidationError
from yamlinclude import YamlIncludeConstructor

from config import CONFIG_DIR
from src.configs.base_config import Config


def read_config_from_yaml(path: str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.
    Instead of the ordinary yaml loader, this function uses the yaml.SafeLoader to avoid code execution.
    Moreover, it uses the YamlIncludeConstructor to enable the inclusion of other YAML files.

    Args:
        path (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.
    """

    with open(path, "r") as stream:
        try:
            YamlIncludeConstructor.add_to_loader_class(
                loader_class=yaml.SafeLoader, base_dir=CONFIG_DIR
            )
            yaml_obj = yaml.load(stream, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return yaml_obj


def parse_config(config_obj: dict) -> Config:
    """
    Parses the given configuration object and returns a Config instance.

    Args:
        config_obj (dict): The configuration object to parse.

    Returns:
        Config: The parsed configuration object.

    Raises:
        ValidationError: If the configuration object is invalid.
    """

    try:
        cfg = Config(**config_obj["config"])
        return cfg
    except ValidationError as exc:
        print(exc)
        raise exc


def patch_data_config_prompt_config(model, data_config, *kargs):
    """
    Patch the data configuration with the given model and update the tokenizer name in the prompt configuration.
    This information is necessary for instantiating the correct chat template in the `ChatFormatterLight` class.

    Args:
        model (str): The name of the model.
        data_config (dict): The data configuration dictionary.

    Returns:
        dict: The updated data configuration dictionary.
    """

    if "prompt_config" in data_config:
        data_config["prompt_config"]["tokenizer_name"] = model
    return data_config


def patch_data_config_add_debug(data_config, subset_size=None):
    data_config["debug"] = True
    if subset_size:
        data_config["subset_size"] = subset_size
    return data_config


def patch_data_configs(config, patch_data_config_fn, subset_size=None):
    """
    Patch the data configurations in the given config using the provided patch_data_config_fn.
    This is done in-place. The function modifies the config dictionary directly.
    Moreover, it supports both single and multiple data configurations. If the data configuration is a list,
    the function iterates over each configuration and applies the patch_data_config_fn.

    Args:
        config (dict): The configuration dictionary.
        patch_data_config_fn (function): The function to patch the data configurations.

    Returns:
        None
    """

    if "data_config" in config["benchmark_configs"][0]:
        data_configs = config["benchmark_configs"][0]["data_config"]
        if isinstance(config["benchmark_configs"][0]["data_config"], list):
            config["benchmark_configs"][0]["data_config"] = [
                patch_data_config_fn(dt_cfg, subset_size) for dt_cfg in data_configs
            ]

        else:
            config["benchmark_configs"][0]["data_config"] = patch_data_config_fn(
                data_configs, subset_size
            )


def patch_config(
    config: dict,
    model,
    model_config,
    batch_size,
    debug_mode,
    answers_file,
    cpu_mode,
    subset_size,
    device,
):
    """
    Patch the given configuration dictionary with command line parameters.

    Args:
        config (dict): The original configuration dictionary.
        model (str): The name of the model.
        model_config (dict): The model configuration.
        batch_size (int): The batch size for the model.
        debug_mode (bool): Whether to enable debug mode.
        answers_file (str): The path to the answers directory.
        cpu_mode (bool): Whether to enable CPU mode.
        device (str | None): The device to use.

    Returns:
        dict: The patched configuration dictionary.
    """

    print(config)
    old_config = config
    config = config["config"]
    # Patch config dictionary with command line parameters
    if model_config:
        model_config_obj = read_config_from_yaml(model_config)
        config["model"] = model_config_obj
        model_name = model_config_obj["name"]
        patch_data_configs(config, partial(patch_data_config_prompt_config, model_name))

    if model:
        config["model"]["name"] = model
        config["model"]["tokenizer_name"] = model
        patch_data_configs(config, partial(patch_data_config_prompt_config, model))

    if answers_file:
        config["model"]["answers"] = answers_file

    if debug_mode:
        patch_data_configs(config, patch_data_config_add_debug, subset_size)

    if cpu_mode:
        config["benchmark_configs"][0]["debug"] = {"cpu_mode": True}

    if batch_size:
        config["model"]["batch_size"] = batch_size

    if device:
        config["model"]["device"] = device

    old_config["config"] = config
    config = old_config

    return config
