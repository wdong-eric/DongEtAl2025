import configparser
import tomllib
import tomli_w
from collections.abc import Mapping

# Define ANSI escape codes for colors
TMNL_GREEN = "\033[92m"
TMNL_BLUE = "\033[94m"
TMNL_RESET = "\033[0m"


def load_toml(config_path: str) -> dict:
    """
    Load the configuration from a TOML file.
    """
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def write_toml(config: dict, output_path: str):
    """
    Write the current configuration to a TOML file.
    """
    with open(output_path, "wb") as configfile:
        tomli_w.dump(config, configfile)


def load_ini(config_path: str, config: configparser.ConfigParser = None):
    """
    Load the configuration from an INI file.
    """
    if config is None:
        config = configparser.ConfigParser(
            interpolation=configparser.ExtendedInterpolation(), allow_no_value=True
        )
    config.read(config_path)
    return config


def write_ini(output_path: str, config: configparser.ConfigParser):
    """
    Write the current configuration to an INI file.
    """
    with open(output_path, "w") as configfile:
        config.write(configfile)


def parse_override_arg(arg_val: str):
    """
    Parse a arg_val string of the form 'a.b.c=d' into a tuple (['a', 'b', 'c'], 'd').
    """
    if "=" not in arg_val:
        raise ValueError(f"Invalid override argument: {arg_val}")
    path, value = arg_val.split("=", 1)
    keys = path.split(".")
    return keys, value


def set_nested_value(config: dict, keys: list, value: str):
    """
    In place setting a nested value in a dictionary (config) given a list of keys.
    """
    subcfg = config
    for key in keys[:-1]:
        subcfg = subcfg.setdefault(key, {})
    subcfg[keys[-1]] = value


def apply_cli_overrides(config: dict, overrides: list[str]):
    """
    Apply CLI overrides to the configuration dictionary.
    """
    merged = config.copy()
    for override in overrides:
        keys, value = parse_override_arg(override)
        set_nested_value(merged, keys, value)
    return merged


def merge_toml(base: dict, override: dict) -> dict:
    """Recursively merge two dictionaries, with override taking precedence."""
    merged = base.copy()
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(val, Mapping)
        ):
            merged[key] = merge_toml(merged[key], val)
        else:
            merged[key] = val
    return merged
