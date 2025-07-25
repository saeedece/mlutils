import argparse
import ast
import tomllib
from pathlib import Path
from typing import Any

PARSER = argparse.ArgumentParser()
_ = PARSER.add_argument(
    "--config_file",
    type=Path,
    required=True,
    help="Path to configuration file (prefer absolute paths)",
)


def parse_cmdline_string(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return s


def load_config() -> dict[str, Any]:
    args, uargs = PARSER.parse_known_args()
    with open(args.config_file, mode="rb") as f:
        config = tomllib.load(f)

    # override config file with command line args
    uargs = zip(uargs[::2], uargs[1::2])
    uargs = [(key[2:].split("."), parse_cmdline_string(value)) for key, value in uargs]

    for keys, value in uargs:
        if len(keys) == 1:
            config[keys[0]] = value
        elif len(keys) == 2:
            if keys[0] in config:
                assert isinstance(config[keys[0]], dict)
                config[keys[0]][keys[1]] = value
            else:
                config[keys[0]] = {keys[1]: value}
        elif len(keys) == 3:
            if keys[0] in config:
                assert isinstance(config[keys[0]], dict)
                if keys[1] in config[keys[0]]:
                    assert isinstance(config[keys[0]][keys[1]], dict)
                    config[keys[0]][keys[1]][keys[2]] = value
                else:
                    config[keys[0]][keys[1]] = {keys[2]: value}
            else:
                config[keys[0]] = {keys[1]: {keys[2]: value}}
        else:
            raise ValueError("Maximum key depth is 3.")

    return config
