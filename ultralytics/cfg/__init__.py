# Ultralytics YOLO 🚀, AGPL-3.0 license

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union


from ultralytics.utils import (
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    colorstr,
    deprecation_warn,
    yaml_load,
)

# Define valid tasks and modes
MODES = {"predict", "track"}
TASKS = {"detect"}
TASK2DATA = {
    "detect": "coco8.yaml",
}
TASK2MODEL = {
    "detect": "yolo11n.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []

CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + ARGV[1:])}."""

CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "conf",
    "iou",
    "line_width_percentage"
}
CFG_INT_KEYS = {  # integer-only arguments
    "seed",
    "max_det",
    "vid_stride"
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
    "exist_ok",
    "verbose",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "show_boxes",
}


def cfg2dict(cfg):
    """
    Converts a configuration object to a dictionary.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration object to be converted. Can be a file path,
            a string, a dictionary, or a SimpleNamespace object.

    Returns:
        (Dict): Configuration object in dictionary format.

    Examples:
        Convert a YAML file path to a dictionary:
        >>> config_dict = cfg2dict("config.yaml")

        Convert a SimpleNamespace to a dictionary:
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        Pass through an already existing dictionary:
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Notes:
        - If cfg is a path or string, it's loaded as YAML and converted to a dictionary.
        - If cfg is a SimpleNamespace object, it's converted to a dictionary using vars().
        - If cfg is already a dictionary, it's returned unchanged.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary, with optional overrides.

    Args:
        cfg (str | Path | Dict | SimpleNamespace): Configuration data source. Can be a file path, dictionary, or
            SimpleNamespace object.
        overrides (Dict | None): Dictionary containing key-value pairs to override the base configuration.

    Returns:
        (SimpleNamespace): Namespace containing the merged configuration arguments.

    Examples:
        >>> from ultralytics.cfg import get_cfg
        >>> config = get_cfg()  # Load default configuration
        >>> config = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    Notes:
        - If both `cfg` and `overrides` are provided, the values in `overrides` will take precedence.
        - Special handling ensures alignment and correctness of the configuration, such as converting numeric
          `project` and `name` to strings and validating configuration keys and values.
        - The function performs type and value checks on the configuration data.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(cfg, overrides)
        # merge cfg and overrides dicts (prefer overrides)
        cfg = {**cfg, **overrides}

    # Special handling for numeric project/name
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # assign model to 'name' arg
        cfg["name"] = cfg.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={
                       cfg['name']}'.")

    # Type and Value checks
    check_cfg(cfg)

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg(cfg, hard=True):
    """
    Checks configuration argument types and values for the Ultralytics library.

    This function validates the types and values of configuration arguments, ensuring correctness and converting
    them if necessary. It checks for specific key types defined in global variables such as CFG_FLOAT_KEYS,
    CFG_FRACTION_KEYS, CFG_INT_KEYS, and CFG_BOOL_KEYS.

    Args:
        cfg (Dict): Configuration dictionary to validate.
        hard (bool): If True, raises exceptions for invalid types and values; if False, attempts to convert them.

    Examples:
        >>> config = {
        ...     "epochs": 50,  # valid integer
        ...     "lr0": 0.01,  # valid float
        ...     "momentum": 1.2,  # invalid float (out of 0.0-1.0 range)
        ...     "save": "true",  # invalid bool
        ... }
        >>> check_cfg(config, hard=False)
        >>> print(config)
        {'epochs': 50, 'lr0': 0.01, 'momentum': 1.2, 'save': False}  # corrected 'save' key

    Notes:
        - The function modifies the input dictionary in-place.
        - None values are ignored as they may be from optional arguments.
        - Fraction keys are checked to be within the range [0.0, 1.0].
    """
    for k, v in cfg.items():
        if v is not None:  # None values may be from optional args
            if k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    if hard:
                        raise TypeError(
                            f"'{k}={v}' is of invalid type {
                                type(v).__name__}. "
                            f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{
                                k}=0.5')"
                        )
                    cfg[k] = v = float(v)
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{
                                     k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{
                            k}' must be an int (i.e. '{k}=8')"
                    )
                cfg[k] = int(v)
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                if hard:
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"'{k}' must be a bool (i.e. '{k}=True' or '{
                            k}=False')"
                    )
                cfg[k] = bool(v)


def get_save_dir(args, name=None):
    """
    Returns the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task',
            'mode', and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name'
            or the 'args.mode'.

    Returns:
        (Path): Directory path where outputs should be saved.

    Examples:
        >>> from types import SimpleNamespace
        >>> args = SimpleNamespace(project="my_project", task="detect", mode="train", exist_ok=True)
        >>> save_dir = get_save_dir(args)
        >>> print(save_dir)
        my_project/detect/train
    """
    if getattr(args, "save_dir", None):
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path

        project = args.project or (
            ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
        name = name or args.name or f"{args.mode}"
        save_dir = increment_path(
            Path(project) / name, exist_ok=args.exist_ok if RANK in {-1, 0} else True)

    return Path(save_dir)


def _handle_deprecation(custom):
    """
    Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings.

    Args:
        custom (Dict): Configuration dictionary potentially containing deprecated keys.

    Examples:
        >>> custom_config = {"boxes": True, "hide_labels": "False", "line_thickness": 2}
        >>> _handle_deprecation(custom_config)
        >>> print(custom_config)
        {'show_boxes': True, 'show_labels': True, 'line_width': 2}

    Notes:
        This function modifies the input dictionary in-place, replacing deprecated keys with their current
        equivalents. It also handles value conversions where necessary, such as inverting boolean values for
        'hide_labels' and 'hide_conf'.
    """
    for key in custom.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            custom["show_boxes"] = custom.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            custom["show_labels"] = custom.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            custom["show_conf"] = custom.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            custom["line_width"] = custom.pop("line_thickness")
        if key == "label_smoothing":
            deprecation_warn(key)
            custom.pop("label_smoothing")

    return custom


def check_dict_alignment(base: Dict, custom: Dict, e=None):
    """
    Checks alignment between custom and base configuration dictionaries, handling deprecated keys and providing error
    messages for mismatched keys.

    Args:
        base (Dict): The base configuration dictionary containing valid keys.
        custom (Dict): The custom configuration dictionary to be checked for alignment.
        e (Exception | None): Optional error instance passed by the calling function.

    Raises:
        SystemExit: If mismatched keys are found between the custom and base dictionaries.

    Examples:
        >>> base_cfg = {"epochs": 50, "lr0": 0.01, "batch_size": 16}
        >>> custom_cfg = {"epoch": 100, "lr": 0.02, "batch_size": 32}
        >>> try:
        ...     check_dict_alignment(base_cfg, custom_cfg)
        ... except SystemExit:
        ...     print("Mismatched keys found")

    Notes:
        - Suggests corrections for mismatched keys based on similarity to valid keys.
        - Automatically replaces deprecated keys in the custom configuration with updated equivalents.
        - Prints detailed error messages for each mismatched key to help users correct their configurations.
    """
    custom = _handle_deprecation(custom)

    print("BASE KETY", base)
    print("CUSTOM KEYS", custom)

    base_keys, custom_keys = (set(x.keys()) for x in (base, custom))
    mismatched = [k for k in custom_keys if k not in base_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, base_keys)  # key list
            matches = [f"{k}={base[k]}" if base.get(
                k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {
                matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)
                          }' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' in a list of strings and joins fragments with brackets.

    This function handles the following cases:
    1. ['arg', '=', 'val'] becomes ['arg=val']
    2. ['arg=', 'val'] becomes ['arg=val']
    3. ['arg', '=val'] becomes ['arg=val']
    4. Joins fragments with brackets, e.g., ['imgsz=[3,', '640,', '640]'] becomes ['imgsz=[3,640,640]']

    Args:
        args (List[str]): A list of strings where each element represents an argument or fragment.

    Returns:
        List[str]: A list of strings where the arguments around isolated '=' are merged and fragments with brackets are joined.

    Examples:
        >>> args = ["arg1", "=", "value", "arg2=", "value2", "arg3", "=value3", "imgsz=[3,", "640,", "640]"]
        >>> merge_and_join_args(args)
        ['arg1=value', 'arg2=value2', 'arg3=value3', 'imgsz=[3,640,640]']
    """
    new_args = []
    current = ""
    depth = 0

    i = 0
    while i < len(args):
        arg = args[i]

        # Handle equals sign merging
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            i += 2
            continue
        # merge ['arg=', 'val']
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:
            new_args.append(f"{arg}{args[i + 1]}")
            i += 2
            continue
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
            i += 1
            continue

        # Handle bracket joining
        depth += arg.count("[") - arg.count("]")
        current += arg
        if depth == 0:
            new_args.append(current)
            current = ""

        i += 1

    # Append any remaining current string
    if current:
        new_args.append(current)

    return new_args


def parse_key_value_pair(pair: str = "key=value"):
    """
    Parses a key-value pair string into separate key and value components.

    Args:
        pair (str): A string containing a key-value pair in the format "key=value".

    Returns:
        key (str): The parsed key.
        value (str): The parsed value.

    Raises:
        AssertionError: If the value is missing or empty.

    Examples:
        >>> key, value = parse_key_value_pair("model=yolo11n.pt")
        >>> print(f"Key: {key}, Value: {value}")
        Key: model, Value: yolo11n.pt

        >>> key, value = parse_key_value_pair("epochs=100")
        >>> print(f"Key: {key}, Value: {value}")
        Key: epochs, Value: 100

    Notes:
        - The function splits the input string on the first '=' character.
        - Leading and trailing whitespace is removed from both key and value.
        - An assertion error is raised if the value is empty after stripping.
    """
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """
    Converts a string representation of a value to its appropriate Python type.

    This function attempts to convert a given string into a Python object of the most appropriate type. It handles
    conversions to None, bool, int, float, and other types that can be evaluated safely.

    Args:
        v (str): The string representation of the value to be converted.

    Returns:
        (Any): The converted value. The type can be None, bool, int, float, or the original string if no conversion
            is applicable.

    Examples:
        >>> smart_value("42")
        42
        >>> smart_value("3.14")
        3.14
        >>> smart_value("True")
        True
        >>> smart_value("None")
        None
        >>> smart_value("some_string")
        'some_string'

    Notes:
        - The function uses a case-insensitive comparison for boolean and None values.
        - For other types, it attempts to use Python's eval() function, which can be unsafe if used on untrusted input.
        - If no conversion is possible, the original string is returned.
    """
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        try:
            return eval(v)
        except Exception:
            return v


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """
    Copies the default configuration file and creates a new one with '_copy' appended to its name.

    This function duplicates the existing default configuration file (DEFAULT_CFG_PATH) and saves it
    with '_copy' appended to its name in the current working directory. It provides a convenient way
    to create a custom configuration file based on the default settings.

    Examples:
        >>> copy_default_cfg()
        # Output: default.yaml copied to /path/to/current/directory/default_copy.yaml
        # Example YOLO command with this new custom cfg:
        #   yolo cfg='/path/to/current/directory/default_copy.yaml' imgsz=320 batch=8

    Notes:
        - The new configuration file is created in the current working directory.
        - After copying, the function prints a message with the new file's location and an example
          YOLO command demonstrating how to use the new configuration file.
        - This function is useful for users who want to modify the default configuration without
          altering the original file.
    """
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{
            new_file}' imgsz=320 batch=8"
    )
