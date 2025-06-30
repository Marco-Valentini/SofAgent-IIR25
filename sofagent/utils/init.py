# Description: Initialization functions.

import os
import random
import numpy as np
import torch


def init_api(api_config: dict):
    """Initialize an API among OpenAI API or Azure OpenAI API

        Args:
            `api_config` (`dict`): OpenAI API configuration, should contain `api_base` and `api_key`.

        OR
        Args:
        `api_config` (`dict`): Azure OpenAI API configuration, should contain:
            - `api_base`: Azure OpenAI endpoint (e.g., "https://your-resource-name.openai.azure.com/")
            - `api_key`: Azure OpenAI API key
            - `api_version`: The API version to use (default: "2024-02-15-preview")
    """
    if api_config["api_type"] == "openai":
        os.environ["OPENAI_API_BASE"] = api_config['api_base']
        os.environ["OPENAI_API_KEY"] = api_config['api_key']
    elif api_config["api_type"] == "azure_openai":
        os.environ["OPENAI_API_BASE"] = api_config['api_base']
        os.environ["OPENAI_API_KEY"] = api_config['api_key']
        os.environ["OPENAI_API_VERSION"] = api_config.get("api_version", "2023-12-01-preview")  # Default version
    else:
        raise ValueError(f"Invalid API type: {api_config['api_type']}")


def init_all_seeds(seed: int = 0) -> None:
    """Initialize all seeds.

    Args:
        `seed` (`int`, optional): Random seed. Defaults to `0`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
