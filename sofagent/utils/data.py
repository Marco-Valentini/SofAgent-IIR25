import json

def read_json(path: str) -> dict:
    """Read json file.

    Args:
        `path` (`str`): Path to the json file.

    Returns:
        `dict`: The json data.
    """
    with open(path, 'r') as f:
        return json.load(f)