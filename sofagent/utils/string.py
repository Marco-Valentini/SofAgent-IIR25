def format_history(history: list[dict]) -> str:
    """Format history prompt. Add a newline between each turn in `history`.

    Args:
        `history` (`list[dict]`): A list of turns in the history. Each turn is a dictionary with keys `command` and `observation`.
    Returns:
        `str`: The formatted history prompt. If `history` is empty, return an empty string.
    """
    if history == []:
        return ''
    else:
        return '\n' + '\n'.join(
            [f"Command: {turn['command']}\nObservation: {turn['observation']}\n" for turn in history]) + '\n'


def get_avatar(agent_type: str) -> str:
    """Get the avatar of the agent.

    Args:
        `agent_type` (`str`): The type of the agent.
    Returns:
        `str`: The avatar of the agent.
    """
    if 'manager' in agent_type.lower():
        return '👨🏻‍💼'
    elif 'searcher' in agent_type.lower():
        return '🔍'
    else:
        return '🤖'

def format_step(step: str) -> str:
    """Format a step prompt. Remove leading and trailing whitespaces and newlines, and replace newlines with spaces.

    Args:
        `step` (`str`): A step prompt in string format.
    Returns:
        `str`: The formatted step prompt.
    """
    return step.strip('\n').strip().replace('\n', '')


def format_chat_history(history: list[tuple[str, str]]) -> str:
    """Format chat history prompt. Add a newline between each turn in `history`.

    Args:
        `history` (`list[tuple[str, str]]`): A list of turns in the chat history. Each turn is a tuple with the first element being the chat record and the second element being the role.
    Returns:
        `str`: The formatted chat history prompt. If `history` is empty, return `'No chat history.\\n'`.
    """
    if history == []:
        return 'No chat history.\n'
    else:
        return '\n' + '\n'.join([f"{role.capitalize()}: {chat}" for chat, role in history]) + '\n'