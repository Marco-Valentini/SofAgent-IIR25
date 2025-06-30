from sofagent.utils.web import add_chat_message, get_color_style
from sofagent.utils.data import read_json
from sofagent.utils.utils import get_rm
from sofagent.utils.parse import parse_action, init_answer, parse_answer
from sofagent.utils.decorator import run_once
from sofagent.utils.string import format_history, get_avatar, format_chat_history, format_step
from sofagent.utils.prompts import read_prompts
from sofagent.utils.init import init_api, init_all_seeds
from sofagent.utils.check import EM, is_correct