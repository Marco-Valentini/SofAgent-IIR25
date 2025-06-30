from argparse import ArgumentParser

from sofagent.tasks.base import Task
from sofagent.systems import CollaborationSystem
from sofagent.utils import init_api, read_json

import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


class CollaborativeChatTask(Task):
    @staticmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--api_config', type=str, default='config/api-config.json', help='Api configuration file')
        parser.add_argument('--system', type=str, default='collaboration', choices=['collaboration'], help='System name')
        parser.add_argument('--system_config', type=str, default="config/systems/collaboration/chat.json",
                            help='System configuration file')
        return parser

    def get_system(self, system: str, config_path: str):
        if system == 'collaboration':
            return CollaborationSystem(config_path=config_path, task='chat')
        else:
            raise NotImplementedError

    def run(self, api_config: str, system: str, system_config: str, *args, **kwargs) -> None:
        init_api(read_json(api_config))
        self.system = self.get_system(system, system_config)
        self.system.chat()


if __name__ == '__main__':
    CollaborativeChatTask().launch()
