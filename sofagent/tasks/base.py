from abc import ABC, abstractmethod
from argparse import ArgumentParser
from loguru import logger
from typing import Any

class Task(ABC):
    @staticmethod
    @abstractmethod
    def parse_task_args(parser: ArgumentParser) -> ArgumentParser:
        """Parse task arguments.

        Args:
            `parser` (`ArgumentParser`): An `ArgumentParser` object to parse arguments.
        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        Returns:
            `ArgumentParser`: The `ArgumentParser` object with arguments added.
        """
        raise NotImplementedError

    def __getattr__(self, __name: str) -> Any:
        # return none if attribute not exists
        if __name not in self.__dict__:
            return None
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{__name}'")

    @abstractmethod
    def run(self, *args, **kwargs):
        """The running pipeline of the task.

        Raises:
            `NotImplementedError`: Should be implemented in subclasses.
        """
        raise NotImplementedError

    def launch(self) -> Any:
        """Launch the task. Parse the arguments with `parse_task_args` and run the task with `run`. The parsed arguments are stored in `self.args` and passed to the `run` method.

        Returns:
            `Any`: The return value of the `run` method.
        """
        parser = ArgumentParser()
        parser = self.parse_task_args(parser)
        args, extras = parser.parse_known_args()
        self.args = args
        # log the arguments
        logger.success(args)
        return self.run(**vars(args))
