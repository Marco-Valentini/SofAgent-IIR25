from sofagent.systems.base import System
from sofagent.systems.collaboration import CollaborationSystem

SYSTEMS: list[type[System]] = [value for value in globals().values() if isinstance(value, type) and issubclass(value, System) and value != System]