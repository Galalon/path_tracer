import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
import numpy as np


class Config(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def validate(self):
        """Validate the configuration values."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary with class name as a key."""
        def serialize_value(value):
            if isinstance(value, Config):
                return value.to_dict()
            elif isinstance(value, np.ndarray):
                return {"__numpy_array__": True, "data": value.tolist()}
            elif isinstance(value, list):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            else:
                return value

        config_dict = {key: serialize_value(value) for key, value in self.__dict__.items()}
        config_dict["config_class"] = self.__class__.__name__
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a configuration instance from a dictionary, verifying the class name."""
        def deserialize_value(value):
            if isinstance(value, dict) and "config_class" in value:
                return cls._get_subclass_from_dict(value)
            elif isinstance(value, dict) and value.get("__numpy_array__"):
                return np.array(value["data"])
            elif isinstance(value, list):
                return [deserialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: deserialize_value(v) for k, v in value.items()}
            else:
                return value

        if config_dict.get("config_class") != cls.__name__:
            raise ValueError(
                f"Invalid config class. Expected {cls.__name__}, but got {config_dict.get('config_class')}")

        config_dict.pop("config_class")

        deserialized_dict = {key: deserialize_value(value) for key, value in config_dict.items()}
        cfg = cls(**deserialized_dict)
        cfg.validate()
        return cfg

    @classmethod
    def _get_subclass_from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Helper method to handle nested Config subclasses."""
        config_class_name = config_dict.get("config_class")
        config_class = globals().get(config_class_name)
        if config_class and issubclass(config_class, Config):
            return config_class.from_dict(config_dict)
        raise ValueError(f"Invalid config class: {config_class_name}")

    def to_json(self) -> str:
        """Convert the configuration to a JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_json(cls, json_str: str) -> 'Config':
        """Create a configuration instance from a JSON string."""
        config_dict = json.loads(json_str)
        cfg = cls.from_dict(config_dict)
        cfg.validate()
        return cfg

    def to_file(self, file_path: str):
        """Write the configuration to a JSON file."""
        with open(file_path, "w") as file:
            json.dump(self.to_dict(), file, indent=4)

    @classmethod
    def from_file(cls, file_path: str) -> 'Config':
        """Load the configuration from a JSON file."""
        with open(file_path, "r") as file:
            config_dict = json.load(file)
        cfg = cls.from_dict(config_dict)
        cfg.validate()
        return cfg
