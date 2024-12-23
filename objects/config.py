import json
from abc import ABC, abstractmethod
from typing import Any, Dict


class Config(ABC):
    def __init__(self):
        self.buffer_size_hw = (None, None)

    @abstractmethod
    def validate(self):
        """Validate the configuration values."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary with class name as a key."""
        config_dict = self.__dict__.copy()  # Get the instance's attributes as a dictionary
        config_dict["config_class"] = self.__class__.__name__  # Add the class name
        # Handle Config subclass attributes
        for key, value in config_dict.items():
            if isinstance(value, Config):
                config_dict[key] = value.to_dict()  # Recursively convert Config subclasses
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create a configuration instance from a dictionary, verifying the class name."""
        if config_dict.get("config_class") != cls.__name__:
            raise ValueError(
                f"Invalid config class. Expected {cls.__name__}, but got {config_dict.get('config_class')}")

        # Remove the class name before passing to the constructor
        config_dict.pop("config_class")

        # Handle nested Config subclasses
        for key, value in config_dict.items():
            if isinstance(value, dict) and "config_class" in value:
                # Recursive call to from_dict for nested Config subclasses
                config_dict[key] = cls._get_subclass_from_dict(value)
        cfg = cls(**config_dict)
        cfg.validate()
        return cfg

    @classmethod
    def _get_subclass_from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Helper method to handle nested Config subclasses."""
        config_class_name = config_dict.get("config_class")
        # Dynamically find the subclass from class name
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

