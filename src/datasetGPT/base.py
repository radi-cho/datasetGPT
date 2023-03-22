import itertools
from typing import List, Any, Dict, Tuple, Generator, Iterator, Protocol

OPTIONS_CONFIG_KEYS = ["temperature"]
GENERATOR_CONFIG_KEYS =  ["temperatures"]


class DatasetGeneratorConfig(Protocol):
    """Base generator configuration protocol."""
    openai_api_key: str
    """OpenAI API key."""
    num_samples: int
    """Number of texts to generate for each options combination."""
    options: List[Tuple[str, str]]
    """Additional options defined in the text prompt with curly brackets."""


class DatasetGenerator:
    """Abstraction of a dataset generator."""

    config: DatasetGeneratorConfig
    """Generator configuration."""
    options_configs: List[Dict[str, Any]]
    """Possible combinations of the provided options."""
    generator_index: int = 0
    """Index of the next item to be returned by the generator."""

    def __init__(self, config: DatasetGeneratorConfig) -> None:
        self.config = config
        self.initialize_options_configs()

    def initialize_options_configs(
        self,
        options_config_keys: List[str] = OPTIONS_CONFIG_KEYS,
        generator_config_keys: List[str] = GENERATOR_CONFIG_KEYS
    ) -> None:
        """Prepare options combinations."""
        options_keys = ["sample_id", *options_config_keys]
        options_values = [range(self.config.num_samples)]
        options_values += [getattr(self.config, key) for key in generator_config_keys]

        for option in self.config.options:
            if option[0] not in options_keys:
                options_keys.append(option[0])
                options_values.append([option[1]])
            else:
                index = options_keys.index(option[0])
                if option[1] not in options_values[index]:
                    options_values[index].append(option[1])

        self.options_configs = list(map(lambda x: dict(zip(options_keys, x)),
                                        itertools.product(*options_values)))

    def generate_item(self) -> Dict[str, Any]:
        """Produce a data item."""
        return {}

    def __next__(self) -> Generator[Dict[str, Any], None, None]:
        return self.generate_item()

    def __iter__(self) -> Iterator:
        return self
