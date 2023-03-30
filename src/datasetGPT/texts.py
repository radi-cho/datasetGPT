from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Union

from langchain.prompts import PromptTemplate
from langchain.llms import BaseLLM
from langchain.chains import LLMChain

from .base import DatasetGenerator

OPTIONS_CONFIG_KEYS = ["backend", "max_length", "temperature"]
GENERATOR_CONFIG_KEYS =  ["backends", "max_lengths", "temperatures"]


@dataclass
class TextsGeneratorConfig:
    prompt: str
    """Text prompt."""
    backends: List[Tuple[str, str, str]]
    """LLM APIs to use as backends."""
    num_samples: int = 1
    """Number of texts to generate for each options combination."""
    max_lengths: List[int] = field(default_factory=lambda: [5])
    """Maximum lengths in tokens for the output of each generation."""
    temperatures: List[float] = field(default_factory=lambda: [0])
    """Possible temperatures for the backend LLM."""
    options: List[Tuple[str, str]] = field(default_factory=lambda: [])
    """Additional options defined in the system prompts with curly brackets."""


class TextsGenerator(DatasetGenerator):
    """Generator producing texts by varying model parameters and prompt options."""

    config: TextsGeneratorConfig
    """Configuration for a TextsGenerator."""

    def __init__(self, config: TextsGeneratorConfig) -> None:
        """Initialize TextsGenerator."""
        super().__init__(config)

    def initialize_options_configs(
        self,
        options_config_keys: List[str] = OPTIONS_CONFIG_KEYS,
        generator_config_keys: List[str] = GENERATOR_CONFIG_KEYS
    ) -> None:
        """Prepare options combinations."""
        super().initialize_options_configs(options_config_keys, generator_config_keys)

    def initialize_backend(self, text_config: Dict[str, Any]) -> BaseLLM:
        """Initialize a specific LLM."""
        backend_str = text_config["backend"]
        temperature = text_config["temperature"]
        max_length = text_config["max_length"]

        backend, model = backend_str.split("|")

        if backend.lower() == "openai":
            from langchain.llms import OpenAI
            llm = OpenAI(model_name=model,
                         temperature=temperature,
                         max_tokens=max_length)
        elif backend.lower() == "cohere":
            from langchain.llms import Cohere
            llm = Cohere(model=model,
                         temperature=temperature,
                         max_tokens=max_length)
        elif backend.lower() == "petals":
            from langchain.llms import Petals
            llm = Petals(model_name=model,
                         temperature=temperature,
                         max_new_tokens=max_length)
        else:
            raise ValueError("Cannot use the specified backend.")

        return llm

    def generate_item(self) -> Dict[str, Union[List[List[Any]], float, int]]:
        """Produce text with a LLM Chain."""
        if self.generator_index >= len(self.options_configs):
            raise StopIteration()

        text_config = self.options_configs[self.generator_index]
        self.generator_index += 1

        input_variables = text_config.keys() - ["sample_id",
                                                "backend",
                                                "temperature",
                                                "max_length"]

        prompt_template = PromptTemplate(template=self.config.prompt,
                                         input_variables=input_variables)

        llm = self.initialize_backend(text_config)

        prompt_params = {k: text_config[k] for k in input_variables}
        input_prompt = prompt_template.format(**prompt_params)

        chain = LLMChain(prompt=prompt_template, llm=llm)
        output = chain.predict(**prompt_params)

        return {**text_config,
                "prompt": input_prompt,
                "output": output}
