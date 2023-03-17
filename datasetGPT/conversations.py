import itertools
from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Union, Generator, Iterator

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage


@dataclass
class ConversationsGeneratorConfig:
    openai_api_key: str
    """OpenAI API key."""
    agent1: str
    """Description of the first agent used to construct its system message."""
    agent2: str
    """Description of the second agent used to construct its system message."""
    num_samples: int = 1
    """Number of conversations to generate for each configuration."""
    interruption: str = "length"
    """Interruption mode."""
    end_phrase: str = "Goodbye!"
    """Phrase to look for when checking whether to interrupt a conversation."""
    end_agent: str = "both"
    """Agent whose messages to check for the interruption phrase."""
    lengths: List[int] = field(default_factory=lambda: [5])
    """Possible lengths of the conversations. If end_phrase interruption is enabled these will be used for maximum lengths."""
    temperatures: List[int] = field(default_factory=lambda: [0])
    """Possible temperatures for the backend LLM."""
    options: List[Tuple[str, str]] = field(default_factory=lambda: [])
    """Additional options defined in the system prompts with curly brackets."""


class ConversationsGenerator:
    """Dataset generator for a given conversation configuration."""

    config: ConversationsGeneratorConfig
    """Dataset configuration to use."""
    conversation_configs: List[Dict[str, Any]]
    """Possible configurations for each conversation."""
    generator_index: int = 0
    """Index of the conversation config to use in the next iteration."""

    def __init__(self, config: ConversationsGeneratorConfig) -> None:
        """Initializes ConversationsGenerator"""
        self.config = config
        self.initialize_conversation_configs()

    def initialize_conversation_configs(self) -> None:
        """Initialize all possible conversation configurations."""
        options_keys = ["length", "temperature", "sample_id"]
        options_values = [self.config.lengths,
                          self.config.temperatures,
                          range(self.config.num_samples)]

        for option in self.config.options:
            if option[0] not in options_keys:
                options_keys.append(option[0])
                options_values.append([option[1]])
            else:
                index = options_keys.index(option[0])
                if option[1] not in options_values[index]:
                    options_values[index].append(option[1])

        self.conversation_configs = list(map(lambda x: dict(zip(options_keys, x)),
                                        itertools.product(*options_values)))

    def initialize_chain(
        self,
        agent: str,
        system_prompt: str,
        conversation_config: Dict[str, Any]
    ) -> Tuple[ConversationChain, str]:
        """Initialize a conversation and return a chain and a formatted system prompt."""
        if self.config.interruption == "end_phrase":
            if self.config.end_agent == agent or self.config.end_agent == "both":
                system_prompt += f" When the whole conversation is over end with \"{self.config.end_phrase}\"."

        system_template = SystemMessagePromptTemplate.from_template(
            system_prompt)
        template_params = {key: conversation_config[key]
                           for key in system_template.input_variables}
        system_message = system_template.format(**template_params).content

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="history"),
            HumanMessagePromptTemplate.from_template("{input}")
        ])

        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatOpenAI(temperature=conversation_config["temperature"],
                         openai_api_key=self.config.openai_api_key)
        chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)

        return chain, system_message

    def check_iterruption(self, agent: str, message: str) -> bool:
        """Check whether to interrupt conversation generation."""
        if self.config.interruption == "end_phrase":
            if self.config.end_agent == agent or self.config.end_agent == "both":
                if self.config.end_phrase in message:
                    raise StopIteration()

    def generate_conversation(
        self,
        initial_utterance: str = "Hello!"
    ) -> Dict[str, Union[List[List[Any]], float, int]]:
        """Run two chains to talk with one another and record the chat history."""
        if self.generator_index >= len(self.conversation_configs):
            raise StopIteration()

        conversation_config = self.conversation_configs[self.generator_index]
        self.generator_index += 1

        chain1, system_prompt1 = self.initialize_chain("agent1",
                                                       self.config.agent1,
                                                       conversation_config)

        chain2, system_prompt2 = self.initialize_chain("agent2",
                                                       self.config.agent2,
                                                       conversation_config)

        utterances = []

        chain1_inp = initial_utterance
        for _ in range(conversation_config["length"]):
            chain1_out = chain1.predict(input=chain1_inp)
            utterances.append(["agent1", chain1_out])
            self.check_iterruption("agent1", chain1_out)

            chain2_out = chain2.predict(input=chain1_out)
            utterances.append(["agent2", chain2_out])
            self.check_iterruption("agent2", chain2_out)
            chain1_inp = chain2_out

        return {**conversation_config,
                "agent1": system_prompt1,
                "agent2": system_prompt2,
                "utterances": utterances}

    def __next__(self) -> Generator[Dict[str, Any], None, None]:
        """Generate a conversation."""
        return self.generate_conversation()
        
    def __iter__(self) -> Iterator:
        return self 
