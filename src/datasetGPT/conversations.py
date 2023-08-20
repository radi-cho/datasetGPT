from dataclasses import dataclass, field
from typing import List, Any, Dict, Tuple, Union

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

from .base import DatasetGenerator

OPTIONS_CONFIG_KEYS = ["length", "temperature", "initial_utterance"]
GENERATOR_CONFIG_KEYS = ["lengths", "temperatures", "initial_utterances"]


@dataclass
class ConversationsGeneratorConfig:
    openai_api_key: str
    """OpenAI API key."""
    agent1: str
    """Description of the first agent used to construct its system message."""
    agent2: str
    """Description of the second agent used to construct its system message."""
    initial_utterances: List[str] = field(default_factory=lambda: ["Hello."])
    """Utterances to be provisioned to the first agent."""
    num_samples: int = 1
    """Number of conversations to generate for each options combination."""
    interruption: str = "length"
    """Interruption mode."""
    end_phrase: str = "Goodbye!"
    """Phrase to look for when checking whether to interrupt a conversation."""
    end_agent: str = "both"
    """Agent whose messages to check for the interruption phrase."""
    lengths: List[int] = field(default_factory=lambda: [5])
    """Possible lengths of the conversations. If end_phrase interruption is enabled these will be used for maximum lengths."""
    temperatures: List[float] = field(default_factory=lambda: [0])
    """Possible temperatures for the backend LLM."""
    options: List[Tuple[str, str]] = field(default_factory=lambda: [])
    """Additional options defined in the system prompts with curly brackets."""
    model: str = "gpt-3.5-turbo"
    """Model to select for both agents"""
    model_agent_one: str = "gpt-3.5-turbo"
    """Model to select for agent1"""
    model_agent_two: str = "gpt-3.5-turbo"
    """Model to select for agent2"""


class ConversationsGenerator(DatasetGenerator):
    """Generator producing conversations between two AI agents."""

    config: ConversationsGeneratorConfig
    """Configuration for a ConversationsGenerator."""

    def __init__(self, config: ConversationsGeneratorConfig) -> None:
        """Initialize ConversationsGenerator."""
        super().__init__(config)

    def initialize_options_configs(
        self,
        options_config_keys: List[str] = OPTIONS_CONFIG_KEYS,
        generator_config_keys: List[str] = GENERATOR_CONFIG_KEYS
    ) -> None:
        """Prepare options combinations."""
        print(self.config.initial_utterances)
        super().initialize_options_configs(options_config_keys, generator_config_keys)

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

        # Select model for each agent. Only if specific model for both agents is provided, value will be used.
        model_for_llm = self.config.model
        if(self.config.model_agent_one and self.config.model_agent_one):
            if(agent == "agent1"):
                model_for_llm = self.config.model_agent_one
            elif(agent == "agent2"):
                model_for_llm = self.config.model_agent_two

        memory = ConversationBufferMemory(return_messages=True)
        llm = ChatOpenAI(temperature=conversation_config["temperature"],
                         openai_api_key=self.config.openai_api_key, model=model_for_llm)
        chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)

        return chain, system_message

    def end_phrase_interruption(self, agent: str, message: str) -> bool:
        """Check whether to interrupt conversation generation."""
        if self.config.interruption == "end_phrase":
            if self.config.end_agent == agent or self.config.end_agent == "both":
                if self.config.end_phrase in message:
                    return True

        return False

    def generate_item(self) -> Dict[str, Union[List[List[Any]], float, int]]:
        """Run two chains to talk with one another and record the chat history."""
        if self.generator_index >= len(self.options_configs):
            raise StopIteration()

        conversation_config = self.options_configs[self.generator_index]
        self.generator_index += 1

        chain1, system_prompt1 = self.initialize_chain("agent1",
                                                       self.config.agent1,
                                                       conversation_config)

        chain2, system_prompt2 = self.initialize_chain("agent2",
                                                       self.config.agent2,
                                                       conversation_config)

        utterances = []

        chain1_inp = conversation_config["initial_utterance"]
        for _ in range(conversation_config["length"]):
            chain1_out = chain1.predict(input=chain1_inp)
            utterances.append(["agent1", chain1_out])

            if self.end_phrase_interruption("agent1", chain1_out):
                break

            chain2_out = chain2.predict(input=chain1_out)
            utterances.append(["agent2", chain2_out])

            if self.end_phrase_interruption("agent2", chain2_out):
                break

            chain1_inp = chain2_out

        return {**conversation_config,
                "agent1": system_prompt1,
                "agent2": system_prompt2,
                "utterances": utterances}
