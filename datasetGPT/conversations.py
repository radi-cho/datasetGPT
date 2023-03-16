import itertools
from typing import List, Any, Dict, Generator, Tuple, Union

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

from openai.error import (
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError
)


def initialize_conversation_chain(
    openai_api_key: str,
    system_prompt: str,
    config: Dict[str, Any]
) -> Tuple[ConversationChain, str]:
    """Initialize a conversation and return a chain and a formatted system prompt."""
    system_template = SystemMessagePromptTemplate.from_template(system_prompt)
    system_config = {key: config[key]
                     for key in system_template.input_variables}
    system_content = system_template.format(**system_config).content

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatOpenAI(temperature=config["temperature"],
                     openai_api_key=openai_api_key)
    chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    return chain, system_content


def safe_predict(
    chain: ConversationChain,
    input: str,
    retrying: bool = False,
) -> None:
    """Get a prediction from the chain with a single retry on connection or rate limit errors."""
    try:
        return chain.predict(input=input)
    except (RateLimitError, APIConnectionError, ServiceUnavailableError) as e:
        print("An error has occurred!", e)

        if retrying != True:
            print("Retrying...")
            return safe_predict(chain, input, True)


def generate_conversation(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    interruption: str,
    end_phrase: str,
    end_agent: str,
    config: Dict[str, Any],
    utterance: str = "Hello!"
) -> Dict[str, Union[List[List[Any]], float, int]]:
    """Run two chains to talk with one another and record the chat history."""
    if interruption == "end_phrase":
        if end_agent == "agent1" or end_agent == "both":
            agent1 += f" When the whole conversation is over end with \"{end_phrase}\"."

        if end_agent == "agent2" or end_agent == "both":
            agent2 += f" When the whole conversation is over end with \"{end_phrase}\"."

    chain1, system_prompt1 = initialize_conversation_chain(
        openai_api_key=openai_api_key,
        system_prompt=agent1,
        config=config)

    chain2, system_prompt2 = initialize_conversation_chain(
        openai_api_key=openai_api_key,
        system_prompt=agent2,
        config=config)

    history = []

    for _ in range(config["length"]):
        chain1_out = safe_predict(chain1, utterance)
        history.append(["agent1", chain1_out])

        if interruption == "end_phrase" and end_agent != "agent2":
            if end_phrase in chain1_out:
                break

        chain2_out = safe_predict(chain2, chain1_out)
        history.append(["agent2", chain2_out])
        utterance = chain2_out

        if interruption == "end_phrase" and end_agent != "agent1":
            if end_phrase in chain2_out:
                break

    return {**config,
            "agent1": system_prompt1,
            "agent2": system_prompt2,
            "history": history}


def generate_conversations_dataset(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    num_samples: int = 1,
    interruption: str = "length",
    end_phrase: str = "Goodbye!",
    end_agent: str = "both",
    lengths: List[int] = [5],
    temperatures: List[int] = 0,
    options: List[Tuple[str, str]] = []
) -> Generator[Dict[str, Any], None, None]:
    """Iterate possible configurations and generate a conversation for each one."""
    options_keys = ["length", "temperature", "sample_id"]
    options_values = [lengths, temperatures, range(num_samples)]

    for option in options:
        if option[0] not in options_keys:
            options_keys.append(option[0])
            options_values.append([option[1]])
        else:
            index = options_keys.index(option[0])
            if option[1] not in options_values[index]:
                options_values[index].append(option[1])

    for config_values in itertools.product(*options_values):
        config = dict(zip(options_keys, config_values))

        yield generate_conversation(openai_api_key=openai_api_key,
                                    agent1=agent1,
                                    agent2=agent2,
                                    interruption=interruption,
                                    end_phrase=end_phrase,
                                    end_agent=end_agent,
                                    config=config)
