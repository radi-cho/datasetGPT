import itertools
from typing import List, Any, Dict, Generator

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from openai.error import (
    RateLimitError,
    APIConnectionError,
    ServiceUnavailableError
)


def initialize_conversation_chain(
    openai_api_key: str,
    system_prompt: str,
    temperature: float
) -> ConversationChain:
    """Initialize a ConversationChain with a ChatPromptTemplate."""
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)
    return ConversationChain(memory=memory, prompt=prompt, llm=llm)


def safe_predict(chain: ConversationChain, input: str, retrying: bool = False):
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
    length: int = 5,
    temperature: float = .0,
    utterance: str = "Hello!"
) -> List[List[Any]]:
    """Run two chains to talk with one another and record the chat history."""
    if interruption == "end_phrase":
        if end_agent == "agent1" or end_agent == "both":
            agent1 += f" When the conversation is over end with \"{end_phrase}\"."

        if end_agent == "agent2" or end_agent == "both":
            agent2 += f" When the conversation is over end with \"{end_phrase}\"."

    chain1 = initialize_conversation_chain(openai_api_key=openai_api_key,
                                           system_prompt=agent1,
                                           temperature=temperature)

    chain2 = initialize_conversation_chain(openai_api_key=openai_api_key,
                                           system_prompt=agent2,
                                           temperature=temperature)

    history = []

    for _ in range(length):
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

    return history


def generate_conversations_dataset(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    num_samples: int = 1,
    interruption: str = "length",
    end_phrase: str = "Goodbye!",
    end_agent: str = "both",
    lengths: List[int] = [5],
    temperatures: List[int] = 0
) -> Generator[Dict[str, Any], None, None]:
    """Iterate possible configurations and generate a conversation for each one."""
    possible_configs = itertools.product(lengths,
                                         temperatures,
                                         range(num_samples))

    for length, temperature, i in possible_configs:
        utterances = generate_conversation(openai_api_key=openai_api_key,
                                           agent1=agent1,
                                           agent2=agent2,
                                           interruption=interruption,
                                           end_phrase=end_phrase,
                                           end_agent=end_agent,
                                           length=length,
                                           temperature=temperature)

        yield {"length": length, "temperature": temperature, "history": utterances}
