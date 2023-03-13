from typing import List, Any, Dict

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


def initialize_conversation_chain(
    openai_api_key: str,
    system_prompt: str,
    temperature: int
) -> ConversationChain:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationBufferMemory(return_messages=True)
    llm = ChatOpenAI(temperature=temperature, openai_api_key=openai_api_key)
    return ConversationChain(memory=memory, prompt=prompt, llm=llm)


def generate_conversation(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    interruption: str,
    end_phrase: str,
    end_agent: str,
    length: int = 5,
    temperature: int = 0,
    utterance: str = "Hello!"
) -> List[List[Any]]:

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
        chain1_out = chain1.predict(input=utterance)
        history.append(["agent1", chain1_out])

        if interruption == "end_phrase" and end_agent != "agent2":
            if end_phrase in chain1_out:
                break

        chain2_out = chain2.predict(input=chain1_out)
        history.append(["agent2", chain2_out])
        utterance = chain2_out

        if interruption == "end_phrase" and end_agent != "agent1":
            if end_phrase in chain2_out:
                break

    return history


def generate_conversations_dataset(
    openai_api_key: str,
    num_samples: int,
    interruption: str,
    end_phrase: str,
    end_agent: str,
    lengths: List[int],
    agent1: str,
    agent2: str
) -> List[Dict[str, Any]]:
    dataset = []

    for length in lengths:
        for _ in range(num_samples):
            utterances = generate_conversation(openai_api_key=openai_api_key,
                                               agent1=agent1,
                                               agent2=agent2,
                                               interruption=interruption,
                                               end_phrase=end_phrase,
                                               end_agent=end_agent,
                                               length=length)

            dataset.append({"length": length, "history": utterances})

    return dataset
