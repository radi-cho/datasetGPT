import click
from typing import List, Any

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)

from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory


@click.command()
@click.option("--openai-api-key",
              "-k",
              type=str,
              envvar="OPENAI_API_KEY",
              help="OpenAI API key.")
@click.option("--num-samples",
              "-n",
              type=int,
              default=1,
              help="Number of conversations for each configuration.")
@click.option("--interruption",
              "-i",
              type=click.Choice(["length", "end_phrase"]),
              default="length",
              help="Interruption mode.")
@click.option("--end-phrase",
              "-e",
              type=str,
              default="Goodbye",
              help="Interrupt after this phrase is outputted by one of the agents.")
@click.option("--end-agent",
              "-i",
              type=click.Choice(["agent1", "agent2", "both"]),
              default="both",
              help="In which agent's messages to look for the end phrase.")
@click.option("--length",
              "-l",
              "lengths",
              type=int,
              multiple=True,
              default=[5],
              help="Maximum number of utterances for each agent. A conversation sample will be generated for each length.")
@click.option("--agent1",
              "-a1",
              type=str,
              required=True,
              help="Agent role description.")
@click.option("--agent2",
              "-a2",
              type=str,
              required=True,
              help="Agent role description.")
def conversations(
    openai_api_key: str,
    num_samples: int,
    interruption: str,
    end_phrase: str,
    end_agent: str,
    lengths: List[int],
    agent1: str,
    agent2: str
) -> None:
    """Produce conversations between two gpt-3.5-turbo agents with given roles."""

    for length in lengths:
        for _ in range(num_samples):
            history = generate_conversation(openai_api_key=openai_api_key,
                                            agent1=agent1,
                                            agent2=agent2,
                                            interruption=interruption,
                                            end_phrase=end_phrase,
                                            end_agent=end_agent,
                                            length=length)

            print(history)


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
