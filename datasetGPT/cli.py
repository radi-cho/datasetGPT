import click
from typing import List
from .conversations import generate_conversations_dataset


@click.group()
def datasetGPT() -> None:
    """Command line interface that generates datasets with LLMs."""
    pass


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
    dataset = generate_conversations_dataset(openai_api_key=openai_api_key,
                                             num_samples=num_samples,
                                             interruption=interruption,
                                             end_phrase=end_phrase,
                                             end_agent=end_agent,
                                             lengths=lengths,
                                             agent1=agent1,
                                             agent2=agent2)

    # TODO: Data saving
    print(dataset)


datasetGPT.add_command(conversations)


def main() -> None:
    datasetGPT()
