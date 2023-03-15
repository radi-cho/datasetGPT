import click
from typing import List, Tuple
from .conversations import generate_conversations_dataset
from .outputs import OutputWriter


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
@click.option("--agent1",
              "-a",
              type=str,
              required=True,
              help="Agent role description.")
@click.option("--agent2",
              "-b",
              type=str,
              required=True,
              help="Agent role description.")
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
              "-d",
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
@click.option("--temperature",
              "-t",
              "temperatures",
              type=float,
              multiple=True,
              default=[0],
              help="Possible temperature values for the backend language model.")
@click.option("--option",
              "-o",
              "options",
              type=(str, str),
              multiple=True,
              help="Values for additional options denoted in your agent description by {OPTION_NAME}.")
@click.option("--path",
              "-p",
              type=click.Path(),
              help="Where to save the dataset.")
@click.option("--single-file",
              "-s",
              type=bool,
              is_flag=True,
              help="Either save the whole dataset to a single file or create multiple files.")
def conversations(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    num_samples: int,
    interruption: str,
    end_phrase: str,
    end_agent: str,
    lengths: List[int],
    temperatures: List[int],
    options: List[Tuple[str, str]],
    path: str,
    single_file: bool
) -> None:
    """Produce conversations between two gpt-3.5-turbo agents with given roles."""
    output_writer = OutputWriter(path, single_file)

    generator = generate_conversations_dataset(openai_api_key=openai_api_key,
                                               agent1=agent1,
                                               agent2=agent2,
                                               num_samples=num_samples,
                                               interruption=interruption,
                                               end_phrase=end_phrase,
                                               end_agent=end_agent,
                                               lengths=lengths,
                                               temperatures=temperatures,
                                               options=options)

    for result in generator:
        output_writer.save_intermediate_result(result)


datasetGPT.add_command(conversations)


def main() -> None:
    """Run the datasetGPT CLI."""
    datasetGPT()
