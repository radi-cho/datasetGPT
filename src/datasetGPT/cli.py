import click
from typing import List, Tuple

from .conversations import ConversationsGeneratorConfig, ConversationsGenerator
from .texts import TextsGeneratorConfig, TextsGenerator
from .outputs import DatasetWriter


@click.group()
def datasetGPT() -> None:
    """Command line interface that generates datasets with LLMs."""
    pass


click_options = click.option("--option",
                             "-o",
                             "options",
                             type=(str, str),
                             multiple=True,
                             help="Values for additional options denoted in your prompts by {OPTION_NAME}.")

click_path = click.option("--path",
                          "-f",
                          "path",
                          type=click.Path(),
                          help="Where to save the dataset. Either a file or a directory (folder).")

click_single_file = click.option("--single-file",
                                 "-s",
                                 "single_file",
                                 type=bool,
                                 is_flag=True,
                                 help="Either save the whole dataset to a single file or create multiple files.")

click_num_samples = click.option("--num-samples",
                                 "-n",
                                 "num_samples",
                                 type=int,
                                 default=1,
                                 help="Number of conversations for each configuration.")

click_temperatures = click.option("--temperature",
                                  "-t",
                                  "temperatures",
                                  type=float,
                                  multiple=True,
                                  default=[0.5],
                                  help="Possible temperature values for the backend language model.")


@click.command()
@click.option("--openai-api-key",
              "-k",
              "openai_api_key",
              type=str,
              envvar="OPENAI_API_KEY",
              help="OpenAI API key.")
@click.option("--agent1",
              "-a",
              "agent1",
              type=str,
              required=True,
              help="Agent role description.")
@click.option("--agent2",
              "-b",
              "agent2",
              type=str,
              required=True,
              help="Agent role description.")
@click.option("--initial-utterance",
              "-u",
              "initial_utterances",
              type=str,
              default=["Hello!"],
              multiple=True,
              help="Utterance to be provisioned to the first agent. For many use cases a \"Hello\" is enough.")
@click.option("--interruption",
              "-i",
              "interruption",
              type=click.Choice(["length", "end_phrase"]),
              default="length",
              help="Interruption mode.")
@click.option("--end-phrase",
              "-e",
              "end_phrase",
              type=str,
              default="Goodbye",
              help="Interrupt after this phrase is outputted by one of the agents.")
@click.option("--end-agent",
              "-d",
              "end_agent",
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
@click_temperatures
@click_num_samples
@click_options
@click_path
@click_single_file
def conversations(
    openai_api_key: str,
    agent1: str,
    agent2: str,
    initial_utterances: List[str],
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
    dataset_writer = DatasetWriter(path, single_file)

    generator_config = ConversationsGeneratorConfig(openai_api_key=openai_api_key,
                                                    agent1=agent1,
                                                    agent2=agent2,
                                                    initial_utterances=initial_utterances,
                                                    num_samples=num_samples,
                                                    interruption=interruption,
                                                    end_phrase=end_phrase,
                                                    end_agent=end_agent,
                                                    lengths=lengths,
                                                    temperatures=temperatures,
                                                    options=options)

    conversations_generator = ConversationsGenerator(generator_config)

    for conversation in conversations_generator:
        dataset_writer.save_intermediate_result(conversation)


@click.command()
@click.option("--prompt",
              "-p",
              "prompt",
              type=str,
              required=True,
              help="Input prompt.")
@click.option("--backend",
              "-b",
              "backends",
              type=str,
              multiple=True,
              default=["openai|text-davinci-003"],
              help="LLM APIs to use as backends. Use \"backend|model_name\" notation. For example: \"openai|text-davinci-003\".")
@click.option("--max-length",
              "-l",
              "max_lengths",
              type=int,
              multiple=True,
              default=[100],
              help="Maximum number of tokens to generate for each prompt.")
@click_temperatures
@click_num_samples
@click_options
@click_path
@click_single_file
def texts(
    prompt: str,
    num_samples: int,
    max_lengths: List[int],
    temperatures: List[int],
    backends: List[str],
    options: List[Tuple[str, str]],
    path: str,
    single_file: bool
) -> None:
    """Inference multiple LLMs at scale."""
    dataset_writer = DatasetWriter(path, single_file)

    generator_config = TextsGeneratorConfig(prompt=prompt,
                                            backends=backends,
                                            num_samples=num_samples,
                                            max_lengths=max_lengths,
                                            temperatures=temperatures,
                                            options=options)

    texts_generator = TextsGenerator(generator_config)

    for text_object in texts_generator:
        dataset_writer.save_intermediate_result(text_object)


datasetGPT.add_command(texts)
datasetGPT.add_command(conversations)


def main() -> None:
    """Run the datasetGPT CLI."""
    datasetGPT()
