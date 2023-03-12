import click
from conversations import conversations


@click.group()
def datasetGPT() -> None:
    """Command line interface that generates datasets with LLMs."""
    pass


datasetGPT.add_command(conversations)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    datasetGPT()
