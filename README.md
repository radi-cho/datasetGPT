# datasetGPT

`datasetGPT` is a Python library and a CLI for inferencing the OpenAI API to generate textual datasets.

Possible use cases may include:

- Constructing textual corpora to train/fine-tune detectors for content written by AI.
- Collecting datasets of LLM-produced conversations for research purposes, analysis of AI performance/impact/ethics, etc.
- Automating a task that a LLM can handle over big amounts of input texts. For example, using GPT-3 to summarize 1000 paragraphs with a single CLI command.
- Leveraging the ChatGPT (and soon [GPT-4](https://github.com/radi-cho/awesome-gpt4)) API to produce diverse texts for a specific task and then fine-tune smaller models such as GPT-3 Ada to handle them.

> This tool is distributed freely and doesn't imply any restrictions on the downstream use cases.
> However, you should make sure to follow the [OpenAI Terms of use](https://openai.com/policies/terms-of-use) in your specific context.

## Installation

Under active development. Coming to `PyPI` soon. Currently you can run `datasetGPT` by cloning this repository as described [here](#contributing).

## Usage examples

### Generate conversations with the ChatGPT API

Start by setting the `OPENAI_API_KEY` environment variable. Alternatively, you can pass it as a parameter to [the config](https://github.com/radi-cho/datasetGPT/blob/main/datasetGPT/conversations.py#L20). Then use our [`ConversationsGenerator`](https://github.com/radi-cho/datasetGPT/blob/main/datasetGPT/conversations.py#L43) to produce texts with the `gpt-3.5-turbo` API. [`DatasetWriter`](https://github.com/radi-cho/datasetGPT/blob/main/datasetGPT/outputs.py#L8) can save each dataset item to a separate JSON file or combine all outputs in a single file.

```python
from datasetGPT import ConversationsGenerator, ConversationsGeneratorConfig, DatasetWriter

dataset_writer = DatasetWriter() # single_file=True

generator_config = ConversationsGeneratorConfig(agent1="You're a shop assistant in a pet store. Answer to customer questions politely.",
                                                agent2="You're a customer in a pet store. You should behave like a human. You want to buy {n} pets. Ask questions about the pets in the store.",
                                                num_samples=2,
                                                interruption="length",
                                                lengths=[4, 5],
                                                temperatures=[0.1, 0.2],
                                                options=[("n", "2"), ("n", "3")])

conversations_generator = ConversationsGenerator(generator_config)

for conversation in conversations_generator:
    dataset_writer.save_intermediate_result(conversation)
```

This should produce a dataset directory with 16 conversations saved as JSON files. Why 16? Because `num_samples` conversations are generated for each possible combination of parameters (conversation length, LLM temperature, and custom prompt options). A dataset item looks like this:

```json
{
    "length": 5,
    "temperature": 0.1,
    "n": "2",
    "agent1": "You're a shop assistant in a pet store. Answer to customer questions politely. When the whole conversation is over end with \"Goodbye\".",
    "agent2": "You're a customer in a pet store. You should behave like a human. You want to buy 2 pets. Ask questions about the pets in the store. When the whole conversation is over end with \"Goodbye\".",
    "utterances": [
        [
            "agent1",
            "Hello! How can I assist you today?"
        ],
        [
            "agent2",
            "Hi! I'm interested in buying two pets. Can you tell me what kind of pets you have available in the store?"
        ],
        [
            "agent1",
            "Certainly! We have a variety of pets available, including dogs, cats, birds, fish, hamsters, guinea pigs, rabbits, and reptiles. Is there a specific type of pet you're interested in?"
        ],
        [
            "agent2",
            "I'm not sure yet. Can you tell me more about the dogs and cats you have available? What breeds do you have?"
        ],
        ...
    ]
}
```

The same could be achieved through the command-line interface:
```bash
datasetGPT conversations \
    --length 4 \
    --length 5 \
    --agent1 "You're a shop assistant in a pet store. Answer to customer questions politely." \
    --agent2 "You're a customer in a pet store. You should behave like a human. You want to buy {n} pets. Ask questions about the pets in the store." \
    --temperature 0.1 \
    --temperature 0.2 \
    --option n 2 \
    --option n 3 \
    --path dataset
```

## CLI Reference

```
Usage: datasetGPT conversations [OPTIONS]

  Produce conversations between two gpt-3.5-turbo agents with given roles.

Options:
  -k, --openai-api-key TEXT       OpenAI API key.
  -a, --agent1 TEXT               Agent role description.  [required]
  -b, --agent2 TEXT               Agent role description.  [required]
  -n, --num-samples INTEGER       Number of conversations for each
                                  configuration.
  -i, --interruption [length|end_phrase]
                                  Interruption mode.
  -e, --end-phrase TEXT           Interrupt after this phrase is outputted by
                                  one of the agents.
  -d, --end-agent [agent1|agent2|both]
                                  In which agent's messages to look for the
                                  end phrase.
  -l, --length INTEGER            Maximum number of utterances for each agent.
                                  A conversation sample will be generated for
                                  each length.
  -t, --temperature FLOAT         Possible temperature values for the backend
                                  language model.
  -o, --option <TEXT TEXT>...     Values for additional options denoted in
                                  your agent description by {OPTION_NAME}.
  -p, --path PATH                 Where to save the dataset.
  -s, --single-file               Either save the whole dataset to a single
                                  file or create multiple files.
  --help                          Show this message and exit.
```

## Contributing

Contributions will be highly appreciated. Currently these features are under development:
- [x] `datasetGPT conversations` - Make two ChatGPT agents talk with one another and record the conversation history.
- [ ] `datasetGPT texts` - Inference the GPT-3 API with a given input prompt and generate multiple outputs by varying parameters.
- [ ] `datasetGPT transformations` - Apply a list of transformations to a list of texts. For example, summarizing a list of texts for a {child | university student | PhD candidate} to understand.

To set up a local development environment:

```bash
git clone https://github.com/radi-cho/datasetGPT/
cd datasetGPT
pip install -e .
```
