# datasetGPT

`datasetGPT` is a command-line interface and a Python library for inferencing Large Language Models to generate textual datasets.

Possible use cases may include:

- Constructing textual corpora to train/fine-tune detectors for content written by AI.
- Collecting datasets of LLM-produced conversations for research purposes, analysis of AI performance/impact/ethics, etc.
- Automating a task that a LLM can handle over big amounts of input texts. For example, using GPT-3 to summarize 1000 paragraphs with a single CLI command.
- Leveraging APIs of especially big LLMs to produce diverse texts for a specific task and then fine-tune a smaller model with them.

> This tool is distributed freely and doesn't imply any restrictions on the downstream use cases.
> However, you should make sure to follow the **Terms of use** of the backend APIs (OpenAI, Cohere, Petals, etc.) in your specific context.

## Installation

```
pip install datasetGPT
```

Most of the generation features rely on third-party APIs. Install their respective packages:

```
pip install openai cohere petals
```

## Usage examples

### Inference LLMs at scale

```bash
export OPENAI_API_KEY="..."
export COHERE_API_KEY="..."

datasetGPT texts \             
    --prompt "If {country} was a planet in the Star Wars universe it would be called" \
    --backend "openai|text-davinci-003" \
    --backend "cohere|medium" \
    --temperature 0.9 \
    --option country Germany \
    --option country France \
    --max-length 50 \
    --num-samples 1 \
    --single-file
```

The command above should produce a dataset file with 4 texts. Each possible combination of options is used for each of the backend LLMs. Check out the [CLI reference](#cli-reference) for more details. A dataset file looks like this:

```json
[
    {
        "sample_id": 0,
        "backend": "openai|text-davinci-003",
        "max_length": 50,
        "temperature": 0.9,
        "country": "Germany",
        "prompt": "If Germany was a planet in the Star Wars universe it would be called",
        "output": " Euron. The planet would be home to a powerful and diverse species of aliens, known as the Eurons, that have evolved to a higher level of understanding and technological advancement compared to many of the other planets in the galaxy. The planet would be"
    },
    {
        "sample_id": 0,
        "backend": "openai|text-davinci-003",
        "max_length": 50,
        "temperature": 0.9,
        "country": "France",
        "prompt": "If France was a planet in the Star Wars universe it would be called",
        "output": " The Empire of Liberty. It would be a peaceful, democratic planet with a strong sense of justice and equality. The planet would be home to many different species of aliens but the majority of its population would be humans. It would have a strong military and"
    },
    {
        "sample_id": 0,
        "backend": "cohere|medium",
        "max_length": 50,
        "temperature": 0.9,
        "country": "Germany",
        "prompt": "If Germany was a planet in the Star Wars universe it would be called",
        "output": " the Hoth of the universe.\nAfter the Soviet invasion of Eastern Europe and the subsequent Western anti-Soviet sentiment, Germany's arms manufacturers went into hyperdrive and the country churned out guns at a frightening pace. By the early 1930"
    },
    ...
]
```

Alternatively, you can use our [`TextsGenerator`](https://github.com/radi-cho/datasetGPT/blob/main/datasetGPT/texts.py#L27) to produce texts in Python.

### Generate conversations with the ChatGPT API

```bash
export OPENAI_API_KEY="..."
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

The command above should produce a dataset directory with 16 conversations saved as JSON files. You can specify if you want all of them to be saved in a single file. But why 16? Because `num_samples` dialogues are generated for each possible combination of parameters (conversation length, LLM temperature, and custom prompt options). A dataset item looks like this:

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

You can also use our [`ConversationsGenerator`](https://github.com/radi-cho/datasetGPT/blob/main/datasetGPT/conversations.py#L43) to produce texts with the `gpt-3.5-turbo` API programatically.

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

## Contributing

> Still under active development.

Contributions will be highly appreciated. Currently these features are under development:
- [x] `datasetGPT conversations` - Make two ChatGPT agents talk with one another and record the conversation history.
- [x] `datasetGPT texts` - Inference different LLMs with a given input prompt and generate multiple outputs by varying parameters.
- [ ] `datasetGPT transformations` - Apply a list of transformations to a list of texts. For example, summarizing a list of texts for a {child | university student | PhD candidate} to understand.
- [ ] Support more backend LLMs.

To set up a local development environment:

```bash
git clone https://github.com/radi-cho/datasetGPT/
cd datasetGPT
pip install -e .
```

## CLI Reference

```
datasetGPT [OPTIONS] COMMAND [ARGS]...

  Command line interface that generates datasets with LLMs.

Options:
  --help  Show this message and exit.

Commands:
  conversations  Produce conversations between two gpt-3.5-turbo agents...
  texts          Inference multiple LLMs at scale.
```

```
datasetGPT texts [OPTIONS]

  Inference multiple LLMs at scale.

Options:
  -p, --prompt TEXT            Input prompt.  [required]
  -b, --backend TEXT           LLM APIs to use as backends. Use
                               "backend|model_name" notation. For example:
                               "openai|text-davinci-003".
  -l, --max-length INTEGER     Maximum number of tokens to generate for each
                               prompt.
  -t, --temperature FLOAT      Possible temperature values for the backend
                               language model.
  -n, --num-samples INTEGER    Number of conversations for each configuration.
  -o, --option <TEXT TEXT>...  Values for additional options denoted in your
                               prompts by {OPTION_NAME}.
  -f, --path PATH              Where to save the dataset. Either a file or a
                               directory (folder).
  -s, --single-file            Either save the whole dataset to a single file
                               or create multiple files.
  --help                       Show this message and exit.
```

- You can specify multiple variants for the following options: `--length`, `--temperature`, `--num-samples`, `--option`. A dataset item will be generated for each possible combination of the supplied values.
- Each `--option` provided must be formatted as follows: `--option option_name "Some option value"`.
- Currently supported backends: GPT-3 model variants by [OpenAI](https://openai.com/blog/openai-api), the language models by [Cohere](https://pypi.org/project/cohere/), BLOOM through the [Petals API](https://petals.ml/).

```
datasetGPT conversations [OPTIONS]

  Produce conversations between two gpt-3.5-turbo agents with given roles.

Options:
  -k, --openai-api-key TEXT       OpenAI API key.
  -a, --agent1 TEXT               Agent role description.  [required]
  -b, --agent2 TEXT               Agent role description.  [required]
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
  -n, --num-samples INTEGER       Number of conversations for each
                                  configuration.
  -o, --option <TEXT TEXT>...     Values for additional options denoted in
                                  your prompts by {OPTION_NAME}.
  -f, --path PATH                 Where to save the dataset. Either a file or
                                  a directory (folder).
  -s, --single-file               Either save the whole dataset to a single
                                  file or create multiple files.
  --help                          Show this message and exit.
```

- The length parameter specifies how many utterances each agent should make. A length of 4 typically produces 8 utterances in total.
- You can specify either `length` (default) or `end_phrase` as an interruption strategy. When using `end_phrase` a conversation will be interrupted once the `--end-phrase` has appeared in the messages of the `--end-agent` (could be both). In this case, the lengths provided will be treated as maximum conversation lengths.
