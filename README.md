# datasetGPT

Generate textual and conversational datasets with LLMs.

# Installation

```bash
git clone https://github.com/radi-cho/datasetGPT/
cd datasetGPT
pip install -e .
```

# Usage

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
