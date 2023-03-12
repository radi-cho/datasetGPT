# datasetGPT

Generate textual and conversational datasets with LLMs.

# Installation

This project is still under active development.

# Usage

```
Usage: datasetGPT conversations [OPTIONS]

  Produce conversations between two gpt-3.5-turbo agents with given roles.

Options:
  -k, --openai-api-key TEXT       OpenAI API key.
  -n, --num-samples INTEGER       Number of conversations for each
                                  configuration.
  -i, --interruption [length|end_phrase]
                                  Interruption mode.
  -e, --end-phrase TEXT           Interrupt after this phrase is outputted by
                                  one of the agents.
  -i, --end-agent [agent1|agent2|both]
                                  In which agent's messages to look for the
                                  end phrase.
  -l, --length INTEGER            Maximum number of utterances for each agent.
                                  A conversation sample will be generated for
                                  each length.
  -a1, --agent1 TEXT              Agent role description.  [required]
  -a2, --agent2 TEXT              Agent role description.  [required]
  --help                          Show this message and exit.
```
