# llm-summarize-demo

Took Deeplearning.ai course and experimented these code snippets to focus on summarizing. These samples are for summarizing a dialog between 2 people.

* LLM Model: Google FLAN-T5 base
* Dataset: <https://huggingface.co/datasets/knkarthick/dialogsum>

Install dependencies with

```sh
pip install -r requirements.txt
```

Run it with

```sh
python3 <filename>
```

## Playground files

| File | Purpose |
| ---- | ------- |
| sample-dataset.py | Loading a dataset into memory |
| encode-decode.py | Tokenize & encode/decode |
| trainingdata.py | Show the data used for experiment 50 & 500 |
| zeroshot-summarize.py | Zero shot inference using "Summarize the following conversation: " |
| zeroshot-dialogue,py | Zero shot inference using "Dialogue:" |
| oneshot.py | One shot inference using "Dialogue:" |
| fewshot.py | Few shots inference using "Dialogue:" |
| fewshot-config.py | Configurable few shot inference |
| full-tine-tuning.py | Full fine-tuning of FLAN-T5 |
| peft-fine-tuning.py | PEFT fine-tuning of FLAN-T5 |

## Articles

* Part 1 - AI/LLM: <https://medium.com/@muthuka/quickies-part-1-ai-llm-5dbaf989e620>
* Part 2 - LLM/Transformer: <https://medium.com/@muthuka/quickies-part-2-llm-transformer-ea8f5e8456d5>
* Part 3 - Transformer/Prompt Engineering: <https://medium.com/@muthuka/quickies-part-3-transformer-prompt-engineering-9b68be365051>
* Generative AI Project - Part 1: <https://medium.com/@muthuka/generative-ai-project-part-1-12331118b353>
* Generative AI Project - Part 2: <https://medium.com/@muthuka/generative-ai-project-part-2-ce277f98aa21>
* Generative AI Project - Part 3: <https://medium.com/@muthuka/generative-ai-project-part-3-7c6793f15326>
