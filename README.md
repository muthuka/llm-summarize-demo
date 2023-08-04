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
