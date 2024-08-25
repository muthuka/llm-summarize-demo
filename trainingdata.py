from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
import torch

# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)

example_indices = [50, 500]
dash_line = '-'.join('' for x in range(100))

model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move the model to the correct device
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, use_fast=True, clean_up_tokenization_spaces=True)

sentence = "What time is it, Tom?"
sentence_encoded = tokenizer(sentence, return_tensors='pt').to(device)
sentence_decoded = tokenizer.decode(
    sentence_encoded["input_ids"][0],
    skip_special_tokens=True
)

print('ENCODED SENTENCE:')
print(sentence_encoded["input_ids"][0])
print('\nDECODED SENTENCE:')
print(sentence_decoded)

for i, index in enumerate(example_indices):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt').to(device)
    generated_output = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
    )

    output = tokenizer.decode(
        generated_output[0],
        skip_special_tokens=True
    )

    print(f'MODEL GENERATION - WITH PROMPT ENGINEERING:\n{output}\n')
    print('Example ', i + 1)
    print(dash_line)
    print(f'INPUT PROMPT:\n{dialogue}')
    print(dash_line)
    print(f'BASELINE HUMAN SUMMARY:\n{summary}')
    print(dash_line)
    print(f'MODEL GENERATION - WITHOUT PROMPT ENGINEERING:\n{output}\n')
