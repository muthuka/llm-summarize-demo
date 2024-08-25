from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# Check if MPS is available and set the device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the dataset
huggingface_dataset_name = "knkarthick/dialogsum"
dataset = load_dataset(huggingface_dataset_name)
# dataset

# Load the model and tokenizer
model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    # If not using mac, use bfloat16
    model_name, torch_dtype=torch.float32).to(device)

# Move the model to the correct device
original_model.to(device)

tokenizer = AutoTokenizer.from_pretrained(
    model_name, clean_up_tokenization_spaces=True)


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
        mls = f"""trainable model parameters: {trainable_model_params}\n
        all model parameters: {all_model_params}\n
        percentage of trainable model parameters:
            {100 * trainable_model_params / all_model_params:.2f}%
        """
    return mls


print(print_number_of_trainable_model_parameters(original_model))

# Before full fine-tuning, let's look at zero-shot performance
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt').to(device)
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')


def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue +
              end_prompt for dialogue in example["dialogue"]]

    op1 = tokenizer(
        prompt, padding="max_length",
        truncation=True, return_tensors="pt").to(device)
    example['input_ids'] = op1.input_ids

    op2 = tokenizer(
        example["summary"], padding="max_length",
        truncation=True, return_tensors="pt").to(device)
    example['labels'] = op2.input_ids

    return example


# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ['id', 'topic', 'dialogue', 'summary',])

# To optimize, we will load a subset of the data
tokenized_datasets = tokenized_datasets.filter(
    lambda example, index: index % 100 == 0, with_indices=True)

print("Shapes of the datasets:\n")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")
print(tokenized_datasets)

# Prepare full-fine-tuning
output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer(
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

# Uncomment below when you have a powerful GPU in a server environment.
# Otherwise, it will take days or may not even complete on a small laptop.
trainer.train()

# Evaluate the model qualitatively
index = 200
dialogue = dataset['test'][index]['dialogue']
human_baseline_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

original_model_outputs = original_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200,
                                                            num_beams=1))
original_model_text_output = tokenizer.decode(
    original_model_outputs[0], skip_special_tokens=True)

# Load instruct model
instruct_model = AutoModelForSeq2SeqLM.from_pretrained(
    # If not using mac, use bfloat16
    "./flan-dialogue-summary-checkpoint", torch_dtype=torch.float32).to(device)

instruct_model_outputs = instruct_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200,
                                                            num_beams=1))
instruct_model_text_output = tokenizer.decode(
    instruct_model_outputs[0], skip_special_tokens=True)

print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{human_baseline_summary}')
print(dash_line)
print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'INSTRUCT MODEL:\n{instruct_model_text_output}')

# Evaluate the model quantitatively
rouge = evaluate.load('rouge')
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []

for _, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    original_model_outputs = original_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(
            max_new_tokens=200))
    original_model_text_output = tokenizer.decode(
        original_model_outputs[0], skip_special_tokens=True)
    original_model_summaries.append(original_model_text_output)

    instruct_model_outputs = instruct_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(
            max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(
        instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries,
                        original_model_summaries, instruct_model_summaries))

df = pd.DataFrame(zipped_summaries, columns=[
    'human_baseline_summaries', 'original_model_summaries',
    'instruct_model_summaries'])
df

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)

# The file `data/dialogue-summary-training-results.csv` contains
# a pre-populated list of all model results
# which you can use to evaluate on a larger section of data.
# Let's do that for each of the models:

results = pd.read_csv("data/ds-training-results.csv")

human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)

print("Absolute percentage improvement of INSTRUCT MODEL over ORIGINAL MODEL")

improvement = (np.array(list(instruct_model_results.values())) -
               np.array(list(original_model_results.values())))
for key, value in zip(instruct_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
