from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np
import datetime

# Check PyTorch has access to MPS (Metal Performance Shader, Apple's GPU architecture)
print(
    f"Is MPS (Metal Performance Shader) built? {torch.backends.mps.is_built()}")
print(f"Is MPS available? {torch.backends.mps.is_available()}")

# Set the device
device = "cpu"

if torch.backends.mps.is_available():
    # Initialize the device
    device = "mps"
elif torch.cuda.is_available():
    # Initialize the device
    device = "cuda"

print(f"Using device: {device}")

# Common things
dash_line = '-'.join('' for x in range(100))
equal_line = '='.join('' for x in range(100))


def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"=> {trainable_model_params} ({100 * trainable_model_params / all_model_params:.2f}%) of {all_model_params}"


def tokenize_function(example):
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue +
              end_prompt for dialogue in example["dialogue"]]
    example['input_ids'] = tokenizer(
        prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    example['labels'] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)

    return example


# Load the dataset
huggingface_dataset_name = "knkarthick/dialogsum"
print(f"\nLoading dataset {huggingface_dataset_name}...")
print(equal_line)
dataset = load_dataset(huggingface_dataset_name)

# Load the model and tokenizer
print("\nLoading model and tokenizer...")
print(equal_line)
model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name, torch_dtype=torch.float32).to(device)  # If not using mac, use bfloat16
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Number of trainable model parameters in original model:")
print(print_number_of_trainable_model_parameters(original_model))

# Before fine-tuning, let's look at zero-shot performance
index = 200  # Example we chose to look at
print(f"\nZero-shot performance before fine-tuning for {index}th row:")
print(equal_line)
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

print(f'INPUT PROMPT:\n{prompt}')
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(dash_line)
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

print("\nDataset tokenization and batching...")
print(equal_line)
# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(
    ['id', 'topic', 'dialogue', 'summary',])
tokenized_datasets = tokenized_datasets.filter(
    lambda example, index: index % 100 == 0, with_indices=True)

print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")
# print(tokenized_datasets)

# Fine-tuning the model if you have a cached version
print("\nFine-tuning the model with PEFT...")
print(equal_line)

now = datetime.datetime.now()
print(f"Fine-tuning start tine : ")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# from peft import PeftModel, PeftConfig

# peft_model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# peft_model = PeftModel.from_pretrained(peft_model_base,
#                                        './peft-dialogue-summary-checkpoint-from-s3/',
#                                        torch_dtype=torch.bfloat16,
#                                        is_trainable=False)
# print("Number of trainable model parameters in trained PEFT model:")

# Define the model PEFT config

lora_config = LoraConfig(
    r=32,  # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
)

peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))

# Define the training arguments
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,  # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1
)

peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

# PEFT is relatively faster. Based on the model size, you choose to train and save it offline
# Otherwise, it will take days or may not even complete on a small laptop.
peft_trainer.train()
peft_model_path = "./peft-dialogue-summary-checkpoint-local"
# Cache it if you want to use it later.
peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

print(f"Fine-tuning end tine : ")
print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

print(print_number_of_trainable_model_parameters(peft_model))

# Evaluate the model qualitatively
print("\nEvaluating the model qualitatively...")
print(equal_line)
dialogue = dataset['test'][index]['dialogue']
baseline_human_summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

original_model_outputs = original_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
original_model_text_output = tokenizer.decode(
    original_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(
    input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200, num_beams=1))
peft_model_text_output = tokenizer.decode(
    peft_model_outputs[0], skip_special_tokens=True)

print(f'PEFT MODEL: {peft_model_text_output}')

# Evaluate the model quantitatively
total = 10
print("\nEvaluating the model quantitatively using ROUGE...")
print(equal_line)
dialogues = dataset['test'][0:total]['dialogue']
human_baseline_summaries = dataset['test'][0:total]['summary']

original_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """

    print(f"Generating summary for row {idx}...")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    human_baseline_text_output = human_baseline_summaries[idx]

    original_model_outputs = original_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(
        original_model_outputs[0], skip_special_tokens=True)

    peft_model_outputs = peft_model.generate(
        input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(
        peft_model_outputs[0], skip_special_tokens=True)

    original_model_summaries.append(original_model_text_output)
    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(
    zip(human_baseline_summaries, original_model_summaries, peft_model_summaries))

print("\nComparing human vs original vs peft model summaries...\n")
df = pd.DataFrame(zipped_summaries, columns=[
                  'human_baseline_summaries', 'original_model_summaries', 'peft_model_summaries'])
print(df)


print("\nEvaluating the model quantitatively using ROUGE...")
print(equal_line)
rouge = evaluate.load('rouge')
human_baseline_summaries = dataset['test'][0:total]['summary']

original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('PEFT MODEL:')
print(peft_model_results)

# check performance on the full test set
print("\nEvaluating the model quantitatively using ROUGE on the full test set...")
print(equal_line)
results = pd.read_csv("data/ds-training-results.csv")

human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
peft_model_summaries = results['peft_model_summaries'].values

human_baseline_summaries = results['human_baseline_summaries'].values
original_model_summaries = results['original_model_summaries'].values
instruct_model_summaries = results['instruct_model_summaries'].values
peft_model_summaries = results['peft_model_summaries'].values

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

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)
print('PEFT MODEL:')
print(peft_model_results)

print("\nAbsolute percentage improvement of PEFT MODEL over ORIGINAL MODEL")
print(equal_line)

improvement = (np.array(list(peft_model_results.values())) -
               np.array(list(original_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')

print("\nAbsolute percentage improvement of PEFT MODEL over INSTRUCT MODEL")
print(equal_line)

improvement = (np.array(list(peft_model_results.values())) -
               np.array(list(instruct_model_results.values())))
for key, value in zip(peft_model_results.keys(), improvement):
    print(f'{key}: {value*100:.2f}%')
