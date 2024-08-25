from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType
# trl: Transformer Reinforcement Learning library
from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl import create_reference_model
from trl.core import LengthSampler
from common import build_dataset, print_model_params
from common import evaluate_toxicity, collator
import torch
import evaluate
import pandas as pd

# Check if MPS is available and set the device
# device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = "cpu"
print(f"Using device: {device}")

# tqdm library makes the loops show a smart progress meter.
tqdm.pandas()

# Load the dataset
model_name = "google/flan-t5-base"
huggingface_dataset_name = "knkarthick/dialogsum"
dataset_original = load_dataset(huggingface_dataset_name)
dataset = build_dataset(model_name=model_name,
                        dataset_name=huggingface_dataset_name,
                        input_min_text_length=200,
                        input_max_text_length=1000)


lora_config = LoraConfig(r=32,  # Rank
                         lora_alpha=32, target_modules=["q", "v"],
                         lora_dropout=0.05, bias="none",
                         task_type=TaskType.SEQ_2_SEQ_LM  # FLAN-T5
                         )

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move the model to the correct device
model.to(device)

peft_model = PeftModel.from_pretrained(
    model, './peft/', lora_config=lora_config, torch_dtype=torch.bfloat16,
    device_map="auto", is_trainable=True)
print(
    f'PEFT model params to be updated:\n{print_model_params(peft_model)}\n')

peft_model.to(device)

ppo_model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(
    peft_model, torch_dtype=torch.bfloat16, is_trainable=True)
ppo_model.to(device)
print(
    f'PPO model parameters to be updated (ValueHead + 769 params):\n{
        print_model_params(ppo_model)}\n')
print(ppo_model.v_head)

ref_model = create_reference_model(ppo_model)
print(
    f'Reference model parameters to be updated:\n{
        print_model_params(ref_model)}\n')
# ref_model.to(device)

# Prepare Reward Model
toxicity_model_name = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = AutoTokenizer.from_pretrained(
    toxicity_model_name, clean_up_tokenization_spaces=True)
toxicity_model = AutoModelForSequenceClassification.from_pretrained(
    toxicity_model_name)
print(toxicity_model.config.id2label)
toxicity_model.to("cpu")

non_toxic_text = "#Person 1# tells Tommy that he didn't like the movie."
toxicity_input_ids = toxicity_tokenizer(
    non_toxic_text, return_tensors="pt").input_ids
logits = toxicity_model(input_ids=toxicity_input_ids).logits
print(f'logits [not hate, hate]: {logits.tolist()[0]}')

# Print the probabilities for [not hate, hate]
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [not hate, hate]: {probabilities}')

# get the logits for "not hate" - this is the reward!
not_hate_index = 0
nothate_reward = (logits[:, not_hate_index]).tolist()
print(f'reward (high): {nothate_reward}')

toxic_text = """#Person 1# tells Tommy that the movie was terrible,
dumb and stupid."""

toxicity_input_ids = toxicity_tokenizer(
    toxic_text, return_tensors="pt").input_ids
logits = toxicity_model(toxicity_input_ids).logits
print(f'logits [not hate, hate]: {logits.tolist()[0]}')

# Print the probabilities for [not hate, hate]
probabilities = logits.softmax(dim=-1).tolist()[0]
print(f'probabilities [not hate, hate]: {probabilities}')

# Get the logits for "not hate" - this is the reward!
nothate_reward = (logits[:, not_hate_index]).tolist()
print(f'reward (low): {nothate_reward}')

sentiment_pipe = pipeline("sentiment-analysis",
                          model=toxicity_model_name,
                          device=device)
reward_logits_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "none",  # Set to "none" to retrieve raw logits.
    "batch_size": 16
}

reward_probabilities_kwargs = {
    "top_k": None,  # Return all scores.
    # Set to "softmax" to apply softmax and retrieve probabilities.
    "function_to_apply": "softmax",
    "batch_size": 16
}

print("Reward model output:")
print("For non-toxic text")
print(sentiment_pipe(non_toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(non_toxic_text, **reward_probabilities_kwargs))

print("For toxic text")
print(sentiment_pipe(toxic_text, **reward_logits_kwargs))
print(sentiment_pipe(toxic_text, **reward_probabilities_kwargs))

# Evaluate toxicity
toxicity_evaluator = evaluate.load("toxicity",
                                   toxicity_model_name,
                                   module_type="measurement",
                                   toxic_label="hate")

toxicity_score = toxicity_evaluator.compute(predictions=[
    non_toxic_text
])
print("Toxicity score for non-toxic text:")
print(toxicity_score["toxicity"])

toxicity_score = toxicity_evaluator.compute(predictions=[
    toxic_text
])
print("\nToxicity score for toxic text:")
print(toxicity_score["toxicity"])


tokenizer = AutoTokenizer.from_pretrained(
    model_name, device_map="auto", clean_up_tokenization_spaces=True)

mean_before_detoxification, std_before_detoxification = evaluate_toxicity(
    model=ref_model,
    toxicity_evaluator=toxicity_evaluator,
    tokenizer=tokenizer,
    dataset=dataset["test"],
    num_samples=10)
print(
    f'toxicity [mean, std] before detox: [{mean_before_detoxification}, {
        std_before_detoxification}]')

# Detoxify
test_data = [{"key1": "value1", "key2": "value2", "key3": "value3"}]
print(f'Collator input: {test_data}')
print(f'Collator output: {collator(test_data)}')

learning_rate = 1.41e-5
max_ppo_epochs = 1
mini_batch_size = 4
batch_size = 16

config = PPOConfig(
    model_name=model_name,
    learning_rate=learning_rate,
    ppo_epochs=max_ppo_epochs,
    mini_batch_size=mini_batch_size,
    batch_size=batch_size
)

ppo_trainer = PPOTrainer(config=config,
                         model=ppo_model,
                         ref_model=ref_model,
                         tokenizer=tokenizer,
                         dataset=dataset["train"],
                         data_collator=collator)

output_min_length = 100
output_max_length = 400
output_length_sampler = LengthSampler(output_min_length, output_max_length)

generation_kwargs = {
    "min_length": 5,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True
}

reward_kwargs = {
    "top_k": None,  # Return all scores.
    "function_to_apply": "none",  # You want the raw logits without softmax.
    "batch_size": 16
}

max_ppo_steps = 10

for step, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # Break when you reach max_steps.
    if step >= max_ppo_steps:
        break

    prompt_tensors = batch["input_ids"]

    # Get response from FLAN-T5/PEFT LLM.
    summary_tensors = []

    for prompt_tensor in prompt_tensors:
        max_new_tokens = output_length_sampler()

        generation_kwargs["max_new_tokens"] = max_new_tokens
        summary = ppo_trainer.generate(prompt_tensor, **generation_kwargs)

        summary_tensors.append(summary.squeeze()[-max_new_tokens:])

    # This needs to be called "response".
    batch["response"] = [tokenizer.decode(
        r.squeeze()) for r in summary_tensors]

    # Compute reward outputs.
    query_response_pairs = [q + r for q,
                            r in zip(batch["query"], batch["response"])]
    rewards = sentiment_pipe(query_response_pairs, **reward_kwargs)

    # You use the `nothate` item because this is the score for the
    # positive `nothate` class.
    reward_tensors = [torch.tensor(
        reward[not_hate_index]["score"]) for reward in rewards]

    # Run PPO step.
    stats = ppo_trainer.step(prompt_tensors, summary_tensors, reward_tensors)
    ppo_trainer.log_stats(stats, batch, reward_tensors)

    print(f'objective/kl: {stats["objective/kl"]}')
    print(f'ppo/returns/mean: {stats["ppo/returns/mean"]}')
    print(f'ppo/policy/advantages_mean: {stats["ppo/policy/advantages_mean"]}')
    print('-'.join('' for x in range(100)))


mean_after_detoxification, std_after_detoxification = evaluate_toxicity(
    model=ppo_model,
    toxicity_evaluator=toxicity_evaluator,
    tokenizer=tokenizer,
    dataset=dataset["test"],
    num_samples=10)
print(
    f"""toxicity [mean, std] after detox: [{mean_after_detoxification},
    {std_after_detoxification}]
    """)

mean_improvement = (mean_before_detoxification -
                    mean_after_detoxification) / mean_before_detoxification
std_improvement = (std_before_detoxification -
                   std_after_detoxification) / std_before_detoxification

print('Percentage improvement of toxicity score after detoxification:')
print(f'mean: {mean_improvement*100:.2f}%')
print(f'std: {std_improvement*100:.2f}%')

# Quantitative Evaluation
batch_size = 20
compare_results = {}

df_batch = dataset["test"][0:batch_size]

compare_results["query"] = df_batch["query"]
prompt_tensors = df_batch["input_ids"]

summary_tensors_ref = []
summary_tensors = []

# Get response from ppo and base model.
for i in tqdm(range(batch_size)):
    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len

    summary = ref_model.generate(
        input_ids=torch.as_tensor(
            prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors_ref.append(summary)

    summary = ppo_model.generate(
        input_ids=torch.as_tensor(
            prompt_tensors[i]).unsqueeze(dim=0).to(device),
        **generation_kwargs
    ).squeeze()[-gen_len:]
    summary_tensors.append(summary)

# Decode responses.
compare_results["response_before"] = [tokenizer.decode(
    summary_tensors_ref[i]) for i in range(batch_size)]
compare_results["response_after"] = [tokenizer.decode(
    summary_tensors[i]) for i in range(batch_size)]

# Sentiment analysis of query/response pairs before/after.
texts_before = [
    d + s for d, s in zip(compare_results["query"],
                          compare_results["response_before"])]
rewards_before = sentiment_pipe(texts_before, **reward_kwargs)
compare_results["reward_before"] = [reward[not_hate_index]["score"]
                                    for reward in rewards_before]

texts_after = [
    d + s for d, s in zip(compare_results["query"],
                          compare_results["response_after"])]
rewards_after = sentiment_pipe(texts_after, **reward_kwargs)
compare_results["reward_after"] = [reward[not_hate_index]["score"]
                                   for reward in rewards_after]

pd.set_option('display.max_colwidth', 500)
df_compare_results = pd.DataFrame(compare_results)
df_compare_results["reward_diff"] = df_compare_results['reward_after'] - \
    df_compare_results['reward_before']
df_compare_results_sorted = df_compare_results.sort_values(
    by=['reward_diff'], ascending=False).reset_index(drop=True)
df_compare_results_sorted
