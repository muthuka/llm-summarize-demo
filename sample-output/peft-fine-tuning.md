# Output for PEFT fine tuning example

```text
Loading dataset knkarthick/dialogsum...
===================================================================================================
Found cached dataset csv (/Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 347.30it/s]

Loading model and tokenizer...
===================================================================================================
Number of trainable model parameters in original model:
=> 247577856 (100.00%) of 247577856

Zero-shot performance before fine-tuning for 200th row:
===================================================================================================
INPUT PROMPT:

Summarize the following conversation.

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

Summary:

---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.

---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
#Person1#: I'm thinking of upgrading my computer.

Dataset tokenization and batching...
===================================================================================================
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-6ef6e7c551cab368.arrow
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-e2262d61da55c409.arrow
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-1b9ac5f54e28fa2d.arrow
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-d731d0350d1f5672.arrow
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-800ae43dbcf4a0be.arrow
Loading cached processed dataset at /Users/muthu/.cache/huggingface/datasets/knkarthick___csv/knkarthick--dialogsum-c8fac5d84cd35861/0.0.0/6954658bab30a358235fa864b05cf819af0e179325c740e4bc853bcc7ec513e1/cache-9848a1ed4008a675.arrow
Shapes of the datasets:
Training: (125, 2)
Validation: (5, 2)
Test: (15, 2)

Fine-tuning the model with PEFT...
===================================================================================================
Fine-tuning start tine : 
2023-08-10 18:24:29
=> 3538944 (1.41%) of 251116800
/Library/Frameworks/Python.framework/Versions/3.11/lib/python3.11/site-packages/transformers/optimization.py:391: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
  warnings.warn(
{'loss': 49.5, 'learning_rate': 0.0, 'epoch': 0.06}                                                                                                                                                                                                                                                                                                                                                                              
{'train_runtime': 870.8345, 'train_samples_per_second': 0.009, 'train_steps_per_second': 0.001, 'train_loss': 49.5, 'epoch': 0.06}                                                                                                                                                                                                                                                                                               
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [14:30<00:00, 870.84s/it]
Fine-tuning end tine : 
2023-08-10 18:39:01
=> 3538944 (1.41%) of 251116800

Evaluating the model qualitatively...
===================================================================================================
PEFT MODEL: #Person2#: I'm not sure what I'm looking for. #Person2#: I'm not sure what exactly I'd like to upgrade. #Person1#: I'm not sure what exactly I'd like to upgrade. #Person2#: I'm not sure what I'd like to upgrade. #Person1#: I'd like to upgrade my computer. #Person2#: I'm not sure. #Person1#: I'm not sure. #Person2: I'm not sure. #Person1#: I'm not sure. #Person2#: I'm not sure. #Person1#: I'm not sure. #Person2#: I'm not sure. #Person1#: I'm not sure. #Person2#: I'm not sure. #

Evaluating the model quantitatively using ROUGE...
===================================================================================================
Generating summary for row 0...
Generating summary for row 1...
Generating summary for row 2...
Generating summary for row 3...
Generating summary for row 4...
Generating summary for row 5...
Generating summary for row 6...
Generating summary for row 7...
Generating summary for row 8...
Generating summary for row 9...

Comparing human vs original vs peft model summaries...

                            human_baseline_summaries                           original_model_summaries                               peft_model_summaries
0  Ms. Dawson helps #Person1# to write a memo to ...  #Person1#: Hello Ms. Dawson, I need you to tak...  The memo will be sent to all employees by this...
1  In order to prevent employees from wasting tim...  Employees will be able to communicate with eac...  - - - - - - - - - - - - - - - - - - - - - - - ...
2  Ms. Dawson takes a dictation for #Person1# abo...                              #Person1#: Thank you.  This memo is to be distributed to all employee...
3  #Person2# arrives late because of traffic jam....  The traffic jam is a problem for Person1 and P...  The driver is a bit concerned about the pollut...
4  #Person2# decides to follow #Person1#'s sugges...              The traffic in this city is terrible.                                The traffic is bad.
5  #Person2# complains to #Person1# about the tra...  #Pr1#: I'm finally here! I got stuck in traffi...  The person who got stuck in a traffic jam is n...
6  #Person1# tells Kate that Masha and Hero get d...               Masha and Hero are getting divorced.  #Person1#: I don't really know what happened. ...
7  #Person1# tells Kate that Masha and Hero are g...            The divorce is taking place in January.              The divorce is happening in New Year.
8  #Person1# and Kate talk about the divorce betw...   Apparently, Masha and Hero are getting divorced.  #Person2: Well, I always thought they were hav...
9  #Person1# and Brian are at the birthday party ...  #: #: #: #: #: #: #: #: #: #: #: #: #: #: #: #...  #Person1#: Happy birthday, Brian. #Person2: Yo...

Evaluating the model quantitatively using ROUGE...
===================================================================================================
ORIGINAL MODEL:
{'rouge1': 0.184122546626306, 'rouge2': 0.05726104559056504, 'rougeL': 0.15690924642052462, 'rougeLsum': 0.1573746595927047}
PEFT MODEL:
{'rouge1': 0.12913314460003766, 'rouge2': 0.03127460242481499, 'rougeL': 0.10981582981821433, 'rougeLsum': 0.1125756251495936}

Evaluating the model quantitatively using ROUGE on the full test set...
===================================================================================================
ORIGINAL MODEL:
{'rouge1': 0.2334158581572823, 'rouge2': 0.07603964187010573, 'rougeL': 0.20145520923859048, 'rougeLsum': 0.20145899339006135}
INSTRUCT MODEL:
{'rouge1': 0.42161291557556113, 'rouge2': 0.18035380596301792, 'rougeL': 0.3384439349963909, 'rougeLsum': 0.33835653595561666}
PEFT MODEL:
{'rouge1': 0.40810631575616746, 'rouge2': 0.1633255794568712, 'rougeL': 0.32507074586565354, 'rougeLsum': 0.3248950182867091}

Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
===================================================================================================
rouge1: 17.47%
rouge2: 8.73%
rougeL: 12.36%
rougeLsum: 12.34%

Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
===================================================================================================
rouge1: -1.35%
rouge2: -1.70%
rougeL: -1.34%
rougeLsum: -1.35%


```
