---
base_model: unsloth/Llama-3.2-3B-Instruct-bnb-4bit
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: comics35_Llama-3.2-3B-Instruct-bnb-4bit
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# comics35_Llama-3.2-3B-Instruct-bnb-4bit

This model is a fine-tuned version of [unsloth/Llama-3.2-3B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-bnb-4bit) on the comics dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 2
- mixed_precision_training: Native AMP

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.44.2
- Pytorch 2.4.1+cu121
- Datasets 2.21.0
- Tokenizers 0.19.1