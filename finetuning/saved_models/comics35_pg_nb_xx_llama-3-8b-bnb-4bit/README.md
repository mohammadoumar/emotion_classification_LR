---
base_model: unsloth/llama-3-8b-bnb-4bit
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: comics35_pg_nb_xx_llama-3-8b-bnb-4bit
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# comics35_pg_nb_xx_llama-3-8b-bnb-4bit

This model is a fine-tuned version of [unsloth/llama-3-8b-bnb-4bit](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit) on the comics dataset.
It achieves the following results on the evaluation set:
- Loss: 0.4311

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
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 8
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 0.4623        | 0.9877 | 40   | 0.4314          |
| 0.407         | 2.0    | 81   | 0.4311          |
| 0.333         | 2.9877 | 121  | 0.4458          |
| 0.2632        | 4.0    | 162  | 0.5337          |
| 0.1819        | 4.9877 | 202  | 0.5900          |
| 0.1467        | 6.0    | 243  | 0.7211          |
| 0.1318        | 6.9877 | 283  | 0.8290          |
| 0.1301        | 7.9012 | 320  | 0.8537          |


### Framework versions

- PEFT 0.12.0
- Transformers 4.45.0
- Pytorch 2.4.1+cu121
- Datasets 2.21.0
- Tokenizers 0.20.1