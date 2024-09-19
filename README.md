# Sentiment Analysis and Emotion Classification for Comics using Large Language Models (LLMS)

Fine-tuning LLMs using LLaMA-Factory for Sentiment Analysis in Comics. We reformulate emotion classification as a text generation task. We experiment with the following models:

- TinyLLamA
- LLaMA-3-8B
- LLaMA-3.1-8B
- Gemma-2-9B
- Mistral-7B
- Qwen-2-7B
- Phi-3-mini


# Data

We experiment with a dataset which consists of 38 (and increasing) annotated Comics titles. We use the Eckman emotions model which consists of six bases emotions: *Anger (AN)*, *Disgust (DI)*, *Fear (FE)*, *Sadness (SA)*, *Surprise (SU)* or *Joy (JO)*, and *Neutral* whic fit neither of the afore-mentioned classes.

# Context Configurations

We finetune LLMs for the Comics dataset on three context levels: 

1) **Utternace level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with no additiona context given.
2) **Page level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page level provided as input to the LLM.
3) **Title level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page complete book level provided as input to the LLM.


# Tasks:

We use LLMs for three classification tasks:

1) **Zero-shot Classification (ZSC):** Zero-shot classification is a Deep Learing technique where the pre-trained model is use *off the shelf* (i.e. witout any further training) for inference on completely unseen data samples.
2) **In-Context Learning (ICL):** In-Context Learning is a Deep Learing technique where a model is *guided* for inference with the help of a few solved demonstrations added in the model's input prompt.
3) **Fine-tuning (FT):** Fine-tuning involves further training of a pre-trained model on a downstream dataset. This helps general-purpose model training to be complemented with task specific supervised training.

# Requirements

We use the following versions of the packages:

```
torch==2.4.0
gradio==4.43.0
pydantic==2.9.0
LLaMA-Factory==0.9.0
transformers==4.44.2
bitsandbytes==0.43.1
```

# Platform

All experiments have been performed on the High Performance Cluster at La Rochelle Université.