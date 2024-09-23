# Sentiment Analysis and Emotion Classification for Comics using Large Language Models (LLMs)

Fine-tuning LLMs using LLaMA-Factory for Sentiment Analysis in Comics. We reformulate emotion classification as a *text generation task*. We experiment with the following models:

- **LLaMA-3-8B**, from Meta AI.
- **LLaMA-3.1-8B**, from Meta AI.
- **Gemma-2-9B**, from Google.
- **Mistral-7B**, from Mistral.
- **Qwen-2-7B**, from Qwen.
- **Phi-3-mini**, from Microsoft.
- **Qwen-2.5-1.5B**, from Qwen.
- **Falcon-7b**, from Technology Innovation Institute.


# Data

We experiment with a dataset which consists of 32 (and increasing) annotated Comics titles. We use the Eckman emotions model which consists of six bases emotions: *Anger (AN)*, *Disgust (DI)*, *Fear (FE)*, *Sadness (SA)*, *Surprise (SU)* or *Joy (JO)*, and *Neutral* which fit neither of the afore-mentioned classes. The 32 titles consist of 5,282 annotated utterances. Of these, the train set comprises of 3506 utterances and the test set of 1776 utternaces.

# Context Configurations

We finetune LLMs for the Comics dataset on three context levels: 

1) **Utternace level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with no additiona context given.
2) **Page level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page level provided as input to the LLM.
3) **Title level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page complete book level provided as input to the LLM.


# Modalities

We use LLMs for three classification tasks:

1) **Zero-shot Classification (ZSC):** Zero-shot classification is a Deep Learning technique where the pre-trained model is use *off the shelf* (i.e. witout any further training) for inference on completely unseen data samples.
2) **In-Context Learning (ICL):** In-Context Learning is a Deep Learning technique where a model is *guided* for inference with the help of a few solved demonstrations added in the model's input prompt.
3) **Fine-tuning (FT):** Fine-tuning involves further training of a pre-trained model on a downstream dataset. This helps general-purpose model training to be complemented with task specific supervised training.

# Prompts

For all three modalities, we experiment with different prompting techniques.

1) **Zero-shot Classification (ZSC):**

```
[{'role': 'system',
  'content': '### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format {"utterance_emotion": <predicted emotion classes for the utterance (str)>}\n\n'},
{'role': 'user',
  'content': '# Utterance:\n {utterance} \n\n# Result:\n'}]
```



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

# Platform and Compute

For fine-tuning LLMs, we use the [**LLaMA-Factory**](https://github.com/hiyouga/LLaMA-Factory) framework. For model checkpoints, we use [**Unsloth**](https://huggingface.co/unsloth).

All experiments have been performed on the High Performance Cluster at **La Rochelle Université**.