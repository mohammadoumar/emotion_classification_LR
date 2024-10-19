# 📣 Introduction 📣

**Sentiment Analysis and Emotion Classification in Comics using LLMs:** This ongoing project addresses sentiment analysis and emotion classification in comics using large langauge models (LLMs). We reformulate emotion classification in comics as a *text generation task* where the LLM is prompted to generate the emotion label for utterance(s). We implement emotion classification as *zero-shot classification* (ZSC), *in-context learning* (ICL) and *fine-tuning* (FT). We incorporate different contextual elements into the textual prompts such as, inter alia, scene information, page level information and title-author information.

<br>

# 📂 Repository Structure

This repository is organized as follows:

1) **bert:** this directory contains the implementation of BERT finetuning.
2) **data:** this directory contains the datasets used for the experiments.
3) **emory_nlp:** this directory contains the implementation for the Emory NLP dataset [**(Emory NLP).**](https://github.com/emorynlp/character-mining)
4) **finetuning:** this directory contains the implementation of LLM finetuning for comics. 
5) **incontext_learning:** this directory contains the implementation of in-context learning with LLMs.
6) **meld:** this directory contains the implementation for the MELD dataset [**(MELD).**](https://affective-meld.github.io/)
7) **zeroshot:** this directory contains the implementation of zero-shot classification for comics using LLMs.

<br>

```
.
├── bert
│   ├── notebooks
├── data
├── emory_nlp
│   ├── model_args
│   ├── notebooks
│   ├── saved_models
│   ├── scripts
├── finetuning
│   ├── saved_models
│   ├── model_args
│   ├── notebooks
│   ├── scripts
│   └── utils
├── incontext_learning
│   ├── notebooks
│   ├── results
│   ├── scripts
│   └── utils
├── meld
│   ├── model_args
│   ├── notebooks
│   ├── saved_models
│   ├── scripts
└── zeroshot
    ├── notebooks
    ├── results
    ├── scripts
    └── utils

```

<br>

# ⛓️ Models

We experiment with the following models:

- **BERT** -- [**Google BERT**](https://huggingface.co/google-bert)

- **LLaMA** -- LLaMA-3-8B-Instruct, LLaMA-3-70B-Instruct, LLaMA-3.1-8B-Instruct, LLaMA-3.1-70B-Instruct, LLaMA-3.2-1B-Instruct, LLaMA-3.2-3B-Instruct -- [**Meta AI**](https://huggingface.co/meta-llama)

- **Gemma** -- Gemma-2-2B-it, Gemma-2-9B-it -- [**Google**](https://huggingface.co/google)

- **Qwen** -- Qwen-2-7B-Instruct, Qwen-2.5-0.5B-Instruct, Qwen-2.5-1.5B-Instruct, Qwen-2.5-7B-Instruct -- [**Qwen**](https://huggingface.co/Qwen)

- **Mistral-7B-Instruct** -- [**Mistral AI**](mistralai/Mistral-7B-Instruct-v0.3)
- **Phi-3-mini-instruct** -- [**Microsoft**](microsoft/Phi-3-mini-4k-instruct)
- **Falcon-7b-instruct** -- [**Technology Innovation Institute**](tiiuae/falcon-7b-instruct)

<br>

# 🧮 Data

We experiment with three datasets:

1) **Comics:** Comics dataset consists of 38 annotated Comics titles. We use the Eckman emotions model which consists of six bases emotions: *Anger (AN)*, *Disgust (DI)*, *Fear (FE)*, *Sadness (SA)*, *Surprise (SU)*, *Joy (JO)* and *Neutral*. The 35 titles consist of 7,129 annotated utterances. Of these, the train set comprises of 5803 utterances and the test set of 1326 utterances.
2) **EmoryNLP:** EmoryNLP dataset consists of 97 episodes, 897 scenes, and 12,606 utterances based on the popular TV show called Friends, where each utterance is annotated with one of the seven emotions borrowed from the six primary emotions in the Willcox (1982)’s feeling wheel, sad, mad, scared, powerful, peaceful, joyful, and a default emotion of neutral.
3) **MELD:** Multimodal EmotionLines Dataset (MELD) consists of more than 1400 dialogues and 13000 utterances from Friends TV series.

<br>

# 📚 Context Configurations

We finetune LLMs for the Comics dataset on three context levels: 

1) **Utterance level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with no additiona context given.
2) **Page level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page level provided as input to the LLM.
3) **Title level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page complete book level provided as input to the LLM.

<br>

# 🎛️ Modalities

We use LLMs for three classification tasks:

1) **Zero-Shot Classification (ZSC):** Zero-shot classification is a Deep Learning technique where the pre-trained model is used *off the shelf* (i.e. witout any further training) for inference on completely unseen data samples.
2) **In-Context Learning (ICL):** In-Context Learning is a Deep Learning technique where a model is *guided* for accurate inference with the help of a few solved demonstrations added in the model's input prompt.
3) **Fine-Tuning (FT):** Fine-tuning involves further training of a pre-trained model on a downstream dataset. This helps general-purpose model training to be complemented with task specific supervised training.
4) **Retrieval Augmented Generation (RAG):** combines information retrieval with text generation models by allowing them to retrieve relevant external information before generating a response or an answer.

<br>

# ⌨️ Prompts

For all three modalities, we experiment with different prompting techniques.

1) **Zero-Shot Classification (ZSC):** The prompt used for **LLaMA** and **Qwen** models is given below:

```
[{'role': 'system',
  'content': '### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format {"utterance_emotion": <predicted emotion classes for the utterance (str)>}\n\n'},
{'role': 'user',
  'content': '# Utterance:\n {utterance} \n\n# Result:\n'}]
```

2) **Fine-Tuning (FT):** For fine-tuning, we used the template default for the respective model. In general, the prompt is in the {"instruction", "input", "output"} format given below:

```
{'instruction': '### Emotion Analysis Expert Role\n\nYou are an advanced emotion analysis expert specializing in comic book dialogue interpretation. Your task is to analyze utterances and identify their emotional content.\n\nINPUT:\n- You will receive a single utterance from a comic book\n- The utterance may express one or multiple emotions\n\nTASK:\n1. Carefully analyze the emotional context and tone of the utterance\n2. Identify applicable emotions from the following classes:\n   "Anger", "Disgust", "Fear", "Sadness", "Surprise", "Joy", "Neutral"\n\nOUTPUT REQUIREMENTS:\n- Format: JSON object with a single key "list_emotion_classes"\n- Value: Array of one or more emotion classes as strings\n- Example: {"list_emotion_classes": ["Anger", "Fear"]}\n\nIMPORTANT NOTES:\n- Do not include any explanations in the output, only the JSON object\n\n', 'input': '### Here is the utterance from a comic book: {utterance}', 'output': '{"list_emotion_classes": ["Fear", "Surprise"]}'}
```

3) **In-Context Learning (ICL):** For ICL, an instance of the prompt with 3 examples (k = 3) is given below:

```
### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You are given one utterance to classify and 3 example utterances to help you. You must absolutely not generate any text or explanation other than the following JSON format: {"utterance_emotion": "<predicted emotion classes for the utterance (str)>}"

### Examples:

## Example 1
Utterance 1={uterrance_1}

# Result:
{"utterance_emotions": "{result_1}"}

## Example 2
Utterance 2={uterrance_2}

# Result:
{"utterance_emotions": "{result_1}"}

## Example 3
Utterance 3={uterrance_3}

# Result:
{"utterance_emotions": "{result_1}"}

# Utterance:
{utterance}

# Result:
```

<br>

# 📦 Requirements

We use the following versions of the packages:

```
torch==2.4.0
gradio==4.43.0
pydantic==2.9.0
LLaMA-Factory==0.9.0
transformers==4.44.2
bitsandbytes==0.43.1
```

For fine-tuning, you need to install LLaMA-Factory. Run the following command to install LLaMA-Factory and all the necessary dependencies and updates:

```
bash setup.sh
```

<br>

# 💻 Platform and Compute

- For fine-tuning LLMs, we use [**LLaMA-Factory.**](https://github.com/hiyouga/LLaMA-Factory)
- For model checkpoints, we use [**Unsloth.**](https://unsloth.ai/)
- We also use [**Hugging Face.**](https://huggingface.co/)

All experiments have been performed on the High Performance Cluster at [**La Rochelle Université.**](https://www.univ-larochelle.fr/)