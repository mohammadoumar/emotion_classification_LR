# 📣 Introduction 📣

**Sentiment Analysis and Emotion Classification in Comics using LLMs:** This ongoing project addresses sentiment analysis and emotion classification in comics using large langauge models (LLMs). We reformulate emotion classification in comics as a *text generation task* where the LLM is prompted to generate the emotion label for utterance(s). We implement emotion classification as zero-shot classification (ZSC), in-context learning (ICL), retrieval augmented generation (RAG) and fine-tuning (FT). We incorporate different contextual elements into the textual prompts such as, inter alia, scene information, page level information and title-author information.

We experiment with the following models:

- **LLaMA-3-8B-Instruct** -- [**Meta AI**](meta-llama/Meta-Llama-3-8B-Instruct)
- **LLaMA-3.1-8B-Instruct** -- [**Meta AI**](meta-llama/Meta-Llama-3.1-8B-Instruct)
- **Gemma-2-9B-it** -- [**Google**](google/gemma-2-9b-it)
- **Mistral-7B-Instruct** -- [**Mistral AI**](mistralai/Mistral-7B-Instruct-v0.3)
- **Qwen-2-7B-Instruct** -- [**Qwen**](Qwen/Qwen2-7B-Instruct)
- **Qwen-2.5-1.5B-Instruct** -- [**Qwen**](Qwen/Qwen2.5-1.5B-Instruct)
- **Phi-3-mini-instruct** -- [**Microsoft**](microsoft/Phi-3-mini-4k-instruct)
- **Falcon-7b-instruct** -- [**Technology Innovation Institute**](tiiuae/falcon-7b-instruct)

<br>

# 📂 Repository Structure

This repository is organized as follows:

1) **data_files**: this directory contains the raw data files containing the annotated data from comics titles. Every utterance is annotated with *emotion* and *speaker_id*.
2) **finetuning**: this directory contains the implementation of LLM finetuning for comics. 
3) **zeroshot**: this directory contains the implementation of zero-shot classification for comics using LLMs.

```
.
├── data_files
├── finetuning
│   ├── data_preparation
│   ├── datasets
│   ├── finetuned_models
│   ├── finetuning_model_args
│   ├── finetuning_scripts
│   ├── notebooks
│   ├── training_logs
│   └── utils
└── zeroshot
    ├── datasets
    ├── notebooks
    ├── results
    ├── scripts
    └── utils
```

<br>

# 🧮 Data

We experiment with a dataset which consists of 32 (and increasing) annotated Comics titles. We use the Eckman emotions model which consists of six bases emotions: *Anger (AN)*, *Disgust (DI)*, *Fear (FE)*, *Sadness (SA)*, *Surprise (SU)* or *Joy (JO)*, and *Neutral* which fit neither of the afore-mentioned classes. The 32 titles consist of 5,282 annotated utterances. Of these, the train set comprises of 3506 utterances and the test set of 1776 utternaces. 

<br>

# 📚 Context Configurations

We finetune LLMs for the Comics dataset on three context levels: 

1) **Utterance level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with no additiona context given.
2) **Page level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page level provided as input to the LLM.
3) **Title level classification:** Every raw utterance in the comics titles is classified into one or more of the emotion classes, with additional context on the page complete book level provided as input to the LLM.

<br>

# 🎛️ Modalities

We use LLMs for three classification tasks:

1) **Zero-Shot Classification (ZSC):** Zero-shot classification is a Deep Learning technique where the pre-trained model is use *off the shelf* (i.e. witout any further training) for inference on completely unseen data samples.
2) **In-Context Learning (ICL):** In-Context Learning is a Deep Learning technique where a model is *guided* for inference with the help of a few solved demonstrations added in the model's input prompt.
3) **Fine-Tuning (FT):** Fine-tuning involves further training of a pre-trained model on a downstream dataset. This helps general-purpose model training to be complemented with task specific supervised training.

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

2) **Fine-Tuning (FT)**: For fine-tuning, we used the template default for the respective model. In general, the prompt is in the {"instruction", "input", "output"} format given below:

```
[{"instruction": 
  "### You are an expert in Emotion Analysis. You are given an utternace from a comic book enclosed by <UT></UT> tags. Your task is to classify each utterance as one or more the following emotion classes: "Anger" (AN), "Disgust" (DI), "Fear" (FE), "Sadness" (SA), "Surprise" (SU) or "Joy" (JO). You must return a list of emotion classes in following JSON format: {"list_emotion_classes": ["emotion_class (str)", "emotion_class (str)" ... "emotion_class (str)"]} where each element "emotion_classes (str)" is replaced by one ore more of the following abbreviated emotion class labels: "AN", "DI", "FE", "SA", "SU" or "JO". \n", 
"input": 
  "### Here is the utterance from a comic book: <UT>DID YOU HAVE TO ELECTROCUTE HER SO HARD?</UT>", 
"output": 
  "{"list_emotion_classes": ["FE", "SU"]}"}]

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