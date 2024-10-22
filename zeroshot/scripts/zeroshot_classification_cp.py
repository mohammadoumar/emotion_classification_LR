### ************************** ZEROSHOT CLASSIFICATION ON COMICS ********************** ###

import os
import ast
import sys
import json
import torch
import pickle
import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

sys.path.append('../')

from utils.pre_process import *
from utils.post_processing import *

### 1. Read argument, set paths ###

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model to use for zero-shot classification.", type=str)

args = parser.parse_args()
model_id = args.model

CURRENT_DIR = Path.cwd()
ZS_DIR = CURRENT_DIR.parent
DATASET_DIR = Path(ZS_DIR).parent / "dataset_files"
OUTPUT_DIR = Path(ZS_DIR) / "results" / f"comics35_zs_{model_id.split('/')[1]}"



## 2. Instantiate Model and Tokenizer ###

inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding='left', padding_side='left')
inference_tokenizer.pad_token = inference_tokenizer.eos_token
terminators = [inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")]


generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


### 3. Read Data from CSV file ###



df = pd.read_csv(os.path.join(DATASET_DIR, "comics_dataset.csv"))
df['emotions_list'] = df.apply(lambda row: extract_emotions(row), axis=1)
df = df.drop(columns=[df.columns[0], df.columns[1]]).reset_index(drop=True)

### 4. Prepare Messages and Prompts ###


def build_instruction():
    
    emotion_classes = ["anger", "disgust", "fear", "sadness", "surprise", "joy", "neutral"]
    formatted_classes = ", ".join([f'"{emotion}"' for emotion in emotion_classes])
    
    instruction = f"""### Emotion Analysis Expert Role

You are an advanced emotion analysis expert specializing in comic book dialogue interpretation. Your task is to analyze utterances and identify their emotional content.

INPUT:
- You will receive a single utterance from a comic book
- The utterance may express one or multiple emotions

TASK:
1. Carefully analyze the emotional context and tone of the utterance
2. Identify applicable emotions from the following classes:
   {formatted_classes}

OUTPUT REQUIREMENTS:
- Format: JSON object with a single key "list_emotion_classes"
- Value: Array of one or more emotion classes as strings
- Example: {{"list_emotion_classes": ["anger", "fear"]}}

IMPORTANT NOTES:
- Do not include any explanations in the output, only the JSON object

"""
    return instruction

instruction = build_instruction()

sys_msg_l = []
task_msg_l = []
#top_msg_l = []
prepared_sys_task_msg_l = []

for row in df.iterrows():

    sys_msg = {"role": "system", "content": build_instruction()}
    task_msg = {"role": "assistant", "content": f"""Input: "{row[1].utterance}"
Output:"""}

    sys_msg_l.append(sys_msg)
    task_msg_l.append(task_msg)


for i in range(len(sys_msg_l)):

    prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])

### 5. Run Classification / Generate labels ###

outputs_l = []
messages = prepared_sys_task_msg_l

inputs = inference_tokenizer.apply_chat_template(
            messages,
            padding=True,
            truncation=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)


def batch_tensor(tensor, batch_size):
    return [tensor[i:i+batch_size] for i in range(0, tensor.size(0), batch_size)]

BATCH_SIZE = 64
input_ids_batches = batch_tensor(inputs['input_ids'], BATCH_SIZE) # type: ignore
attention_mask_batches = batch_tensor(inputs['attention_mask'], BATCH_SIZE) # type: ignore


generated_outputs = []

for i, (input_ids_batch, attention_mask_batch) in enumerate(zip(input_ids_batches, attention_mask_batches)):
    
    print(f"Processing batch {i + 1}")
    
    # Move tensors to model device
    inputs = {
        'input_ids': input_ids_batch.to(generation_model.device), # type: ignore
        'attention_mask': attention_mask_batch.to(generation_model.device) # type: ignore
    }
    
    # Generate output using model.generate
    #generated = generation_model.generate(**inputs, max_new_tokens=32) # correct answers!
    generated = generation_model.generate(**inputs, max_new_tokens=32, pad_token_id=inference_tokenizer.eos_token_id, eos_token_id=terminators, do_sample=True,
     temperature=0.1,
     top_p=0.9,)
    
    # Store the generated output
    generated_outputs.append(generated)


### 6. Save Results, Post Process and Save Classficiation Reports ###

grounds = df.emotions_list.tolist()

decoded_outputs = []

for batch in generated_outputs:

    for prediction in batch:

        decoded_outputs.append(inference_tokenizer.decode(prediction, skip_special_tokens=True))

results_file = Path(OUTPUT_DIR) / "results.pickle"
results_file.parent.mkdir(parents=True, exist_ok=True)

results_d = {"grounds": grounds,
            "predictions": decoded_outputs            
}

with results_file.open('wb') as fh:
    
    pickle.dump(results_d, fh)


grounds_matrix, preds_matrix, classes = post_process(results_d) # type: ignore

print(classification_report(grounds_matrix, preds_matrix, target_names=classes, digits=3))

classification_file = Path(OUTPUT_DIR) / "classification_report.pickle"

with classification_file.open('wb') as fh:
    
    pickle.dump(classification_report(grounds_matrix, preds_matrix, target_names=classes, output_dict=True), fh)
