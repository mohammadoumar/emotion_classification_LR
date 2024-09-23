### ************************** ZEROSHOT CLASSIFICATION ON COMICS ********************** ###

import os
import ast
import sys
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
from utils.post_process import *

### 1. Read argument, set paths ###

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model to use for zero-shot classification.", type=str)

args = parser.parse_args()
model_id = args.model

CURRENT_DIR = Path.cwd()
ZS_DIR = CURRENT_DIR.parent
DATASET_DIR = Path(ZS_DIR) / "datasets"
OUTPUT_DIR = Path(ZS_DIR) / "results" / f"zs_{model_id.split('/')[1]}"

# DATASET_DIR = os.path.join(ZS_DIR, "datasets")
# OUTPUT_DIR = os.path.abspath(os.path.join(ZS_DIR, "results", f"""zs_{model_id.split("/")[1]}"""))


## 2. Instantiate Model and Tokenizer ###

inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding='left', padding_side='left')
inference_tokenizer.pad_token = inference_tokenizer.eos_token
terminators = [
    inference_tokenizer.eos_token_id,
    inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


### 3. Read Data from CSV file ###



df = pd.read_csv(os.path.join(DATASET_DIR, "comics_data_processed.csv"))
df['emotions_list'] = df.apply(lambda row: extract_emotions(row), axis=1)
df = df.drop(columns=[df.columns[0], df.columns[1]]).reset_index(drop=True)

### 4. Prepare Messages and Prompts ###

sys_msg_l = []
task_msg_l = []
prepared_sys_task_msg_l = []

for row in df.iterrows():

    sys_msg = {"role":"system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format {\"utterance_emotion\": <predicted emotion classes for the utterance (str)>}\n\n"}
    task_msg = {"role":"user", "content": f"# Utterance:\n{row[1].utterance}\n\n# Result:\n"}

    sys_msg_l.append(sys_msg)
    task_msg_l.append(task_msg)

for i in range(len(sys_msg_l)):

    prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])

### 5. Run Classification / Generate labels ###

outputs_l = []

for i in tqdm(range(len(prepared_sys_task_msg_l))):

    messages = prepared_sys_task_msg_l[i]

    input_ids = inference_tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    padding=True,
    truncation=True,
    return_tensors="pt"
).to(generation_model.device) # type: ignore

    outputs = generation_model.generate(
    input_ids = input_ids,
    max_new_tokens=1024,
    pad_token_id=inference_tokenizer.eos_token_id,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    )
    # inference_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    outputs_l.append(inference_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))

### 6. Save Results, Post Process and Save Classficiation Reports ###

grounds = df.emotions_list.tolist()
preds = [list(ast.literal_eval(output).values()) for output in outputs_l]

results_file = Path(OUTPUT_DIR) / "results.pickle"
results_file.parent.mkdir(parents=True, exist_ok=True)

with results_file.open('wb') as fh:
    results_d = {"ground_truths": grounds,
                 "predictions": preds    
        
    }
    pickle.dump(results_d, fh)

grounds_matrix, preds_matrix = post_process_zs(grounds, preds)

print(classification_report(grounds_matrix, preds_matrix, target_names=all_labels, digits=3))

classification_file = Path(OUTPUT_DIR) / "classification_report.pickle"

with classification_file.open('wb') as fh:
    
    pickle.dump(classification_report(grounds_matrix, preds_matrix, target_names=all_labels, output_dict=True), fh)
