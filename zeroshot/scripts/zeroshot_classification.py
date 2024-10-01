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
#dist.init_process_group(backend='nccl', rank=0, world_size=1)

from utils.pre_process import *
from utils.post_processing import *

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
#terminators = [inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#inference_tokenizer.chat_template = "llama"

generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

#print(f"After model instantiation: {torch.cuda.memory_allocated(generation_model.device) / (1024**2)}")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generation_model.to(device)
# print(generation_model.device)
#generation_model = nn.DataParallel(generation_model)


### 3. Read Data from CSV file ###



df = pd.read_csv(os.path.join(DATASET_DIR, "comics_data_processed.csv"))
df['emotions_list'] = df.apply(lambda row: extract_emotions(row), axis=1)
df = df.drop(columns=[df.columns[0], df.columns[1]]).reset_index(drop=True)

### 4. Prepare Messages and Prompts ###

# * Prompt Prepration for LLaMA and Qwen 
# sys_msg_l = []
# task_msg_l = []
# prepared_sys_task_msg_l = []

# for row in df.iterrows():

#     sys_msg = {"role":"system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format {\"utterance_emotion\": <predicted emotion classes for the utterance (str)>}\n\n"}
#     task_msg = {"role":"user", "content": f"# Utterance:\n{row[1].utterance}\n\n# Result:\n"}

#     sys_msg_l.append(sys_msg)
#     task_msg_l.append(task_msg)

# for i in range(len(sys_msg_l)):

#     prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])

# * Prompt Prepration for Gemma

sys_msg_l = []
task_msg_l = []
#top_msg_l = []
prepared_sys_task_msg_l = []

for row in df.iterrows():

    #top_msg = {"role": "system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format: {\"utterance_emotion\": \"<predicted emotion classes for the utterance (str)>}\"\n\n"}
    sys_msg = {"role":"system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format: {\"utterance_emotion\": <predicted emotion classes for the utterance (str)>}" + f"\n\n# Utterance:\n{row[1].utterance}\n\n# Result:\n"}
    #task_msg = {"role":"user", "content": f"# Utterance:\n{row[1].utterance}\n\n# Result:\n"}

    sys_msg_l.append(sys_msg)
    #task_msg_l.append(task_msg)
    #top_msg_l.append(top_msg)

for i in range(len(sys_msg_l)):

    #prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])
    prepared_sys_task_msg_l.append([sys_msg_l[i]])

### 5. Run Classification / Generate labels ###

outputs_l = []

#for i in tqdm(range(len(prepared_sys_task_msg_l))):

messages = prepared_sys_task_msg_l

input_ids = inference_tokenizer.apply_chat_template(
messages,
add_generation_prompt=True,
tokenize=True,
#tokenize=False,
padding=True,
truncation=True,
#return_dict=True,
return_tensors="pt"
)#.to(generation_model.device)
input_ids = torch.tensor(input_ids)
# print(input_ids)
print(input_ids.shape)
# x = input_ids.ids[:32, :]
# print(x.shape)
# x.to(generation_model.device)

BATCH_SIZE = 256
n = len(prepared_sys_task_msg_l)

for batch_num, i in enumerate(tqdm(range(0, n, BATCH_SIZE)), start=1):
    #batch = input_ids[:BATCH_SIZE, :].to(generation_model.device) 
    batch = input_ids[i:i + BATCH_SIZE, :].to(generation_model.device)
    print(f"Processing Batch: {batch_num}")
    # print(batch.shape)
    #print(batch.nelement() * batch.element_size())
    # print(f"After batch {batch_num} to CUDA: {torch.cuda.memory_allocated(generation_model.device) / (1024**2)}")
    outputs = generation_model.generate(
    input_ids = batch,
    max_new_tokens=64,
    pad_token_id=inference_tokenizer.eos_token_id,
    #eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    )
    outputs = [inference_tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True) for output in outputs]
    outputs_l.append(outputs)
    #outputs_l.append(inference_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)) 
    del batch

print(len(outputs_l))


# outputs = generation_model.generate(
#     input_ids = input_ids,
#     max_new_tokens=64,
#     pad_token_id=inference_tokenizer.eos_token_id,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.9,
#     )

# for i in tqdm(range(len(prepared_sys_task_msg_l))):

#     messages = prepared_sys_task_msg_l[i]

#     input_ids = inference_tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     #tokenize=False,
#     padding=True,
#     truncation=True,
#     #return_dict=True,
#     return_tensors="pt"
# )#.to(generation_model.device) 
#     x = torch.tensor(input_ids).to(generation_model.device)
#     #y = torch.tensor(input_ids["attention_mask"]).to(generation_model.device)
#     #print(x)
#     #print(y)
#     #break
#     #idz = x.to(generation_model.device)
#     #x = torch.tensor(input_ids.)
#     #x.to(generation_model.device)
#     #print(x)
#     #print(type(x))
#     #print(x.device)
#     #break
#     outputs = generation_model.generate(
#     input_ids = x,
#     max_new_tokens=64,
#     pad_token_id=inference_tokenizer.eos_token_id,
#     eos_token_id=terminators,
#     do_sample=True,
#     temperature=0.1,
#     top_p=0.9,
#     )
#     # inference_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
#     outputs_l.append(inference_tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True))  # type: ignore

### 6. Save Results, Post Process and Save Classficiation Reports ###

grounds = df.emotions_list.tolist()
#preds = [list(ast.literal_eval(output).values()) for output in outputs_l]

results_file = Path(OUTPUT_DIR) / "results.pickle"
results_file.parent.mkdir(parents=True, exist_ok=True)

with results_file.open('wb') as fh:
    results_d = {"ground_truths": grounds,
                 "predictions": outputs_l    
        
    }
    pickle.dump(results_d, fh)


preds = []
for output in outputs_l:
    for prediction in output:
        try:
            # Use json.loads to safely parse the JSON-like string
            parsed_prediction = json.loads(prediction)
            # Append the values of the parsed prediction to preds
            preds.append(list(parsed_prediction.values()))
        except json.JSONDecodeError as e:
            print(f"Error decoding prediction: {e}")
            # Optionally, append a placeholder or handle error
            #preds.append([])  # or handle the error differently

# preds = []
# for output in outputs_l:
#     for prediction in output:
#         preds.append([list(ast.literal_eval(prediction).values())])

#preds = [list(ast.literal_eval(output).values()) for output in outputs_l]

# preds = []

# for output in outputs_l:

#     print(len(output))

#     for prediction in output:

#         emotion_data = json.loads(prediction)
        
#         # Split by comma and strip any extra whitespace
#         emotions = [emotion.strip() for emotion in emotion_data["utterance_emotion"].split(',')]
        
#         # Append each emotion list to the final list
#         preds.append(emotions)

print(len(preds))

grounds_matrix, preds_matrix = post_process_zs(grounds, preds)

print(classification_report(grounds_matrix, preds_matrix, target_names=all_labels, digits=3))

classification_file = Path(OUTPUT_DIR) / "classification_report.pickle"

with classification_file.open('wb') as fh:
    
    pickle.dump(classification_report(grounds_matrix, preds_matrix, target_names=all_labels, output_dict=True), fh)
