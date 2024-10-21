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

#from utils.post_processing import *

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

# DATASET_DIR = os.path.join(ZS_DIR, "datasets")
# OUTPUT_DIR = os.path.abspath(os.path.join(ZS_DIR, "results", f"""zs_{model_id.split("/")[1]}"""))


## 2. Instantiate Model and Tokenizer ###

inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding='left', padding_side='left')
inference_tokenizer.pad_token = inference_tokenizer.eos_token
terminators = [inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
#inference_tokenizer.chat_template = "llama"

generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    #attn_implementation="flash_attention_2",
    device_map="auto",
)

#print(f"After model instantiation: {torch.cuda.memory_allocated(generation_model.device) / (1024**2)}")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#generation_model.to(device)
# print(generation_model.device)
#generation_model = nn.DataParallel(generation_model)


### 3. Read Data from CSV file ###



df = pd.read_csv(os.path.join(DATASET_DIR, "comics_dataset.csv"))
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
3. ONLY use the valid emotion classes listed above. 

OUTPUT REQUIREMENTS:
- Format: JSON object with a single key "list_emotion_classes"
- Value: Array of one or more emotion classes as strings
- Example: {{"list_emotion_classes": ["anger", "fear"]}}

IMPORTANT NOTES:
- Do not include any explanations in the output, only the JSON object
- ONLY use labels from {formatted_classes} - no variations or new labels allowed

"""
    return instruction

instruction = build_instruction()

sys_msg_l = []
task_msg_l = []
#top_msg_l = []
prepared_sys_task_msg_l = []

for row in df.iterrows():

    #top_msg = {"role": "system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You must absolutely not generate any text or explanation other than the following JSON format: {\"utterance_emotion\": \"<predicted emotion classes for the utterance (str)>}\"\n\n"}
    #sys_msg = {"role":"user", "content": instruction}
    #task_msg = {"role":"assistant", "content": f"# Utterance:\n{row[1].utterance}\n\n# Result:\n"}
    sys_msg = {"role": "system", "content": build_instruction()}
    task_msg = {"role": "assistant", "content": f"""Input: "{row[1].utterance}"
Output:"""}

    sys_msg_l.append(sys_msg)
    task_msg_l.append(task_msg)
    #top_msg_l.append(top_msg)

for i in range(len(sys_msg_l)):

    prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])
    #prepared_sys_task_msg_l.append([sys_msg_l[i]])

### 5. Run Classification / Generate labels ###

outputs_l = []

#for i in tqdm(range(len(prepared_sys_task_msg_l))):

messages = prepared_sys_task_msg_l

inputs = inference_tokenizer.apply_chat_template(
            messages,
            #tools=tools,
            # pad_token = tokenizer.eos_token,
            padding=True,
            truncation=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)

def batch_tensor(tensor, batch_size):
    return [tensor[i:i+batch_size] for i in range(0, tensor.size(0), batch_size)]


BATCH_SIZE = 128
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

decoded_outputs = []

for batch in generated_outputs:

    for prediction in batch:

        decoded_outputs.append(inference_tokenizer.decode(prediction, skip_special_tokens=True))

results_file = Path(OUTPUT_DIR) / "results.pickle"
results_file.parent.mkdir(parents=True, exist_ok=True)

with results_file.open('wb') as fh:
    results_d = {"grounds": grounds,
                 "predictions": decoded_outputs    
        
    }
    pickle.dump(results_d, fh)


# preds = []
# for output in outputs_l:
#     for prediction in output:
#         try:
#             # Use json.loads to safely parse the JSON-like string
#             parsed_prediction = json.loads(prediction)
#             # Append the values of the parsed prediction to preds
#             preds.append(list(parsed_prediction.values()))
#         except json.JSONDecodeError as e:
#             print(f"Error decoding prediction: {e}")
#             # Optionally, append a placeholder or handle error
#             #preds.append([])  # or handle the error differently

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

# print(len(preds))

# grounds_matrix, preds_matrix, classes = post_process(grounds, preds) # type: ignore

# print(classification_report(grounds_matrix, preds_matrix, target_names=classes, digits=3))

# classification_file = Path(OUTPUT_DIR) / "classification_report.pickle"

# with classification_file.open('wb') as fh:
    
#     pickle.dump(classification_report(grounds_matrix, preds_matrix, target_names=classes, output_dict=True), fh)
