# ***************** Fine-Tuning LLMs on Comics dataset *********************** #

# ********** Libraries and GPU *************

import os
import ast
import sys
import json
import torch
import pickle
import subprocess

sys.path.append('../')

import pandas as pd

from pathlib import Path
from tqdm import tqdm
from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc
from sklearn.metrics import classification_report
from utils.post_processing import post_process

try:    
    assert torch.cuda.is_available() is True
    
except AssertionError:
    
    print("Please set up a GPU before using LLaMA Factory...")


# ************** PATH SETTINGS *************

CURRENT_DIR = Path.cwd()
FT_DIR = CURRENT_DIR.parent
DATASET_DIR = os.path.join(FT_DIR, "datasets")

ERC_DIR = FT_DIR.parent
LLAMA_FACTORY_DIR = os.path.join(ERC_DIR, "LLaMA-Factory")

BASE_MODEL = "unsloth/Llama-3.2-3B-Instruct-bnb-4bit"
LOGGING_DIR = os.path.join(FT_DIR, "training_logs")
OUTPUT_DIR = os.path.join(FT_DIR, "saved_models", f"""comics35_pg_old_{BASE_MODEL.split("/")[1]}""")

# print(CURRENT_DIR, FT_DIR, DATASET_DIR, ERC_DIR, LLAMA_FACTORY_DIR, BASE_MODEL, OUTPUT_DIR, sep="\n")


# ****************** DATASET FILES ******************


# *** TRAIN/TEST DATASET NAME/FILENAME *** #

train_dataset_name = f"""comics35_utterance_pg_train.json"""
test_dataset_name = f"""comics35_utterance_pg_test.json"""

train_dataset_file = os.path.join(DATASET_DIR, train_dataset_name)
test_dataset_file = os.path.join(DATASET_DIR, test_dataset_name)


# *** TRAIN ARGS FILE PATH *** #

if not os.path.exists(os.path.join(FT_DIR, "model_args")):
    os.mkdir(os.path.join(FT_DIR, "model_args"))

train_file = os.path.join(FT_DIR, "model_args", f"""{train_dataset_name.split(".")[0].split("train")[0]}{BASE_MODEL.split("/")[1]}.json""")

# *** UPDATE dataset_info.json file in LLaMA-Factory *** #

dataset_info_line =  {
  "file_name": f"{train_dataset_file}",
  "columns": {
    "prompt": "instruction",
    "query": "input",
    "response": "output"
  }
}

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "r") as jsonFile:
    data = json.load(jsonFile)

data["comics"] = dataset_info_line

with open(os.path.join(LLAMA_FACTORY_DIR, "data/dataset_info.json"), "w") as jsonFile:
    json.dump(data, jsonFile)


# ************************** TRAIN MODEL ******************************#

NB_EPOCHS = 3

args = dict(
    
  stage="sft",                           # do supervised fine-tuning
  do_train=True,

  model_name_or_path=BASE_MODEL,         # use bnb-4bit-quantized Llama-3-8B-Instruct model
  num_train_epochs=NB_EPOCHS,            # the epochs of training
  output_dir=OUTPUT_DIR,                 # the path to save LoRA adapters
  overwrite_output_dir=True,             # overrides existing output contents

  dataset="comics",                      # dataset name
  template="llama3",                     # use llama3 prompt template

  finetuning_type="lora",                # use LoRA adapters to save memory
  lora_target="all",                     # attach LoRA adapters to all linear layers
  per_device_train_batch_size=2,         # the batch size
  gradient_accumulation_steps=4,         # the gradient accumulation steps
  lr_scheduler_type="cosine",            # use cosine learning rate scheduler
  logging_steps=10,                      # log every 10 steps
  warmup_ratio=0.1,                      # use warmup scheduler
  #save_steps=5000,                       # save checkpoint every 1000 steps
  learning_rate=5e-5,                    # the learning rate
  max_samples=5000,                       # use 500 examples in each dataset
  max_grad_norm=1.0,                     # clip gradient norm to 1.0
  quantization_bit=4,                    # use 4-bit QLoRA
  loraplus_lr_ratio=16.0,                # use LoRA+ algorithm with lambda=16.0
  fp16=True,                             # use float16 mixed precision training
  logging_dir=LOGGING_DIR,
  report_to="tensorboard"                       # discards wandb

)

json.dump(args, open(train_file, "w", encoding="utf-8"), indent=2)

p = subprocess.Popen(["llamafactory-cli", "train", train_file], cwd=LLAMA_FACTORY_DIR)
p.wait()


# ********************** INFERENCES ON FINE_TUNED MODEL ******************** #

# LOAD MODEL, ADD LORA ADAPTERS #

args = dict(
  model_name_or_path=BASE_MODEL, # use bnb-4bit-quantized Llama-3-8B-Instruct model
  adapter_name_or_path=OUTPUT_DIR,            # load the saved LoRA adapters
  template="llama3",                     # same to the one in training
  finetuning_type="lora",                  # same to the one in training
  quantization_bit=4,                    # load 4-bit quantized model
)

model = ChatModel(args)

# LOAD TEST SET #

with open(test_dataset_file, "r+") as fh:
    test_dataset = json.load(fh)

test_prompts = []
test_grounds = []

for sample in test_dataset:
    test_prompts.append("\nUser:" + sample["instruction"] + sample["input"])
    test_grounds.append(sample["output"])


# INFERENCE ON TEST SET #

# batch_size = 16
# test_predictions = []
# responses = []

# for i in tqdm(range(0, len(test_prompts), batch_size), desc="Running Inference"):
    
#     batch_prompts = test_prompts[i:i + batch_size]  # Create a batch of prompts
    
#     messages = [{"role": "user", "content": prompt} for prompt in batch_prompts]
    
#     for message in messages:
#     # Directly get the responses for the batch
#         response = ""
        
#         for new_text in model.stream_chat(messages):
#             #print(new_text, end="", flush=True)
#             response += new_text
            
#         responses.append(response)
            
#         torch_gc()

batch_size = 32
test_predictions = []

for i in tqdm(range(0, len(test_prompts), batch_size), desc="Running Inference"):
    
    batch_prompts = test_prompts[i:i + batch_size]  # Create a batch of prompts

    for prompt in tqdm(batch_prompts):

        messages = []
        messages.append({"role": "user", "content": prompt})

        response = ""
        
        for new_text in model.stream_chat(messages):
            response += new_text

        test_predictions.append({"role": "assistant", "content": response})

        torch_gc()

# batch_size = 8  # Adjust as needed

# test_predictions = []

# # Create batches of test prompts
# batched_test_prompts = [test_prompts[i:i + batch_size] for i in range(0, len(test_prompts), batch_size)]

# for batch in tqdm(batched_test_prompts):

#     batch_messages = [{"role": "user", "content": prompt} for prompt in batch]
#     batch_responses = [""] * len(batch)  # Initialize a response for each prompt in the batch

#     # Assuming `model.stream_chat` can process batched messages
#     for i, response_chunks in enumerate(model.stream_chat(batch_messages, batch=True)):  # `batch=True` here is hypothetical
#         for new_text in response_chunks:
#             batch_responses[i] += new_text  # Append streamed text

#     # Append each response in the batch to `test_predictions`
#     for response in batch_responses:
#         test_predictions.append({"role": "assistant", "content": response})

#     torch_gc()

# SAVE GROUNDS AND PREDICTIONS *

with open(os.path.join(OUTPUT_DIR, f"""comics_results_{NB_EPOCHS}.pickle"""), 'wb') as fh:
    results_d = {"grounds": test_grounds,
                 "predictions": test_predictions    
        
    }
    pickle.dump(results_d, fh)


# **************************** POST-PROCESSING ************************ #

# with open(os.path.join(OUTPUT_DIR, f"""comics_results_{NB_EPOCHS}.pickle"""), "rb") as fh:
        
#         results = pickle.load(fh)

# task_grounds, task_preds = post_process(results)

# print(classification_report(task_grounds, task_preds, digits=3))

# with open(f"""{OUTPUT_DIR}/classification_report.pickle""", 'wb') as fh:
    
#     pickle.dump(classification_report(task_grounds, task_preds, output_dict=True), fh)
