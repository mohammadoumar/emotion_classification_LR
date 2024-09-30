import sys
import ast
import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F

sys.path.append('../')

from pathlib import Path
from tqdm.notebook import tqdm
from operator import itemgetter
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils.pre_process import *

embedding_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
embedding_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

model_id = "unsloth/llama-3-8b-Instruct-bnb-4bit"

inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding='left', padding_side='left')
inference_tokenizer.pad_token = inference_tokenizer.eos_token
terminators = [inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")]

generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# *** Read data *** #

DATASET_DIR = Path(Path.cwd().as_posix()) / "emotion_analysis_comics" / "incontext_learning" / "datasets"

df = pd.read_csv(DATASET_DIR / "comics_data_processed.csv")
df = df.drop(columns=[df.columns[0], df.columns[1]]).reset_index(drop=True)
df['emotions_list'] = df.apply(lambda row: extract_emotions(row), axis=1)
