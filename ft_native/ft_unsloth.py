### IMPORTS ###

import sys
import torch
import json_repair
import pandas as pd

from tqdm import tqdm
from datasets import Dataset

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template
from unsloth import FastLanguageModel, is_bfloat16_supported

from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from get_emotions import *


### TOKENIZER AND MODEL ###

max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    target_modules=["q_proj", "k_proj", "v_proj", "up_proj", "down_proj", "o_proj", "gate_proj"], 
    use_rslora=True,
    use_gradient_checkpointing=True
)


### DATASET ###

df = pd.read_csv("/Utilisateurs/umushtaq/emotion_analysis_comics/dataset_files/comics_dataset.csv")
df['emotion_u'] = df.apply(lambda row: get_emotions(row), axis=1)
df_train = df[df.split == "TRAIN"].reset_index(drop=True)

instruction = build_instruction()

sys_msg_l = []
user_msg_l = []
assistant_msg_l = []

for _, row in df_train.iterrows():
        
        #prompt = instruction.replace("<comic_title>", row['comics_title']).replace("<speaker_id>", row['speaker_id']).replace("<utterance>", row['utterance'])
        
        #human_msg = {'from': 'human', 'value': prompt}
        
        sys_msg = {'role': 'system', 'content': instruction}
        user_msg = {'role': 'user', 'content': "xxx"}

        #obj = {"list_emotion_classes": row['emotion_u']}
        #obj = row['emotion_u']
        #assistant_msg = {'from': 'gpt', 'value': obj}
        assistant_msg = {'role': 'assistant', 'content': f'{{"list_emotion_classes": {row["emotion_u"]}}}'}


        sys_msg_l.append(sys_msg)
        user_msg_l.append(user_msg)
        assistant_msg_l.append(assistant_msg)
        
        
        
comics_dataset = []

for i in range(len(sys_msg_l)):
    
    #obj = {"list_emotion_classes": ["Anger", "Fear"]}

    #comics_dataset.append([human_msg_l[i], assistant_msg_l[i]])
    comics_dataset.append([sys_msg_l[i], user_msg_l[i], assistant_msg_l[i]])
    
    
fixed_comics_dataset = fix_comics_dataset(comics_dataset)

dataset = Dataset.from_dict({
    'conversations': fixed_comics_dataset
})

tokenizer = get_chat_template(
    tokenizer,
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
    chat_template="chatml",
)

def apply_template_comics(examples):
    messages = examples["conversations"]
    #messages = examples['input'] + examples['output']
    text = [tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=False) for message in messages]
    return {"text": text}

comics_dataset = dataset.map(apply_template_comics, batched=True)
dataset_split = split_dataset(comics_dataset)
train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']


### TRAINER ###

args=TrainingArguments(
        learning_rate=3e-4,
        lr_scheduler_type="cosine",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        warmup_steps=10,
        
        eval_strategy="steps",  # Run evaluation during training (can also use "epoch")
        eval_steps=25,  # Perform evaluation every 50 steps
        save_strategy="steps",  # Save the model every few steps
        save_steps=50,  # Save every 200 steps
        load_best_model_at_end=True,
    
        output_dir="ft_uns_comics_qwen2.5",
        seed=0,
    )

trainer=SFTTrainer(
    model=model,
    tokenizer=tokenizer, # type: ignore
    train_dataset=train_dataset,  # Replace with your train dataset
    eval_dataset=eval_dataset, 
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    #dataset_num_proc=2,
    packing=False,
    args=args,
)

trainer.train()


