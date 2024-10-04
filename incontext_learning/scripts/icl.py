import sys
import torch
import pickle
import argparse
import pandas as pd

sys.path.append('../')

from pathlib import Path
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from utils.batch import *
from utils.pre_process import *
from utils.post_process import *
from utils.get_embeddings import *
from utils.prepare_kneighbours_prompt import *

# *** Read Args *** #

parser = argparse.ArgumentParser()
parser.add_argument("model", help="The model to use for zero-shot classification.", type=str)
parser.add_argument("k", help="the K examples to include.", type=int)

args = parser.parse_args()
model_id, k = args.model, args.k

# *** Instantiate model and tokenizer *** #

print("\n\n********* Instantiating model and tokenizer **********\n\n")

#inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding='left', padding_side='left')
inference_tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
inference_tokenizer.pad_token = inference_tokenizer.eos_token
terminators = [inference_tokenizer.eos_token_id, inference_tokenizer.convert_tokens_to_ids("<|eot_id|>")]

generation_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# *** Set paths *** #

CURRENT_DIR = Path.cwd()
ICL_DIR = CURRENT_DIR.parent
DATASET_DIR = Path(ICL_DIR) / "datasets"
OUTPUT_DIR = Path(ICL_DIR) / "results" / f"icl_{model_id.split('/')[1]}"

# *** Read data *** #

df = pd.read_csv(DATASET_DIR / "comics_data_processed.csv")
df = df.drop(columns=[df.columns[0], df.columns[1]]).reset_index(drop=True)
df['emotions_list'] = df.apply(lambda row: extract_emotions(row), axis=1)

print("\n\n********** Computing embeddings using BERT **********\n\n")
utterance_embed_d = get_utterance_embeddings(df)
df['utterance_embedding'] = df.utterance.apply(lambda x: utterance_embed_d[x])

train_df = df[df.split == "TRAIN"].reset_index(drop=True)
test_df = df[df.split == "TEST"].reset_index(drop=True)

# *** Get prompts with K-neighbours *** #

sys_msg_l = []
task_msg_l = []

print("\n\n********** Computing K-neighbours and preparing prompts **********\n\n")
for row in tqdm(test_df.iterrows(), total=len(test_df)):
    
    sys_msg = {"role": "system", "content": "### Task description: You are an expert sentiment analysis assistant that takes an utterance from a comic book and must classify the utterance into appropriate emotion class(s): anger, surprise, fear, disgust, sadness, joy, neutral. You are given one utterance to classify and 3 example utterances to help you. You must absolutely not generate any text or explanation other than the following JSON format: {\"utterance_emotion\": \"<predicted emotion classes for the utterance (str)>}\"\n\n" + "### Examples:\n\n" + prepare_similar_example_prompts(row[1].utterance, k, train_df=train_df, test_df=test_df)}
    #sys_msg = {"role":"system", "content": "### Task description: You are an expert biomedical assistant that takes 1) an abstract text, 2) the list of all arguments from this abstract text, and must classify all arguments into one of two classes: Claim or Premise. " + proportion_desc + " You must absolutely not generate any text or explanation other than the following JSON format {\"Argument 1\": <predicted class for Argument 1 (str)>, ..., \"Argument n\": <predicted class for Argument n (str)>}\n\n### Class definitions:" + " Claim = " + claim_fulldesc + " Premise = " + premise_fulldesc + "\n\n### Examples:\n\n" + prepare_similar_example_prompts(title_l[i], experiment_df, k=3, seed=seed)}  # Sample by similar title
    task_msg = {"role": "user", "content": f"# Utterance:\n{row[1].utterance}\n\n# Result:\n"}
    
    sys_msg_l.append(sys_msg)
    task_msg_l.append(task_msg)
    
prepared_sys_task_msg_l = []

for i in range(len(sys_msg_l)):
    prepared_sys_task_msg_l.append([sys_msg_l[i], task_msg_l[i]])
    
# *** Tokenize and apply chat template *** #

inputs = inference_tokenizer.apply_chat_template(
            prepared_sys_task_msg_l,
            #pad_token = inference_tokenizer.eos_token,
            padding=True,
            truncation=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
)

# *** Batch and run inferences *** #

BATCH_SIZE = 16

input_ids_batches = batch_tensor(inputs['input_ids'], BATCH_SIZE) # type: ignore
attention_mask_batches = batch_tensor(inputs['attention_mask'], BATCH_SIZE) # type: ignore

generated_outputs = []

for i, (input_ids_batch, attention_mask_batch) in tqdm(enumerate(zip(input_ids_batches, attention_mask_batches)), total=len(input_ids_batches)):
    
    print(f"\n\n ***** Processing batch {i + 1} *****\n\n")
    
    inputs = {
        'input_ids': input_ids_batch.to(generation_model.device), # type: ignore
        'attention_mask': attention_mask_batch.to(generation_model.device) # type: ignore
    }

    outputs = generation_model.generate(
    **inputs,
    max_new_tokens=64,
    pad_token_id=inference_tokenizer.eos_token_id,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.1,
    top_p=0.9,
    )
    
    generated_outputs.append(outputs)
    
decoded_outputs = []

for batch in generated_outputs:

    for prediction in batch:

        decoded_outputs.append(inference_tokenizer.decode(prediction[inputs['input_ids'].shape[1]:], skip_special_tokens=True)) # type: ignore
    
 
grounds = test_df.emotions_list.tolist()   

results_file = Path(OUTPUT_DIR) / f"results_{k}.pickle"
results_file.parent.mkdir(parents=True, exist_ok=True)

with results_file.open('wb') as fh:
    results_d = {"ground_truths": grounds,
                 "predictions": decoded_outputs    
        
    }
    pickle.dump(results_d, fh)
    
predictions = read_json_preds(decoded_outputs)

true_matrix, predicted_matrix = post_process_icl(grounds, predictions)

print(classification_report(true_matrix, predicted_matrix, target_names=all_labels, digits=3))

classification_file = Path(OUTPUT_DIR) / f"classification_report_{k}.pickle"

with classification_file.open('wb') as fh:
    
    pickle.dump(classification_report(true_matrix, predicted_matrix, target_names=all_labels, output_dict=True), fh)
