import os
import json
import pickle

from pathlib import Path

all_reports = {}

CURRENT_DIR = Path.cwd()
FT_DIR = CURRENT_DIR.parent
FINE_TUNED_MODELS_DIR = os.path.join(FT_DIR, "saved_models")


directories = [d for d in os.listdir(FINE_TUNED_MODELS_DIR) if os.path.isdir(os.path.join(FINE_TUNED_MODELS_DIR, d))]

for directory in directories:

    for file_name in os.listdir(os.path.join(FINE_TUNED_MODELS_DIR, directory)):

        if file_name.startswith('classification_report') and file_name.endswith('.pickle'):
            file_path = os.path.join(FINE_TUNED_MODELS_DIR, directory, file_name)
            

            with open(file_path, 'rb') as f:
                classification_report = pickle.load(f)
            

            all_reports[directory] = classification_report

with open(os.path.join(CURRENT_DIR, 'finetuning_classification_reports.json'), 'w') as json_file:
    json.dump(all_reports, json_file, indent=4)

print("Classification reports have been successfully dumped into 'finetuning_classification_reports.json'")


