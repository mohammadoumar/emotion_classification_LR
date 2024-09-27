import os
import json
import pickle

from pathlib import Path

all_reports = {}

CURRENT_DIR = Path.cwd()
ZS_DIR = CURRENT_DIR.parent
RESULTS_DIR = os.path.join(ZS_DIR, "results")


directories = [d for d in os.listdir(RESULTS_DIR) if os.path.isdir(os.path.join(RESULTS_DIR, d))]

for directory in directories:

    for file_name in os.listdir(os.path.join(RESULTS_DIR, directory)):

        if file_name.startswith('classification_report') and file_name.endswith('.pickle'):
            file_path = os.path.join(RESULTS_DIR, directory, file_name)
            

            with open(file_path, 'rb') as f:
                classification_report = pickle.load(f)
            

            all_reports[directory] = classification_report

with open(os.path.join(RESULTS_DIR, 'zeroshot_classification_reports.json'), 'w') as json_file:
    json.dump(all_reports, json_file, indent=4)

print("Classification reports have been successfully dumped into 'zeroshot_classification_reports.json'")


