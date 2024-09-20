#!/bin/bash


list_models=("unsloth/llama-3-8b-Instruct-bnb-4bit" "unsloth/llama-3-8b-Instruct" "unsloth/llama-3-70b-Instruct-bnb-4bit" 
"unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit" "unsloth/Meta-Llama-3.1-70B-Instruct-bnb-4bit" 
"unsloth/Qwen2-7B-Instruct-bnb-4bit" "unsloth/Phi-3-mini-4k-instruct-bnb-4bit" "unsloth/gemma-2-9b-it-bnb-4bit")

#"vivo-ai/BlueLM-7B-Chat-4bits" (unsuccessful.)
#"unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" (done)
#"tiiuae/falcon-mamba-7b-instruct-4bit"
#"deepseek-ai/DeepSeek-V2-Lite-Chat"

# list_tasks=("emotion_classification")


arguments=()


for item1 in "${list_models[@]}"; do
    for item2 in "${list_tasks[@]}"; do
        arguments+=("$item1 $item2")
    done
done


for args in "${arguments[@]}"; do
    echo "Running emo_classification_finetune.py with arguments: $args"

    python3 notebooks/emo_classification_finetune.py $args


    if [ $? -ne 0 ]; then
        echo -e "Error encountered with arguments: $args. Skipping to the next pair. \n \n  ************* \n"
        continue 
    fi

    echo -e  "Successfully ran emo_classification_finetune.py with arguments: $args \n \n  *************** \n"
done
