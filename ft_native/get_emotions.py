import os
import pandas as pd

def get_emotions(row):

    utterance_emotions = row.emotion
    utterance_emotions_l = []
    emotion_class_labels = ["Anger", "Disgust", "Fear", "Sadness", "Surprise", "Joy"]

    if utterance_emotions == 'Neutral':
        
        utterance_emotions_l.append(utterance_emotions)
    
    else:
        utterance_emotions = utterance_emotions.split("-")

        for idx, emotion_annotation in enumerate(utterance_emotions):

            if '0' not in emotion_annotation:
                
                utterance_emotions_l.append(emotion_class_labels[idx])                

    return utterance_emotions_l

def build_instruction():
    emotion_classes = ["Anger", "Disgust", "Fear", "Sadness", "Surprise", "Joy", "Neutral"]
    formatted_classes = ", ".join([f'"{emotion}"' for emotion in emotion_classes])
    
    instruction = f"""### Emotion Analysis Expert Role

You are an advanced emotion analysis expert specializing in comic book dialogue interpretation. Your task is to analyze utterances and identify their emotional content while considering conversational context.

INPUT:
- You will receive a list of 6 consecutive utterances from a comic book:
  * The first 5 utterances provide conversational context
  * The 6th (last) utterance is the one to be classified
- Each utterance may express one or multiple emotions

TASK:
1. Read through the context utterances to understand the emotional flow
2. Carefully analyze the emotional context, tone, and potential emotional shifts in the final utterance
3. Identify applicable emotions for the final utterance only from the following classes:
   {formatted_classes}

OUTPUT REQUIREMENTS:
- Format: JSON object with a single key "list_emotion_classes"
- Value: Array of one or more emotion classes as strings
- The classification should be for the final utterance only, using previous utterances as context

IMPORTANT NOTES:
- Do not include any explanations in the output, only the JSON object
- Use the context to better understand emotional transitions and current emotional state
- Context utterances may reveal emotional buildup or shifts that influence the final utterance

"""
    return instruction

def fix_comics_dataset(comics_dataset):
    fixed_comics_dataset = []
    for conversation in comics_dataset:
        fixed_conversation = []
        for message in conversation:
            if isinstance(message['content'], list):  # If the 'value' is a list of emotions
                message['content'] = ', '.join(message['content'])  # Join the list into a string
            fixed_conversation.append(message)
        fixed_comics_dataset.append(fixed_conversation)
    return fixed_comics_dataset

def split_dataset(dataset, train_ratio=0.8):
    train_test = dataset.train_test_split(test_size=1 - train_ratio)
    return train_test
