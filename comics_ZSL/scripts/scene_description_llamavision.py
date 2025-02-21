import json
import pandas as pd

from PIL import Image # type: ignore
from tqdm import tqdm
from unsloth import FastVisionModel # type: ignore

max_seq_length = 4096

model, tokenizer = FastVisionModel.from_pretrained(

    model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    dtype=None,
)

df = pd.read_csv("/Utilisateurs/umushtaq/emotion_analysis_comics/dataset_files/comics_pg_w_images.csv", index_col=0)

def generation_instruction():
    
    instruction = f"""Describe this comics page with focus on the characters' emotional states. Include:
1. The facial expressions, body language, and micro-expressions of each character
2. The emotional atmosphere of the scene (tense, joyful, melancholic, etc.)
3. Any emotional subtext or contrast between characters
4. How the emotional state relates to the narrative context

Incorporate all text elements present in the panel:
- Analyze dialogue and captions to understand character emotions
- Analyze how typography (size, style, coloring of text) emphasizes emotional states
- Include how narrative text provides emotional context
- Analyze how spoken/thought text relate to the visual emotional cues


IMPORTANT: Your complete description MUST fit within a strict 256-token limit. Plan your response to conclude naturally and completely without being cut off abruptly.
"""
    return instruction

def build_image_modality(image_path):
    
    return Image.open(image_path)

def convert_to_conversation_test(row):
  
    image_path = row.image_path
  
    instruction = generation_instruction()
    image = build_image_modality(image_path)
    
    conversation = [
        { "role": "user",
          "content" : [
            {"type" : "text",  "text"  : instruction},
            {"type" : "image", "image" : image} ]
        },
        { "role" : "assistant",
          "content" : ""
        },
    ]
    return { "messages" : conversation }
pass

comics_mm_dataset = [convert_to_conversation_test(row) for _, row in tqdm(df.iterrows())]

model = FastVisionModel.for_inference(model)

raw_outputs = []

for message in tqdm(comics_mm_dataset):
    
    image = message['messages'][0]['content'][1]['image']
    input_text = tokenizer.apply_chat_template(message['messages'], add_generation_prompt = True)

    inputs = tokenizer(
    image,
    input_text,
    add_special_tokens = False,
    return_tensors = "pt",
).to("cuda")


    output = model.generate(**inputs, max_new_tokens=512)[0]
    
    input_length = inputs.input_ids.shape[1]
    generated_tokens = output[input_length:]
    decoded_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    raw_outputs.append(decoded_output)
    
with open("scene_discriptions_vision.json", "w") as file:
    json.dump(raw_outputs, file, indent=4)  # Save as JSON with indentation