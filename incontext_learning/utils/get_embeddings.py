from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

embedding_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
embedding_model = AutoModel.from_pretrained("google-bert/bert-base-uncased")

def get_utterance_embeddings(df):
    
    utterance_embed_d = {}

    for utterance in tqdm(df.utterance):
        # print(utterance)
        while True:
            try:
                inputs = embedding_tokenizer(utterance, return_tensors="pt")
                output = embedding_model(**inputs)
                embedding = output[1][0].squeeze()
                utterance_embed_d[utterance] = embedding.detach().numpy()
                break
            except Exception as e:
                print(e)
                
    return utterance_embed_d