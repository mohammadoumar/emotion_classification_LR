import torch
import random
from operator import itemgetter
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_k_neighbours(k, utterance, train_df, test_df):
    
    
    test_utterance_embedding = test_df[test_df.utterance == utterance]["utterance_embedding"].values[0]
    #test_utterance_embedding = torch.tensor(test_utterance_embedding)#.to(device)

    utterance_embed_d = {}
    for e in train_df.iterrows():
        if e[1].utterance not in utterance_embed_d:
            #utterance_embed_d[e[1].utterance] = e[1].utterance_embedding
            utterance_embed_d[e[1].utterance] = e[1].utterance_embedding#.to(device)

    # train_titles = set(df[df.split == 'TRAIN'].title.unique())
    train_utterances = set(train_df.utterance)

    dist_l = []
    for t, v in utterance_embed_d.items():
        if t in train_utterances:
            # d = cos_sim(title_embed_d[title], v)
            d = F.cosine_similarity(torch.tensor(test_utterance_embedding), torch.tensor(v), dim=0)
            dist_l.append((t, d.item()))

    sorted_dist_l = sorted(dist_l, key=itemgetter(1), reverse=True)
    
    return sorted_dist_l[0: k]

def prepare_similar_example_prompts(utterance, k, train_df, test_df, seed=33):
    """
    Create a part of prompt made of k examples in the train set, whose topic is most similar to a given title.
    """

    random.seed(seed)

    neighbours_l = get_k_neighbours(2*k, utterance, train_df=train_df, test_df=test_df) # Fetch the 2*k closest neighbors
    # print(neighbours_l)
    sampled_neighbours_l = random.sample(neighbours_l, k) # Only keep k of them
    # bprint(sampled_neighbours_l)

    prompt = ''
    cnt = 0
    for i, (utterance, dist) in enumerate(sampled_neighbours_l):
        prompt += f'## Example {i+1}\n'

        example_df = train_df[train_df.utterance == utterance]
        # example_df = example_df[example_df.aty != 'none'].reset_index()
        
        class_l = []
        for k in example_df.iterrows():
            
            # if k[0] == 0:

            #     prompt += f'# Abstract:\n{example_df.iloc[0].utterance}\n\n# Arguments:\n'
            #     cnt = 0
                
            # prompt += f'Argument {cnt + 1}={k[1].text} - Class={k[1].aty}\n'
            prompt += f'Utterance {cnt + 1}={k[1].utterance}\n'
            class_l.append(k[1].emotions_list)
            cnt += 1
            
        prompt += '\n# Result:\n'
        prompt += '{' + ', '.join([f'"utterance_emotions": "{class_l[i]}"' for i in range(len(class_l))]) + '}'
        prompt += '\n\n'

    return prompt