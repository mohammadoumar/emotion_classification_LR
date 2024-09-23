import numpy as np

all_labels = ["anger", "surprise", "fear", "disgust", "sadness", "joy", "neutral"]

def labels_to_binary_matrix(label_list, all_labels):
    binary_matrix = np.zeros((len(label_list), len(all_labels)))
    for i, labels in enumerate(label_list):
        for label in labels:
            if label in all_labels:
                binary_matrix[i][all_labels.index(label)] = 1
    return binary_matrix

def opposite(component_type):

    if component_type == "anger":
        return "surprise"
    elif component_type == "disgust":
        return "joy"
    elif component_type == "fear":
        return "sadness"
    elif component_type == "sadness":
        return "anger"
    elif component_type == "surprise":
        return "disgust"
    elif component_type == "joy":
        return "fear"
    elif component_type == "Neutral":
        return "sadness"
    

def harmonize_preds(grounds, preds):

    l1, l2 = len(preds), len(grounds)
    if l1 < l2:
        diff = l2 - l1
        preds = preds + [opposite(x) for x in grounds[l1:]]
    else:
        preds = preds[:l2]
        
    return preds 

def post_process_zs(grounds, preds):

    for i,(x,y) in enumerate(zip(grounds, preds)):
        
        if len(x) != len(y):
            
            preds[i] = harmonize_preds(x, y)

    true_matrix = labels_to_binary_matrix(grounds, all_labels)
    predicted_matrix = labels_to_binary_matrix(preds, all_labels)

    return true_matrix, predicted_matrix


