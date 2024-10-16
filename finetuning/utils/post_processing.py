import json
from sklearn.preprocessing import MultiLabelBinarizer

def post_process(results):
    
    grounds, predictions = extract_results(results)
    predictions = harmonize_preds(predictions)
    grounds_matrix, predictions_matrix = get_mlb(grounds, predictions)
    
    return grounds_matrix, predictions_matrix

def extract_results(results):
    
    grounds = results["grounds"]
    predictions = results["predictions"]
    predictions = [x["content"] for x in predictions]   

    grounds_l = []
    for i in range(len(grounds)):
        grounds_l.append(json.loads(grounds[i])['list_emotion_classes'])    

    predictions_l = []
    for i in range(len(predictions)):
        predictions_l.append(json.loads(predictions[i])['list_emotion_classes'])
    
    
    return grounds_l, predictions_l

def harmonize_preds(predictions):
    
    # TODO
    
    return predictions
    
def get_mlb(grounds, predictions):
    
    mlb = MultiLabelBinarizer()
    grounds_mhot = mlb.fit_transform(grounds)
    predictions_mhot = mlb.transform(predictions)
    
    return grounds_mhot, predictions_mhot
    

# def opposite(component_type):

#     if component_type == "Anger":
#         return "Surprise"
#     elif component_type == "Disgust":
#         return "Joy"
#     elif component_type == "Fear":
#         return "Sadness"
#     elif component_type == "Sadness":
#         return "Anger"
#     elif component_type == "Surprise":
#         return "Disgust"
#     elif component_type == "Joy":
#         return "Fear"
#     elif component_type == "Neutral":
#         return "Joy"

# def harmonize_preds(grounds, preds):

#     l1, l2 = len(preds), len(grounds)
#     if l1 < l2:
#         diff = l2 - l1
#         preds = preds + [opposite(x) for x in grounds[l1:]]
#     else:
#         preds = preds[:l2]
        
#     return preds 

# def post_process(results):

#     grounds = results["ground_truths"]
#     preds = results["predictions"]
#     preds = [x["content"] for x in preds]   

#     grounds_l = []
#     for i in range(len(grounds)):
#         grounds_l.append(json.loads(grounds[i])['list_emotion_classes'])    

#     preds_l = []
#     for i in range(len(preds)):
#         preds_l.append(json.loads(preds[i])['list_emotion_classes'])

#     for i,(x,y) in enumerate(zip(grounds_l, preds_l)):
#         if len(x) != len(y):
            
#             preds_l[i] = harmonize_preds(x, y)

#     task_preds = [item for row in preds_l for item in row]
#     task_grounds = [item for row in grounds_l for item in row]

#     task_grounds = ['Neutral' if x == ['Neutral'] else x for x in task_grounds]
#     task_preds = ['Neutral' if x == ['Neutral'] else x for x in task_preds]

#     return task_grounds, task_preds

    