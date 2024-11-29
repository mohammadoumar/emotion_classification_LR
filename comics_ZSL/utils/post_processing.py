import json
from sklearn.preprocessing import MultiLabelBinarizer

def post_process(results):
    
    grounds, predictions = extract_results(results)
    grounds_matrix, predictions_matrix, classes = get_mlb(grounds, predictions)
    
    return grounds_matrix, predictions_matrix, classes

def extract_results(results):
    
    grounds = results["grounds"]
    raw_predictions = results["predictions"]
    
    predictions = []

    for pred in raw_predictions:
        
        predictions.append(pred.split("\n\n")[-1])
        
    predictions_l = []
    non_matching_indices = []

    for i, pred in enumerate(predictions):
        try:
            clean_pred = pred.strip('```json\n').strip('```')
            predictions_l.append(json.loads(clean_pred)["list_emotion_classes"])
        except:
            print(i)
            non_matching_indices.append(i)  
            
    non_matching_indices.sort(reverse=True)
    for idx in non_matching_indices:
    
        del grounds[idx]
    
    
    return grounds, predictions_l

    
def get_mlb(grounds, predictions):
    
    mlb = MultiLabelBinarizer()
    grounds_mhot = mlb.fit_transform(grounds)
    predictions_mhot = mlb.transform(predictions)
    
    return grounds_mhot, predictions_mhot, mlb.classes_