import ast
import json
from sklearn.preprocessing import MultiLabelBinarizer


def process_raw_predictions(raw_predictions):
    
    predictions_l = []

    for i, prediction in enumerate(raw_predictions):
            try:
                # Use json.loads to safely parse the JSON-like string
                parsed_prediction = json.loads(prediction)
                # Append the values of the parsed prediction to preds
                predictions_l.append(parsed_prediction["list_emotion_classes"])
                
            except json.JSONDecodeError as e:
                print(f"Error decoding prediction: {i}")
    
    predictions = []

    for item in predictions_l:
        if isinstance(item, str):
            # Convert the string to a list using ast.literal_eval
            predictions.append(ast.literal_eval(item))
        else:
            # If the item is already a list, append as is
            predictions.append(item)
            
    return predictions

def get_mlb(grounds, predictions):
    
    mlb = MultiLabelBinarizer()
    grounds_mhot = mlb.fit_transform(grounds)
    predictions_mhot = mlb.transform(predictions)
    
    return grounds_mhot, predictions_mhot, mlb.classes_
                

def post_process_icl(results):
    
    grounds = results["grounds"]
    raw_predictions = results["predictions"]
    predictions = process_raw_predictions(raw_predictions)
    grounds_matrix, predictions_matrix, classes = get_mlb(grounds, predictions)

    return grounds_matrix, predictions_matrix, classes