emotion_map = {
    'AN': 'anger',
    'DI': 'disgust',
    'FE': 'fear',
    'SA': 'sadness',
    'SU': 'surprise',
    'JO': 'joy'
}

def extract_emotions(row):

    emotion_str = row.emotion

    if emotion_str == 'Neutral':
        return ['neutral']

    emotions = emotion_str.split('-')
    tags = []

    for emotion in emotions:
        abbrev = emotion[:2]  # Get the abbreviation
        value_part = emotion[2:]  # Get the value part
        
        # Ensure that the value part is a valid integer and abbrev is in the emotion_map
        if abbrev in emotion_map and value_part.isdigit():
            value = int(value_part)
            if value > 0:
                tags.append(emotion_map[abbrev].lower())
        else:
            print(f"Warning: Skipping invalid emotion entry: '{emotion}'")
    return tags  