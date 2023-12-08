import json
import numpy as np

def z_score_normalize_landmarks(file_path, output_file):
    # Read the content from the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Calculate the z-score for each coordinate in the landmarks
    for obj_id, obj in data.items():
        landmarks = obj.get("landmarks", [])
        if landmarks:
            # Convert landmarks to a NumPy array for efficient calculations
            landmarks_array = np.array(landmarks)

            # Calculate z-score for each coordinate
            z_score_landmarks = (landmarks_array - np.mean(landmarks_array, axis=0)) / np.std(landmarks_array, axis=0)

            # Normalize the z-scored landmarks to the range [0, 1]
            normalized_landmarks = (z_score_landmarks - np.min(z_score_landmarks, axis=0)) / (np.max(z_score_landmarks, axis=0) - np.min(z_score_landmarks, axis=0))

            # Update the object with the normalized landmarks
            obj["landmarks"] = normalized_landmarks.tolist()

    # Write the updated data to the output JSON file
    with open(output_file, 'w') as output:
        json.dump(data, output, indent=2)

# Replace 'selected_objects.json' and 'normalized_objects.json' with your file names
z_score_normalize_landmarks('selected_objects.json', 'normalized_objects.json')
