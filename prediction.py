import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def load_data(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def prepare_data(data):
    labels = []
    landmarks = []

    for _, obj in data.items():
        label = obj.get("label_identifier")
        if label is not None:
            labels.append(label)
            landmarks.append(obj.get("landmarks"))

    # Flatten the landmarks array
    landmarks_flat = [lm for sublist in landmarks for lm in sublist]

    # Print the shape of landmarks_flat before trimming or padding
    print(f"Shape before trimming or padding: {len(landmarks_flat)}")

    # Trim or pad the landmarks array to make it divisible by 20
    # landmarks_flat = landmarks_flat[:len(landmarks_flat) // 20 * 20]


    # TODO: DO NOT TRIM THE LANDMARK.
    # Print the shape of landmarks_flat after trimming or padding
    # print(f"Shape after trimming or padding: {len(landmarks_flat)}")

    return np.array(landmarks_flat), np.array(labels)

def train_knn_model(X_train, y_train, k=3):
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    return knn_model

def predict_label(model, live_capture_landmarks):
    prediction = model.predict(np.array([live_capture_landmarks]))
    return prediction[0] if prediction else None

if __name__ == "__main__":
    # Load normalized landmarks data
    normalized_json_file_path = "normalized_objects.json"  # Update with your file path
    normalized_data = load_data(normalized_json_file_path)
    print(normalized_data)
    # Prepare normalized data
    X, y = prepare_data(normalized_data)
    print(f"Shape before trimming or padding: {X.shape[1]}")
    print(X)
    # Reshape landmarks to 2D array
    # X = reshape_landmarks(X)
    # y = reshape_landmarks(y)
    # Check the shapes of X and y
    print(f"Shape of X: {X.shape}")
    # print(f"Shape of y: {y.shape}")
    print(f"Number of unique labels in y: {len(np.unique(y))}")

    # Train KNN model
    k_value = 3  # You can adjust the value of k
    knn_model = train_knn_model(X, y, k=k_value)

    # Load live capture data
    live_capture_json_file_path = "live_capture_landmarks.json"  # Update with your file path
    live_capture_data = load_data(live_capture_json_file_path)

    # Extract live capture landmarks
    live_capture_landmarks = live_capture_data["live_capture"].get("landmarks")
    
    print(live_capture_landmarks)
    if live_capture_landmarks:
        # Reshape live capture landmarks to 2D array
        live_capture_landmarks = reshape_landmarks(np.array(live_capture_landmarks))

        # Predict the label for live capture data
        predicted_label = predict_label(knn_model, live_capture_landmarks)
        print(f"Prediction for live capture: {predicted_label}")
    else:
        print("Live capture landmarks not found.")

    print("Finished predicting label for live capture data.")
