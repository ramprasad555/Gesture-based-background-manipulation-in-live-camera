import numpy as np
import json

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def normalize_keypoints(keypoints):
    x_coords, y_coords = zip(*keypoints)
    x_coords = (np.array(x_coords) - min(x_coords)) / (max(x_coords) - min(x_coords))
    y_coords = (np.array(y_coords) - min(y_coords)) / (max(y_coords) - min(y_coords))
    normalized_keypoints = list(zip(x_coords, y_coords))
    return normalized_keypoints

def flatten_landmarks(landmarks):
    flattened_landmarks = [np.array(landmark).flatten() for landmark in landmarks]
    return flattened_landmarks

def predict_label(new_landmarks, classifier):
    normalized_landmarks = normalize_keypoints(new_landmarks)
    flattened_landmarks = flatten_landmarks(normalized_landmarks)
    flattened_landmarks = np.array(flattened_landmarks).reshape(1, -1)
    print("hi")
    prediction = classifier.predict(flattened_landmarks)
    return prediction[0]

def train_model(data):
    # Assuming 'data' is your input data in the format provided
    X, y = [], []

    for filename, details in data.items():
        landmarks = details["landmarks"]
        normalized_landmarks = normalize_keypoints(landmarks)
        flattened_landmarks = flatten_landmarks(normalized_landmarks)
        
        X.append(flattened_landmarks)
        y.append(details["label_identifier"])

    X = np.array(X)
    y = np.array(y)
    X = X.reshape(X.shape[0], -1)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    classifier = SVC()
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate the accuracy of the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classifier Accuracy: {accuracy}")
    return classifier

new_landmarks = [
      [
        0.0,
        0.8653271001052925
      ],
      [
        0.10806256409071716,
        0.6007275826391721
      ],
      [
        0.21708384188796012,
        0.37681702458136734
      ],
      [
        0.24815572744808892,
        0.16509655380204089
      ],
      [
        0.21393561602168465,
        0.0
      ],
      [
        0.7536115399868322,
        0.4336116686828698
      ],
      [
        0.987442711401506,
        0.5044836267269022
      ],
      [
        0.9286935825218919,
        0.5724988968596493
      ],
      [
        0.8236146458598841,
        0.5885080300527725
      ],
      [
        0.8526176606102143,
        0.6074173914112968
      ],
      [
        1.0,
        0.6666351624646704
      ],
      [
        0.8955573181171222,
        0.7126006481973118
      ],
      [
        0.7596830726068414,
        0.7249904155535785
      ],
      [
        0.872168618104673,
        0.7833252797017374
      ],
      [
        0.9492372643162138,
        0.8232113094691963
      ],
      [
        0.8480323267481125,
        0.8571006229084482
      ],
      [
        0.7110177637970709,
        0.8576316561587985
      ],
      [
        0.8460735090846787,
        0.9459712801380608
      ],
      [
        0.8705412432989506,
        0.9904058852743243
      ],
      [
        0.7659166368271838,
        1.0
      ],
      [
        0.6519317375306256,
        0.9809650164581503
      ]
    ]

with open("output.json", 'r') as f:
    data = json.load(f)

pl = predict_label(new_landmarks, train_model(data))
print(f"Predicted Label Identifier: {pl}")
