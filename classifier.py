import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

class Classifier:
	def __init__(self) -> None:
		self.classifier = SVC()

	def normalize_keypoints(self, keypoints):
		x_coords, y_coords = zip(*keypoints)
		x_coords = (np.array(x_coords) - min(x_coords)) / (max(x_coords) - min(x_coords))
		y_coords = (np.array(y_coords) - min(y_coords)) / (max(y_coords) - min(y_coords))
		normalized_keypoints = list(zip(x_coords, y_coords))
		return normalized_keypoints

	def flatten_landmarks(self, landmarks):
		flattened_landmarks = [np.array(landmark).flatten() for landmark in landmarks]
		return flattened_landmarks

	def predict_label(self,new_landmarks):
		normalized_landmarks = self.normalize_keypoints(new_landmarks)
		flattened_landmarks = self.flatten_landmarks(normalized_landmarks)
		flattened_landmarks = np.array(flattened_landmarks).reshape(1, -1)
		# print("hi") 
		prediction = self.classifier.predict(flattened_landmarks)
		return prediction[0]

	def train_model(self, data):
		X, y = [], []
		for filename, details in data.items():
				landmarks = details["landmarks"]
				normalized_landmarks = self.normalize_keypoints(landmarks)
				flattened_landmarks = self.flatten_landmarks(normalized_landmarks)
				
				X.append(flattened_landmarks)
				y.append(details["label_identifier"])

		X = np.array(X)
		y = np.array(y)

		X = X.reshape(X.shape[0], -1)

		# Split the data into training and testing sets
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

		self.classifier = SVC()
		self.classifier.fit(X_train, y_train)

		y_pred =self.classifier.predict(X_test)

		# Evaluate the accuracy of the classifier
		accuracy = accuracy_score(y_test, y_pred)
		f1score = f1_score(y_test,y_pred)
		print(f"Classifier Accuracy: {accuracy}")
		print(f"Classifier F1 score: {f1score}")