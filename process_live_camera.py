import cv2 as cv
import mediapipe as mp
import json
import numpy as np

def process_live_camera(hands):
    cap = cv.VideoCapture(0)  # Use the camera with index 0, you can change it if you have multiple cameras

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv.flip(frame, 1)
        # Process the frame and get hand landmarks
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            all_data = {}
            for handType, handLms in zip(results.multi_handedness, results.multi_hand_landmarks):
                my_lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    px, py = lm.x, lm.y
                    my_lm_list.append([px, py])

                # Normalize landmarks
                normalized_landmarks = normalize_keypoints(my_lm_list)

                # Create a dictionary for the current frame
                current_data = {
                    "label": "live_capture",
                    # "label_identifier": 2,
                    "landmarks": normalized_landmarks,
                }
                # Add the data to the overall dictionary
                all_data["live_capture"] = current_data

                # Display the landmarks on the frame (optional)
                mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            # Display the frame
            cv.imshow("Hand Landmarks", frame)

            # Write the current frame's data to a new JSON file
            with open("live_capture_landmarks.json", 'w') as json_file:
                json.dump(all_data, json_file, indent=2)

        if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the live capture
            break

    cap.release()
    cv.destroyAllWindows()


def normalize_keypoints(keypoints):
    x_coords, y_coords = zip(*keypoints)
    x_coords = (np.array(x_coords) - min(x_coords)) / (max(x_coords) - min(x_coords))
    y_coords = (np.array(y_coords) - min(y_coords)) / (max(y_coords) - min(y_coords))
    normalized_keypoints = list(zip(x_coords, y_coords))
    return normalized_keypoints

if __name__ == "__main__":
    hands_handler = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    process_live_camera(hands_handler)

    hands_handler.close()
    print("Finished live capture and processing.")
