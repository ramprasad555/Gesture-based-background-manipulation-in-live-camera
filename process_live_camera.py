import cv2 as cv
import mediapipe as mp
import json
from classifier import Classifier

def process_live_camera(hands, model):
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
                    

                if my_lm_list == []:
                    print("No hand detected")
                    continue
                predicted_label = model.predict_label(my_lm_list )
                print(f"Predicted Label Identifier: {predicted_label}")

                # Display the landmarks on the frame (optional)
                mp.solutions.drawing_utils.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

            # Display the frame
            cv.imshow("Hand Landmarks", frame)

        if cv.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit the live capture
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    hands_handler = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    with open("data.json", 'r') as f:
        data = json.load(f)

    model = Classifier()
    model.train_model(data)

    print("model trained")
    process_live_camera(hands_handler, model)

    hands_handler.close()
    print("Finished live capture and processing.")
