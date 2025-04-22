import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import os
import sys

# Check if a model type was provided as a command line argument
if len(sys.argv) > 1:
    model_type = sys.argv[1].lower()
else:
    print("Available models:")
    print("1. alphabets - Sign language alphabet recognition")
    print("2. numbers - Number gesture recognition")
    print("3. gestures - Various hand gestures recognition")
    model_type = input("Enter model type (alphabets/numbers/gestures): ").lower()

# Setup based on the model type
if model_type == "alphabets":
    MODEL_PATH = "Hand_gesture_model/Alphabets/model_alpha.h5"
    alphabets = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    classes = alphabets
elif model_type == "numbers":
    MODEL_PATH = "Hand_gesture_model/Numbers/model_numbers.h5"
    numbers = [str(i) for i in range(10)]
    classes = numbers
elif model_type == "gestures":
    MODEL_PATH = "Hand_gesture_model/Gestures/model_gestures.h5"
    gestures = ["peace", "thumbs up", "thumbs down", "fist", "okay"]  # Update with actual gesture names
    classes = gestures
else:
    print("Invalid model type. Choose from: alphabets, numbers, gestures")
    sys.exit(1)

# Check if model exists
if not os.path.exists(MODEL_PATH):
    # Try alternative path
    alt_path = MODEL_PATH.replace("Hand_gesture_model/", "")
    if os.path.exists(alt_path):
        MODEL_PATH = alt_path
    else:
        print(f"Error: Model file not found at {MODEL_PATH}")
        print(f"Checking for model at: {os.path.abspath(MODEL_PATH)}")
        print(f"Please ensure the model file exists in the correct location")
        sys.exit(1)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Setup MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

print(f"Starting hand gesture recognition for {model_type.upper()}")
print("Press 'q' to quit")

# Start hands detection
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Flip the image horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hands
        results = hands.process(frame_rgb)
        
        # Draw the hand annotations on the image
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x)
                    landmarks.append(landmark.y)
                    landmarks.append(landmark.z)
                landmarks = np.array(landmarks)
                
                # Normalize the landmarks
                landmarks = landmarks.reshape(1, -1)
                landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
                
                # Make prediction
                predictions = model.predict(landmarks, verbose=0)  # Set verbose=0 to suppress output
                predicted_index = np.argmax(predictions)
                
                # Get the predicted class
                if predicted_index < len(classes):
                    predicted_gesture = classes[predicted_index]
                else:
                    predicted_gesture = "Unknown"
                
                # Display the prediction
                cv2.putText(frame, f'Predicted: {predicted_gesture}', (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('Hand Gesture Recognition', frame)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Hand gesture recognition stopped") 