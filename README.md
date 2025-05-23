﻿# Hand Gesture Recognition

A computer vision project that recognizes hand gestures in real-time using MediaPipe and TensorFlow.

## Overview

This project uses machine learning to recognize different types of hand gestures:
- Sign language alphabets (A-Z)
- Number gestures (0-9)
- Various hand gestures (peace, thumbs up, etc.)

The system captures hand landmarks using MediaPipe's hand tracking solution and uses a trained neural network to classify the gestures.

## Project Structure

- `run_gesture_recognition.py` - Standalone script to run the recognition system
- `Hand_gesture_model/` - Main project directory
  - `Alphabets/` - Alphabet sign language recognition
  - `Numbers/` - Number gesture recognition
  - `Gestures/` - Various hand gestures recognition
  - `*.ipynb` - Jupyter notebooks for model development and testing

## Requirements

- Python 3.6+
- OpenCV
- TensorFlow
- MediaPipe
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hand_gesture_recognition.git
cd hand_gesture_recognition

# Install dependencies
pip install numpy opencv-python tensorflow mediapipe
```

## Usage

Run the standalone script:

```bash
python run_gesture_recognition.py
```

When prompted, choose which type of gestures to recognize:
- `alphabets` - for sign language alphabet recognition
- `numbers` - for number gesture recognition
- `gestures` - for various hand gestures

You can also specify the model directly:

```bash
python run_gesture_recognition.py alphabets
```

The system will open your webcam and begin recognizing hand gestures in real-time. Press 'q' to quit.

## Model Training

The models were trained using hand landmark data extracted using MediaPipe. The Jupyter notebooks in each directory contain the training process and evaluation.

## License

MIT License
