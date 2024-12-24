from flask import Flask, render_template, jsonify, request, Response
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = load_model('action.h5')  # Replace with your model path

# Mediapipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions for prediction
actions = np.array(['coughing', 'headache', 'sorethroat'])
confidence_threshold = 0.8
sequence_length = 30
sequence = []
predicted_action = None
confidence = 0

# Extract keypoints from landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Route to serve the main HTML page
@app.route('/')
def index():
    return render_template('main.html')

# Route to handle the webcam feed
def generate_frames():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        global sequence, predicted_action, confidence
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with Mediapipe
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints and append to sequence
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:]

            # Perform prediction if sequence is ready
            if len(sequence) == sequence_length:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predicted_action = actions[np.argmax(res)]
                confidence = np.max(res)

                if confidence >= confidence_threshold:
                    cv2.putText(image, f'{predicted_action} ({confidence:.2f})', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, 'Uncertain', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Send both the video frame and prediction as response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/disease_prediction')
def disease_prediction():
    global predicted_action, confidence
    return jsonify({'disease': predicted_action, 'confidence': confidence})

# Helper functions
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

if __name__ == '__main__':
    app.run(debug=True)
