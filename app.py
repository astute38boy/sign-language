from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('action.h5')

# Mediapipe initialization
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Actions for prediction
actions = np.array(['coughing', 'headache', 'sorethroat'])
threshold = 0.5  # Lower threshold for testing

# Webcam capture
cap = cv2.VideoCapture(0)

# Global variables for predictions
sequence = []  # Stores the last 30 frames
sentence = []  # Stores the predicted actions
predictions = []  # Stores model prediction indices

# Extract keypoints from landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Debugging Keypoints
    print(f"Pose keypoints shape: {pose.shape}, Face keypoints shape: {face.shape}")
    return np.concatenate([pose, face, lh, rh])

# Sets up home route for Flask
@app.route('/')
def index():
    return render_template('main.html')

# Video Feed Route
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            global sequence, sentence, predictions
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame and make predictions
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Extract keypoints and append to sequence
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(f"Raw Model Predictions: {res}")  # Debugging prediction outputs
                    print(f"Max Prediction Confidence: {max(res)}")  # Confidence score

                    predictions.append(np.argmax(res))

                    # Prediction threshold logic
                    if res[np.argmax(res)] > threshold:
                        if len(sentence) > 0:
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                print(f"Detected Action: {actions[np.argmax(res)]}")
                        else:
                            sentence.append(actions[np.argmax(res)])
                            print(f"Detected Action: {actions[np.argmax(res)]}")
                    else:
                        print("Prediction below threshold, no action detected.")

                    if len(sentence) > 5:
                        sentence = sentence[-5:]

                # Encode frame and yield
                _, buffer = cv2.imencode('.jpg', image)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predict', methods=['POST'])
def predict():
    if len(sentence) > 0:
        action = sentence[-1]
        confidence = max(model.predict(np.expand_dims(sequence, axis=0))[0])
        return jsonify({'action': action, 'confidence': str(confidence)})
    else:
        return jsonify({'action': 'No action detected', 'confidence': '0.0'})

# Mediapipe detection
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

# Draw landmarks on the frame
def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

if __name__ == "__main__":
    app.run(debug=True)
