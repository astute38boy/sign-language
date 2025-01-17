from flask import Flask, render_template, jsonify, request, Response, redirect, url_for
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model # type: ignore
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
video_running = True
prediction_confirmed = False


# Extract keypoints from landmarks
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# Route to serve the sign-in page
@app.route('/', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        # Get username and password from the form
        username = request.form.get('username')
        password = request.form.get('password')

        # Simple authentication logic (replace with actual authentication)
        if username == 'admin' and password == 'password':  # Replace with your authentication logic
            return redirect(url_for('main'))  # Redirect to the main page on successful login
        else:
            error = 'Invalid username or password. Please try again.'
            return render_template('sign-in.html', error=error)
    return render_template('sign-in.html')

@app.route('/main', methods=['GET', 'POST'])
def main():
    # Handle POST request logic if needed (e.g., authentication checks)
    return render_template('main.html')

# Route to handle the webcam feed
# Global variable to store the patient's response
patient_response = None  # None, 'yes', or 'no'

def generate_frames():
    global sequence, predicted_action, confidence, video_running
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while video_running:  # Only run while video_running is True
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

            # Add prediction text to the video frame
            if predicted_action:
                text = f'{predicted_action} ({confidence:.2f})' if confidence >= confidence_threshold else 'Uncertain'
                cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0) if confidence >= confidence_threshold else (0, 0, 255), 2, cv2.LINE_AA)

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()

            # Yield the frame as part of the video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()


@app.route('/get_prediction', methods=['GET'])
def get_prediction():
    global predicted_action, confidence, prediction_confirmed

    if prediction_confirmed:
        return jsonify({'action': None, 'message': "Prediction process completed. No further predictions."})

    if predicted_action and confidence >= confidence_threshold:
        return jsonify({'action': predicted_action, 'message': f"Predicted disease: {predicted_action}. Is this correct?"})
    else:
        return jsonify({'action': None, 'message': "Waiting for a confident prediction..."})


@app.route('/confirm_response', methods=['POST'])
def confirm_response():
    global sequence, predicted_action, video_running, prediction_confirmed
    data = request.json
    patient_response = data.get('response')

    if patient_response == 'yes':
        # Patient confirms the prediction
        video_running = False  # Stop video capture if the predicted disease is correct
        prediction_confirmed = True  # Mark prediction as confirmed
        response = {
            'status': 'confirmed',
            'message': f"The disease is confirmed as {predicted_action}.",
            'disease': predicted_action
        }
    elif patient_response == 'no':
        # Patient rejects the prediction, reset the sequence
        sequence = []
        response = {
            'status': 'rejected',
            'message': "Prediction rejected. Please try again."
        }
    else:
        response = {
            'status': 'error',
            'message': "Invalid response. Please respond with 'yes' or 'no'."
        }

    return jsonify(response)




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
