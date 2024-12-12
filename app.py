

from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

app = Flask(__name__)

# Load the trained emotion recognition model (CNN or ResNet50V2)
emotion_model = load_model('models/ResNet50V2_Model.keras')  # Update with your model file

# Load the face detection classifier (Haar Cascades or other)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')  # Update with your classifier

# Define a list of emotion labels for reference
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Calm', 'Energetic']

# Load a sample music dataset for recommendations (e.g., CSV with name, artist, mood, and popularity)
music_data = pd.read_csv('data/data_moods.csv')  # Update with your dataset

cap = cv2.VideoCapture(0)  # Open a connection to the camera

def recommend_music(emotion):
    # Filter music based on the mood category
    filtered_music = music_data[music_data['mood'] == emotion]
    return filtered_music

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/discography')
def discography():
    return render_template('discography.html')

@app.route('/tours')
def tours():
    return render_template('tours.html')

@app.route('/videos')
def videos():
    return render_template('videos.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/process_emotion', methods=['POST'])
def process_emotion():
    try:
        # Capture a frame from the camera
        ret, frame = cap.read()
        if not ret:
            return jsonify({"error": "Failed to capture frame from the camera."})

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use the face detection classifier to detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        emotion_results = []

        for (x, y, w, h) in faces:
            face_roi = gray[y:y + h, x:x + w]
            # Convert grayscale to a 3-channel image (RGB)
            face_img = cv2.cvtColor(face_roi, cv2.COLOR_GRAY2BGR)
            face_img = cv2.resize(face_img, (224, 224))
            face_img = np.reshape(face_img, [1, 224, 224, 3])  # Prepare the face image for prediction
            face_img = face_img / 255.0  # Normalize pixel values

            emotion_prediction = emotion_model.predict(face_img)
            predicted_emotion = emotion_labels[np.argmax(emotion_prediction)]

            # Get the music recommendations for the predicted emotion
            music_recommendations = recommend_music(predicted_emotion)

            # Convert int32 values to regular integers
            x, y, w, h = int(x), int(y), int(w), int(h)

            emotion_results.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "emotion": predicted_emotion,
                "music_recommendations": music_recommendations.to_dict(orient='records')
            })

        # Return the results as JSON to the web page
        return jsonify(emotion_results)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)

# Release the camera and close the window when the app is stopped
cap.release()
cv2.destroyAllWindows()
