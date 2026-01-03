import cv2
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# DEBUG: Confirm TensorFlow model loads
print("Loading model...")
model = load_model("emotion_model.h5")
print("Model loaded successfully ✅")
# ----------------------------

# Emotion labels (must match your dataset)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Emotion colors (for bounding boxes)
emotion_colors = {
    'Angry': (0,0,255),
    'Disgust': (0,255,0),
    'Fear': (255,0,0),
    'Happy': (0,255,255),
    'Neutral': (255,255,255),
    'Sad': (255,0,255),
    'Surprise': (0,165,255)
}

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("Haar cascade loaded ✅")

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Cannot open webcam")
else:
    print("Webcam opened ✅")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # DEBUG: Show number of faces detected
    print(f"Faces detected: {len(faces)}")

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        # Prediction
        prediction = model.predict(face, verbose=0)
        max_index = np.argmax(prediction)
        emotion = emotion_labels[max_index]
        confidence = prediction[0][max_index]

        # DEBUG: Show prediction and confidence
        print(f"Detected emotion: {emotion}, Confidence: {confidence:.2f}")

        # Draw bounding box and label
        color = emotion_colors.get(emotion, (0,255,0))
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{emotion} ({confidence*100:.1f}%)",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            
        )

    cv2.imshow("Face Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed ✅")
