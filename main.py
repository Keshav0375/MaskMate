from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('masknet.h5')

def predict_mask(face_roi, model, input_shape=(224, 224)):
    # Preprocess the face region
    face_roi = cv2.resize(face_roi, input_shape)
    face_roi = np.reshape(face_roi, [1, *input_shape, 3]) / 255.0

    # Make predictions using the model
    mask_result = model.predict(face_roi)
    label = mask_result.argmax()

    return label

import cv2
import numpy as np
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(rgb_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        face_roi = rgb_frame[y:y+h, x:x+w]

        # Make predictions
        label = predict_mask(face_roi, model)

        # Draw rectangle around the face and label it
        color = (0, 255, 0) if label == 0 else (0, 0, 255)  # Green for MASK, Red for NO MASK
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, 'MASK' if label == 0 else 'NO MASK', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow("Webcam", frame)

    # Break the loop when 'x' is pressed or the window is closed
    if cv2.waitKey(1) & 0xFF == ord('x') or cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
