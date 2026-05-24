import cv2
from tensorflow.keras.models import model_from_json
import numpy as np

# Load model JSON
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()

model = model_from_json(model_json)

# Load weights
model.load_weights("emotiondetector.h5")

# Load Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Feature extraction
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Start webcam
webcam = cv2.VideoCapture(0)

# ✅ FIXED labels — matching your model training order
# Your model: ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',       # ✅ FIXED (was neutral)
    5: 'surprise',  # ✅ FIXED (was sad)
    6: 'neutral'    # ✅ FIXED (was surprise)
}

# Emotion colors (BGR)
colors = {
    'angry':    (0,   0,   255),
    'disgust':  (0,   140, 0),
    'fear':     (128, 0,   128),
    'happy':    (0,   215, 255),
    'neutral':  (255, 255, 255),
    'sad':      (255, 100, 0),
    'surprise': (0,   165, 255),
}

while True:
    ret, im = webcam.read()
    if not ret:
        break

    # ✅ Histogram equalization for better contrast
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # ✅ Better face detection
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    for (x, y, w, h) in faces:

        # ✅ Small padding
        pad = int(0.1 * w)
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(im.shape[1], x + w + pad)
        y2  = min(im.shape[0], y + h + pad)

        face = gray[y1:y2, x1:x2]
        face = cv2.resize(face, (48, 48))

        # ✅ CLAHE for local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face  = clahe.apply(face)

        img  = extract_features(face)
        pred = model.predict(img, verbose=0)[0]

        # ✅ Temperature scaling — reduces happy/neutral bias
        temperature = 0.5
        pred_scaled = np.exp(np.log(pred + 1e-7) / temperature)
        pred_scaled = pred_scaled / pred_scaled.sum()

        label      = labels[pred_scaled.argmax()]
        confidence = int(pred_scaled.max() * 100)
        color      = colors.get(label, (0, 255, 0))

        # Draw box
        cv2.rectangle(im, (x, y), (x + w, y + h), color, 2)

        # Label + confidence
        cv2.putText(im,
                    f'{label.upper()}  {confidence}%',
                    (x, y - 12),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL,
                    1.2, color, 2)

        # ✅ Top 3 emotions on right side
        sorted_emos = sorted(
            enumerate(pred_scaled),
            key=lambda e: e[1],
            reverse=True
        )[:3]

        bar_x = x + w + 8
        bar_y = y
        for i, (idx, score) in enumerate(sorted_emos):
            ename  = labels[idx]
            escore = int(score * 100)
            ecolor = colors.get(ename, (200, 200, 200))

            cv2.putText(im,
                        f'{ename[:3].upper()} {escore}%',
                        (bar_x, bar_y + i * 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, ecolor, 1)
            cv2.rectangle(im,
                          (bar_x, bar_y + i * 24 + 4),
                          (bar_x + int(score * 80), bar_y + i * 24 + 12),
                          ecolor, -1)

    cv2.imshow("Emotion Detection", im)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()