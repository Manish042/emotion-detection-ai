import cv2
import numpy as np
from tensorflow.keras.models import model_from_json

# Load model
with open("emotiondetector.json", "r") as f:
    model = model_from_json(f.read())
model.load_weights("emotiondetector.h5")

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# ── Dono possible label orders ──
labels_A = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad',     6:'surprise'}
labels_B = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad',     5:'surprise', 6:'neutral'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

webcam = cv2.VideoCapture(0)

print("\n" + "="*60)
print("  LABEL FINDER TEST")
print("="*60)
print("  KARO: Khuslh feel karo — SMILE karo camera pe")
print("  Dekho: Kaun sa label HAPPY dikhata hai")
print("  Q = quit")
print("="*60 + "\n")

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))

    display = frame.copy()

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img  = extract_features(face)
        pred = model.predict(img, verbose=0)[0]

        idx_max = pred.argmax()

        # ── Terminal output ──
        print(f"\nRaw predictions (index → score):")
        for i, score in enumerate(pred):
            bar = '█' * int(score * 40)
            mark = ' ◀ MAX' if i == idx_max else ''
            print(f"  [{i}] A={labels_A[i]:8s} | B={labels_B[i]:8s} : {score*100:5.1f}% {bar}{mark}")

        print(f"\n  Order A says: {labels_A[idx_max].upper()}")
        print(f"  Order B says: {labels_B[idx_max].upper()}")
        print(f"  Aap actually kya feel kar rahe ho? (smile=happy)")
        print("-"*60)

        # ── On screen ──
        cv2.rectangle(display, (x,y), (x+w,y+h), (0,255,0), 2)

        # Show both predictions
        cv2.putText(display, f"A: {labels_A[idx_max].upper()}",
                    (x, y-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.putText(display, f"B: {labels_B[idx_max].upper()}",
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,100,0), 2)

        # Raw index
        cv2.putText(display, f"Raw index: {idx_max}",
                    (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

        # All scores on left
        for i, score in enumerate(pred):
            cv2.putText(display,
                        f"[{i}] {score*100:.0f}%",
                        (5, 20 + i*20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0,255,0) if i==idx_max else (150,150,150), 1)

    cv2.imshow("Label Finder — Q to quit", display)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()