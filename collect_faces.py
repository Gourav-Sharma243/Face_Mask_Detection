import cv2
import os
from mtcnn.mtcnn import MTCNN

# Change this before each run: "face_with_mask", "face_no_mask", or "uncertain"
LABEL = "face_with_mask"

SAVE_DIR = f"dataset/{LABEL}"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)
detector = MTCNN()
img_size = 50
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_frame)

    for face in faces:
        if face['confidence'] < 0.95:
            continue

        x, y, w, h = face['box']
        x, y = max(0, x), max(0, y)
        face_img = frame[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (img_size, img_size))

        file_path = os.path.join(SAVE_DIR, f"{LABEL}_{count}.png")
        cv2.imwrite(file_path, resized_face)
        count += 1

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f"{LABEL} {count}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("Collecting Face Data", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
