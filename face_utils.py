from pathlib import Path
import csv
import cv2
import numpy as np

DATA = Path("data")
FACES = DATA / "faces"
STUDENTS = DATA / "students.csv"
MODEL = DATA / "lbph_model.yml"
LABELS = DATA / "labels.csv"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def ensure_data():
    DATA.mkdir(exist_ok=True)
    FACES.mkdir(parents=True, exist_ok=True)
    if not STUDENTS.exists():
        with STUDENTS.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["student_id", "student_name"])
    if not LABELS.exists():
        with LABELS.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["label", "student_id", "student_name"])

def read_students():
    ensure_data()
    with STUDENTS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def add_student(student_id, name):
    ensure_data()
    student_id, name = student_id.strip(), name.strip()
    if not student_id or not name:
        raise ValueError("Student ID and name are required.")
    students = read_students()
    if any(s["student_id"].lower() == student_id.lower() for s in students):
        raise ValueError("That student ID already exists.")
    with STUDENTS.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([student_id, name])
    (FACES / student_id).mkdir(exist_ok=True)

def _detector():
    return cv2.CascadeClassifier(CASCADE_PATH)

def save_face_photo(student_id, image_bgr, photo_index):
    ensure_data()
    folder = FACES / student_id
    folder.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _detector().detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    if len(faces) == 0:
        raise ValueError("No face detected. Try a clear, front-facing photo with good lighting.")
    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
    crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    path = folder / f"{student_id}_{photo_index}.png"
    cv2.imwrite(str(path), crop)
    return str(path)

def train_model():
    ensure_data()
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    images, labels = [], []
    student_map = {}
    label = 0
    for student in read_students():
        sid = student["student_id"]
        folder = FACES / sid
        if not folder.exists():
            continue
        student_images = list(folder.glob("*.png"))
        if not student_images:
            continue
        student_map[label] = (sid, student["student_name"])
        for path in student_images:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(cv2.resize(img, (200, 200)))
                labels.append(label)
        label += 1
    if not images:
        raise ValueError("No face photos found. Enroll at least one student first.")
    recognizer.train(images, np.array(labels, dtype=np.int32))
    recognizer.write(str(MODEL))
    with LABELS.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "student_id", "student_name"])
        for lab, (sid, name) in student_map.items():
            writer.writerow([lab, sid, name])
    return len(images), len(student_map)

def load_model():
    if not MODEL.exists() or not LABELS.exists():
        raise FileNotFoundError("Train the model first in Student Enrollment.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL))
    with LABELS.open(newline="", encoding="utf-8") as f:
        label_map = {int(r["label"]): r for r in csv.DictReader(f)}
    return recognizer, label_map

def recognize_frame(frame, threshold=75):
    recognizer, label_map = load_model()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _detector().detectMultiScale(gray, 1.2, 5, minSize=(80, 80))
    results = []
    for (x, y, w, h) in faces:
        crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label, confidence = recognizer.predict(crop)
        if confidence <= threshold and label in label_map:
            row = label_map[label]
            name = f'{row["student_name"]} ({row["student_id"]})'
            sid = row["student_id"]
        else:
            name, sid = "Unknown", None
        results.append((x, y, w, h, name, sid, float(confidence)))
    return results
