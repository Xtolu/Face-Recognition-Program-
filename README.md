# Face-Recognition-Program-
Face detector 
import argparse
import face_recognition
import cv2
import numpy as np
import pickle
import os
from pathlib import Path
from PIL import Image, ImageDraw
from collections import Counter

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Ensure directories exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """Train: Generate face encodings from training images."""
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        if filepath.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
            name = filepath.parent.name
            image = face_recognition.load_image_file(filepath)
            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)
            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)
    name_encodings = {"names": names, "encodings": encodings}
    encodings_location.parent.mkdir(exist_ok=True)
    with encodings_location.open("wb") as f:
        pickle.dump(name_encodings, f)
    print(f"Encoded {len(names)} faces and saved to {encodings_location}")

def _recognize_face(unknown_encoding, loaded_encodings):
    """Helper: Vote on best match."""
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)
    return votes.most_common(1)[0][0] if votes else None

def recognize_faces(image_location, model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """Test: Recognize in image and draw boxes."""
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)
    input_image = face_recognition.load_image_file(image_location)
    input_face_locations = face_recognition.face_locations(input_image, model=model)
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)
    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)
    for bounding_box, unknown_encoding in zip(input_face_locations, input_face_encodings):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        name = name if name else "Unknown"
        top, right, bottom, left = bounding_box
        draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
        text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
        draw.rectangle(((text_left, text_top), (text_right, text_bottom)), fill=BOUNDING_BOX_COLOR)
        draw.text((text_left, text_top), name, fill=TEXT_COLOR)
    pillow_image.show()
    print(f"Recognized faces in {image_location}")

def live_recognition(model="hog", encodings_location=DEFAULT_ENCODINGS_PATH):
    """Live: Webcam face recognition."""
    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)
    video_capture = cv2.VideoCapture(0)
    while True:
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]  # BGR to RGB
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = _recognize_face(face_encoding, loaded_encodings)
            name = name if name else "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
        cv2.imshow('Live Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Face Recognition")
    parser.add_argument("--train", action="store_true", help="Train model from training/ folder")
    parser.add_argument("--test", "-f", help="Test on image file")
    parser.add_argument("--live", action="store_true", help="Live webcam recognition")
    parser.add_argument("-m", choices=["hog", "cnn"], default="hog", help="Model: hog (CPU) or cnn (GPU)")
    args = parser.parse_args()
    if args.train:
        encode_known_faces(model=args.m)
    elif args.test:
        recognize_faces(args.test, model=args.m)
    elif args.live:
        live_recognition(model=args.m)
    else:
        parser.print_help()
