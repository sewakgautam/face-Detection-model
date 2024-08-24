import pickle
import cv2
import numpy as np
import face_recognition
import pyttsx3
import datetime
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from playsound import playsound

# Initialize speech engine based on the platform
def init_speech_engine():
    try:
        engine = pyttsx3.init('espeak')  # 'espeak' is for Linux
        voices = engine.getProperty('voices')
        engine.setProperty('voice', voices[1].id)
        engine.setProperty('rate', 140)
        return engine
    except Exception as e:
        print(f"Error initializing speech engine: {e}")
        return None

# Function to make the engine speak
def speak(engine, audio):
    if engine:
        print(audio)
        engine.say(audio)
        engine.runAndWait()

# Function to detect faces and predict masks
def detect_and_predict_mask(img, faceNet, maskNet):
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (250, 250), (180.0, 125.0, 130.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = img[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (250, 250))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    
    return locs, preds

# Function to initialize the camera
def initialize_camera():
    for i in range(2):  # Try two camera indices (0 and 1)
        cam = cv2.VideoCapture(i)
        if cam.isOpened():
            print(f"Using camera {i}")
            return cam
    print("No camera found.")
    return None

# Function to greet the user based on the time of day
def greet_user(engine):
    hour = int(datetime.datetime.now().hour)
    if 0 <= hour < 12:
        speak(engine, "Good Morning!")
    elif 12 <= hour < 17:
        speak(engine, "Good Afternoon!")
    elif 17 <= hour < 21:
        speak(engine, "Good Evening!")
    else:
        speak(engine, "Good Evening!")

# Function to load face encodings
def load_face_encodings(filename='FaceRec_Trained_Model.pickle'):
    try:
        data = pickle.loads(open(filename, "rb").read())
        print("Encoding data loaded successfully.")
        return data['encodings'], data['names']
    except Exception as e:
        print(f"Error loading encodings: {e}")
        return None, None

# Main face recognition and mask detection function
def rec(engine, knownEncodeList, classNames):
    cam = initialize_camera()
    if not cam:
        return

    pTime = 0
    name_ = []
    frame_count = 0
    skip_frames = 2  # Skip every other frame

    try:
        while True:
            success, img = cam.read()
            if not success:
                print("Failed to capture image from camera.")
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

            facesCurFrame = face_recognition.face_locations(imgS, model='hog')
            encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

            for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
                faceDis = face_recognition.face_distance(knownEncodeList, encodeFace)
                
                if len(faceDis) > 0:
                    matchIndex = np.argmin(faceDis)

                    if faceDis[matchIndex] < 0.6:
                        name = classNames[matchIndex].title()

                        if name_ != name:
                            speak(engine, name + " Detected")
                            print(f"Hello {name}")
                            name_ = name

                        cTime = time.time()
                        fps = 1 / (cTime - pTime)
                        pTime = cTime

                        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                                    (255, 0, 255), 3)

                        y1, x2, y2, x1 = faceLoc
                        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # Scaling back
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.6, (128, 0, 128), 1)
                    else:
                        print("No known faces detected.")
                else:
                    print("No face distance calculations available. Please Train Your Model First.")

            cv2.imshow('Face Recognition', img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("0"):  # Check if "0" is pressed
                print("Exiting...")
                break
            elif key == ord("b"):  # If 'b' is pressed, break the loop
                break

    except KeyboardInterrupt:
        print("----------------------------------------------------------")
        print("---------- KeyboardInterrupt detected. Exiting -----------")
        print("----------------------------------------------------------")

    finally:
        cam.release()
        cv2.destroyAllWindows()

# Main function
def main():
    engine = init_speech_engine()
    greet_user(engine)

    knownEncodeList, classNames = load_face_encodings()
    if knownEncodeList is not None and classNames is not None:
        rec(engine, knownEncodeList, classNames)
    else:
        print("Face encodings could not be loaded. Exiting...")

if __name__ == "__main__":
    main()
