import os
import face_recognition
import cv2



KNOWN_OFFICE_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 2
FONT_THICKNESS = 1
MODEL = "hog"

VIDEO = cv2.VideoCapture("officeclip2480.mp4")


def make_1080p():
    VIDEO.set(3, 1920)
    VIDEO.set(4, 1080)

def make_720p():
    VIDEO.set(3, 1280)
    VIDEO.set(4, 720)

def make_480p():
    VIDEO.set(3, 640)
    VIDEO.set(4, 480)

def change_res(width, height):
    VIDEO.set(3, width)
    VIDEO.set(4, height)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

print("Loading known faces...")
print("Encoding faces in database...")

known_faces = []
known_names = []

for name in os.listdir(KNOWN_OFFICE_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_OFFICE_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_OFFICE_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)


while True:
    ret, image = VIDEO.read()
    image = rescale_frame(image, percent=40)

    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        MATCH = None
        if True in results:
            MATCH = known_names[results.index(True)]
            print(f"Match found: {MATCH}")
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            color = [0, 0, 255]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, MATCH, (face_location[3]+5, face_location[2]+10),
            cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 0, 0), FONT_THICKNESS)



    cv2.imshow("", image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
