import streamlit as st
import cv2
import face_recognition
import numpy as np
import pickle

# Load the saved model data
with open("face_recognition_model.pkl", "rb") as model_file:
    model_data = pickle.load(model_file)

known_face_encodings = model_data["known_face_encodings"]
known_face_names = model_data["known_face_names"]

def recognize_faces(frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)

    return face_names

def main():
    st.title("Face Recognition App")

    video_capture = cv2.VideoCapture(0)

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        face_names = recognize_faces(frame)

        st.write("Recognized Faces:")
        for name in face_names:
            st.write(name)

        st.image(frame, channels="BGR")

if __name__ == "__main__":
    main()
