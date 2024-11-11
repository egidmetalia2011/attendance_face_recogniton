import cv2
import face_recognition
import pickle
import threading

# Global variable to store the frame
current_frame = None
face_locations = []
face_encodings = []
data = None

def process_frame():
    global current_frame, face_locations, face_encodings
    while True:
        if current_frame is None:
            continue
        # Resize the frame for processing
        small_frame = cv2.resize(current_frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

        # Proceed to face encoding only if face locations are found
        if face_locations:
            # Compute face encodings for each detected face
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Scale back the face locations to the original frame size
            face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]
        else:
            # Reset encodings if no faces are found
            face_encodings = []

# Start a thread for processing frames
processing_thread = threading.Thread(target=process_frame, daemon=True)
processing_thread.start()

def recognize_faces():
    global current_frame, face_locations, face_encodings, data
    # Load the known face encodings and names
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("No encodings found. Please run the train_faces.py script first.")
        return

    # Start the webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open camera")
        return
    print("Press 'q' to exit")

    while True:
        # Grab a single frame of the video
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to read frame.")
            continue

        # Update the global frame variable
        current_frame = frame

        # Draw rectangles and names on the detected faces
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(data["encodings"], face_encoding, tolerance=0.6)
            name = "Unknown"

            if True in matches:
                matched_indexes = [i for (i, match) in enumerate(matches) if match]
                name_counts = {}
                for index in matched_indexes:
                    matched_name = data["names"][index]
                    name_counts[matched_name] = name_counts.get(matched_name, 0) + 1

                name = max(name_counts, key=name_counts.get)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # Display the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Display the resulting frame
        cv2.imshow("Face Recognition", frame)

        # Exit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

# Run the program
recognize_faces()