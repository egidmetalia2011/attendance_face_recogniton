import cv2
import face_recognition
import pickle

def recgonize_faces():
    #load the known face encoding and names
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
    
    #start the webcam
    video_capture = cv2.VideoCapture(0)
    print("press 'q' to exit")

    while True:
        #grab a sinle frame of the video
        ret, frame = video_capture.read()
        if not ret:
            print("failed to capture video frame")
            break

        #convert the framfrom bgr  to rgb fro face_recgonition
        rgb_frame = frame[:, :, ::-1]

        #find all the face locations and face encoding in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        try:
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        except Exception as e:
            #if an error occure print it and skip the fram
            print(f"error: {e}")
            continue
        
        #loop over each detected face
        for (top, right, bottom, left), face_encodings in zip(face_locations, face_encodings):
            #check if the face matches with any known face encodings
            matches = face_recognition.compare_faces(data["encodings"],  face_encodings, tolerance=0.6)
            name = "Unknown"

            #if a match found. Use the name of the first match
            if True in matches:
                matched_indexes = [i for (i, match) in enumerate(matches) if match]
                name_counts = {}
                for index in matched_indexes:
                    matched_name = data["names"][index]
                    name_counts[matched_name] = name_counts.get(matched_name, 0)
            
            name =  max(name_counts, key=name_counts.get)
            
            #draw a rectangle aroudn the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            #display the name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPELX, 0.6, (255, 255, 255), 1)

        #DISPLAY THE RESULTING IMAGE
        cv2.imshow("Face recogniton", frame)

        #exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #release the capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

#run the program
recgonize_faces()