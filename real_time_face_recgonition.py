import cv2
import face_recognition
import pickle

def recognize_faces():
    #load the known face encoding and names
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("No encodings found. Please run the train_faces.py script first.")
        return
    
    #start the webcam
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open camera")
        return
    print("press 'q' to exit")

    #frame processing frequency (process every nth frame)
    frame_process_interval = 10
    frame_count = 0

    while True:
        #grab a sinle frame of the video
        ret, frame = video_capture.read()
        if not ret:
            print("failed to capture video frame")
            break

        #check if frame empty:
        if frame is None:
            print("frame is empty")
            continue

        #reszie the frame
        small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)

        #convert the frame from bgr  to rgb fro face_recgonition
        rgb_small_frame = small_frame[:, :, ::-1]

        #only process every nth frame
        if frame_count % frame_process_interval == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")

            try:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            except Exception as e:
                print(f"Error calculating face encondngs: {e}")
                continue
            
            if len(face_encodings) == 0:
                print("No face encodings found, skipping frame")
                #continue

            #convert face locations back to the original frame size
            face_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]
            
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
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        #DISPLAY THE RESULTING IMAGE
        cv2.imshow("Face recogniton", frame)
        cv2.waitKey(1)

        #incremet the frame count
        frame_count += 1

        #exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    #release the capture and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

#run the program
recognize_faces()