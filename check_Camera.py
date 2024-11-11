import cv2

def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Unable to open camera")
        return
    print("Press 'q' to exit")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to read frame.")
            continue  # Skip this frame if failed

        # Resize the frame for display
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Display the resulting image
        cv2.imshow("Face Recognition", small_frame)

        # Exit on 'q' key press
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Run the simplified function
recognize_faces()
