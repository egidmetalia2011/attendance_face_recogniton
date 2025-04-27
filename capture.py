import cv2
import os

def capture_images(name, num_images=5):
    # Create a folder for the images
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    # Start the webcam
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit and 'c' to capture")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture")
            break

        # Show the captured image
        cv2.imshow('Capture Image', frame)

        # Wait for the user to press a key
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Save the image
            image_path = os.path.join(person_dir, f"{name}_{count+1}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"Image {count+1} saved at {image_path}")
            count += 1

    # Release the webcam and close any open windows
    cap.release()
    cv2.destroyAllWindows()

# Example usage
name = input("Enter the name of the person: ")
capture_images(name)