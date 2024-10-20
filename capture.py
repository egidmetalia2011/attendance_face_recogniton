import cv2
import os

def capture_iamges(name, num_images=5):
    # Create a folder for the images
    dataset_dir = "dataset"
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    #start the webcam
    cap = cv2.VideoCapture(0)
    print("pres 'q' to quit and 'c' to capture")

    count = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("failed to capture")
            break

        #show the captured image
        cv2.imshow('Capture imAGE', frame)

        #Wait for the use to  press a key
        key = cv2.waitKey(1) & 0xFF
        if  key == ord('q'):
            break
        elif key == ord('c'):
            #save the image
            img_path = os.path.join(person_dir, f"{name}_{count+1}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"image {count+1} image saved at {img_path}")
            count +=1

    #release the webcame and close any open window
    cap.release()
    cv2.destroyAllWindows()

#Example usage
name = input("enter the name of the person")
capture_iamges(name)