import face_recognition
import os
import pickle
import sys

def encode_faces(dataset_dir="dataset"):
    try:
        # Verify dataset directory exists
        if not os.path.exists(dataset_dir):
            print(f" Error: Dataset directory '{dataset_dir}' not found.")
            return False

        # Lists to store encodings and names
        known_encodings = []
        known_names = []
        
        # Count for progress tracking
        total_processed = 0
        failed_images = 0

        # Loop through each person in the dataset directory
        for person_name in os.listdir(dataset_dir):
            person_dir = os.path.join(dataset_dir, person_name)

            # Skip files that are not directories
            if not os.path.isdir(person_dir):
                continue

            print(f"Processing images for {person_name}...")

            # Loop through each image for the person
            for image_name in os.listdir(person_dir):
                if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue

                image_path = os.path.join(person_dir, image_name)
                try:
                    # Load the image and convert it to RGB
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    # If a face was found, save the encoding and the person's name
                    if len(encodings) > 0:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)
                        total_processed += 1
                        print(f" Successfully processed {image_name}")
                    else:
                        failed_images += 1
                        print(f"  No face found in {image_name}")
                except Exception as e:
                    failed_images += 1
                    print(f" Error processing {image_name}: {str(e)}")

        if total_processed == 0:
            print(" Error: No faces were successfully encoded.")
            return False

        # Save the encodings and names to a file
        data = {"encodings": known_encodings, "names": known_names}
        with open("encodings.pickle", "wb") as f:
            pickle.dump(data, f)

        print(f"\n✅ Encoding complete:")
        print(f"   - Successfully processed: {total_processed} images")
        print(f"   - Failed to process: {failed_images} images")
        print(f"   - Data saved to encodings.pickle")
        print(f"   - Encoded {len(set(known_names))} unique persons")
        return True

    except Exception as e:
        print(f"❌ An unexpected error occurred: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the function to encode faces
    success = encode_faces()
    if not success:
        sys.exit(1)
