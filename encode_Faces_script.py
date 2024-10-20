import face_recognition
import os
import pickle

def encode_faces(dataset_dir = "dataset"):
    #create a list to store the encodings and names
    known_encodings = []
    known_names = []

    #loop through each person in the dataset directory
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        #skip files that aare not directories
        if not os.path.isdir(person_dir):
            continue

        #loop through each image for the person
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            #load the image and convert it from opencv forma to rgb
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            #if a face was foun, save the encoding and the person's name
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)
    
    #save the encoding and the names to a file
    data = {"encodings ": known_encodings, "names ": known_names}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)
    
    print("Encoding complete. Data saved to encoding.pickle.")

#run the function to encode the faces
encode_faces()
