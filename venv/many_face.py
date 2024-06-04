import face_recognition
import cv2
import numpy as np
import imutils

# Load the image from the zoom meeting
image_path = "face_many.jpg"
zoom_image = face_recognition.load_image_file(image_path)

# Load the sample pictures and learn how to recognize them
def load_face_encoding(file_path):
    image = face_recognition.load_image_file(file_path)
    encodings = face_recognition.face_encodings(image)
    if len(encodings) > 0:
        return encodings[0]
    else:
        print(f"No faces found in {file_path}")
        return None

Bob_face_encoding = load_face_encoding("train\many_face_Bob.jpg")
body_face_encoding = load_face_encoding("train\many_face_body.jpg")
jone_face_encoding = load_face_encoding("train\many_face_jone.jpg")

# Create arrays of known face encodings and their names
known_face_encodings = []
known_face_names = []

if Bob_face_encoding is not None:
    known_face_encodings.append(Bob_face_encoding)
    known_face_names.append("Bob")

if body_face_encoding is not None:
    known_face_encodings.append(body_face_encoding)
    known_face_names.append("Body")

if jone_face_encoding is not None:
    known_face_encodings.append(jone_face_encoding)
    known_face_names.append("Jone")

# Find all the faces and face encodings in the zoom image
face_locations = face_recognition.face_locations(zoom_image)
face_encodings = face_recognition.face_encodings(zoom_image, face_locations)

# Initialize an array for the names of detected faces
face_names = []

# Loop through each face found in the zoom image
for face_encoding in face_encodings:
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    name = "Unknown"

    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index]

    face_names.append(name)

# Display the results
for (top, right, bottom, left), name in zip(face_locations, face_names):
    # Draw a box around the face with a red rectangle
    cv2.rectangle(zoom_image, (left, top), (right, bottom), (0, 0, 255), 2)

    # Draw a label with a name below the face
    cv2.rectangle(zoom_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(zoom_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Convert the image from RGB color (which face_recognition uses) to BGR color (which OpenCV uses)
bgr_image = cv2.cvtColor(zoom_image, cv2.COLOR_RGB2BGR)

# Save the resulting image to a file
output_image_path = "output_face_recognition.jpg"
cv2.imwrite(output_image_path, bgr_image)

# Display the resulting image
cv2.imshow('Zoom Image', bgr_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
