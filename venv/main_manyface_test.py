import cv2
import imutils
import numpy as np
import time

# Haar Cascade for face detection
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Initialize camera
cap = cv2.VideoCapture(0)

# Load the model
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Initialize variables
faces_data = []
face_count = 0  # 顔が認識された回数をカウントする変数
i = 0
face_id_counter = 1  # 顔のIDの初期値を1に設定
faces_dict = {}

def detect_faces(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    # Return faces coordinates
    return faces

def draw_face_info(img, face_id, start_x, end_y, elapsed_time):
    # Draw face ID and elapsed time
    cv2.putText(img, face_id, (start_x, end_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(img, elapsed_time, (start_x, end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# Start video loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to width 800px
    img = imutils.resize(frame, width=800)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Apply the blob to the detector
    net.setInput(blob)
    detections = net.forward()

    # Prepare to track detected faces
    current_faces = []

    for j in range(0, detections.shape[2]):
        # Extract confidence
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            # Compute bounding box coordinates
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # Assign a new ID to new faces or use the existing ID
            face_center = ((startX + endX) // 2, (startY + endY) // 2)
            matched_face_id = None
            
            for face_id, face_data in faces_dict.items():
                prev_center = face_data['center']
                if abs(face_center[0] - prev_center[0]) < 50 and abs(face_center[1] - prev_center[1]) < 50:
                    matched_face_id = face_id
                    break
            
            if matched_face_id is None:
                matched_face_id = f"face:{face_id_counter}"
                face_id_counter += 1
                faces_dict[matched_face_id] = {'start_time': time.time(), 'center': face_center}
            else:
                faces_dict[matched_face_id]['center'] = face_center

            # Calculate the elapsed time
            elapsed_time = int(time.time() - faces_dict[matched_face_id]['start_time'])
            # Format elapsed time as HH:MM:SS
            elapsed_time_str = "{:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60)

            # Draw bounding box and elapsed time
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            draw_face_info(img, matched_face_id, startX, endY, elapsed_time_str)
            current_faces.append(matched_face_id)

    # Remove faces that are no longer detected
    for face_id in list(faces_dict.keys()):
        if face_id not in current_faces:
            del faces_dict[face_id]

    # Show the output image
    cv2.imshow("face Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("OK")
