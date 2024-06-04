import cv2
import imutils
import numpy as np

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
face_count = 0  # 顔が認証された回数をカウントする変数
i = 0

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

    face_detected = False  # フレームに顔が検出されたかどうかを追跡するフラグ

    for j in range(0, detections.shape[2]):
        # Extract confidence
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            # Compute bounding box coordinates
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # Draw bounding box and confidence
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            face_detected = True  # 顔が検出されたことを示すフラグをセット

# 顔が検出された場合にカウントを進める
    if face_detected:
        face_count += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)
        i += 1
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the face count on the frame
    cv2.putText(img, f"Face Count: {face_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("face Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

cap.release()
cv2.destroyAllWindows()

print("OK")
