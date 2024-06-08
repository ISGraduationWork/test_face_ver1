import cv2
import imutils
import numpy as np
import time
import sqlite3

# Haar Cascadeによる顔検出のためのモデルを読み込む
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# カメラを読み込む
cap = cv2.VideoCapture(0)

# Caffeモデルを読み込む
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# 顔のデータベースに接続
conn = sqlite3.connect('faces.db')
c = conn.cursor()

# 画像から顔を検出する関数
def detect_faces(image):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 画像中の顔を検出
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    # 検出された顔の座標を返す
    return faces

# 画像に顔IDと経過時間を描画する関数
def draw_face_info(img, face_id, start_x, end_y, elapsed_time, face_name):
    # 顔IDと経過時間を描画
    cv2.putText(img, f"ID: {face_id}", (start_x, end_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(img, f"Name: {face_name}", (start_x, end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(img, f"Time: {elapsed_time}", (start_x, end_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# ビデオループを開始
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # フレームを幅800pxにリサイズ
    img = imutils.resize(frame, width=800)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # モデルに入力としてblobを設定
    net.setInput(blob)
    detections = net.forward()

    for j in range(0, detections.shape[2]):
        # 信頼度を抽出
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            # バウンディングボックスの座標を計算
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 経過時間を計算
            elapsed_time = int(time.time() - start_time)
            # 経過時間をHH:MM:SS形式でフォーマット
            elapsed_time_str = "{:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60)

            # 顔情報をデータベースから取得
            c.execute("SELECT name FROM faces WHERE id=?", (face_id,))
            result = c.fetchone()
            face_name = result[0] if result else "Unknown"

            # バウンディングボックスと顔情報を描画
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            draw_face_info(img, face_id, startX, endY, elapsed_time_str, face_name)
    # 検出されなくなった顔をリストから削除
    for face_id in list(faces_dict.keys()):
        if face_id not in current_faces:
            del faces_dict[face_id]

    # 結果画像を表示
    cv2.imshow("Face Detection", img)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) == ord('q'):
        break

# カメラを解放し、ウィンドウを閉じる
cap.release()
cv2.destroyAllWindows()

print("OK")
