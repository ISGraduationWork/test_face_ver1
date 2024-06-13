import cv2
import imutils
import numpy as np
import time
import pickle
import face_recognition
from screenshot import draw_corner_rect
from database import connect_database, close_database
import keyboard  # 追加

# Haar Cascadeによる顔検出のためのモデルを読み込む
path = "haarcascade_frontalface_default.xml"
facedetect = cv2.CascadeClassifier(path)

# カメラを読み込む
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

# Caffeモデルを読み込む
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# データベースに接続して顔データを取得
conn, c = connect_database()
c.execute('SELECT name, face_encodings FROM faces')
rows = c.fetchall()
known_face_encodings = []
known_face_names = []
for row in rows:
    stored_name = row[0]
    stored_encodings = pickle.loads(row[1])
    known_face_encodings.append(stored_encodings)
    known_face_names.append(stored_name)

# 変数の初期化
faces_dict = {}

# 画像に経過時間、名前を描画する関数
def draw_face_info(img, start_x, end_y, elapsed_time, name):
    """
    画像に経過時間、名前を描画する関数
    :param img: 入力画像
    :param start_x: 顔の左上のx座標
    :param end_y: 顔の右下のy座標
    :param elapsed_time: 経過時間（HH:MM:SS形式の文字列）
    :param name: 顔の名前
    """
    # 経過時間、名前を描画
    cv2.putText(img, elapsed_time, (start_x, end_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(img, name, (start_x, end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# ビデオループを開始
while True:
    # カメラからフレームを取得
    ret, frame = cap.read()
    if not ret:
        # カメラからフレームを取得できない場合は終了
        print("カメラからフレームを取得できませんでした。")
        break

    # フレームを幅900pxにリサイズ
    img = imutils.resize(frame, width=900)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # モデルに入力としてblobを設定
    net.setInput(blob)
    detections = net.forward()

    # 検出された顔を追跡する準備
    current_faces = []

    # 各検出結果について処理
    for j in range(0, detections.shape[2]):
        # 信頼度を抽出
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            # バウンディングボックスの座標を計算
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 顔の特徴量のエンコーディングを取得
            face_img = img[startY:endY, startX:endX]
            rgb_face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(rgb_face_img)

            name = "Unknown"
            elapsed_time_str = "00:00:00"
            if face_encodings:
                face_encoding = face_encodings[0]
                # データベースの顔の特徴量と比較
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    if name not in faces_dict:
                        faces_dict[name] = time.time()

                    # 経過時間
                    elapsed_time = int(time.time() - faces_dict[name])
                    elapsed_time_str = "{:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60)

            # バウンディングボックスと経過時間、名前を描画
            draw_corner_rect(img, startX, startY, endX - startX, endY - startY)
            draw_face_info(img, startX, endY, elapsed_time_str, name)
            current_faces.append(name)

    # 検出されなくなった顔をリストから削除
    for name in list(faces_dict.keys()):
        if name not in current_faces:
            del faces_dict[name]

    # 結果画像を表示
    cv2.imshow("Face Detection", img)

    # 'q'キーが押されたらループを終了
    if keyboard.is_pressed('q'):
        break

# カメラを解放し、ウィンドウとデータベースを閉じる
cap.release()
close_database(conn)
cv2.destroyAllWindows()

print("OK")
