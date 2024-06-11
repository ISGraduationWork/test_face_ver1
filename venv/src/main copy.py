import cv2
import imutils
import numpy as np
import time
from database import connect_database, close_database
from screenshot import draw_corner_rect

# Haar Cascadeによる顔検出のためのモデルを読み込む
facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# カメラを読み込む
cap = cv2.VideoCapture(0)

# Caffeモデルを読み込む
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# 変数の初期化
faces_data = []
face_count = 0  # 顔が認識された回数をカウントする変数
i = 0
face_id_counter = 1  # 顔のIDの初期値を1に設定
faces_dict = {}

# データベースに接続
conn, c = connect_database()

# 画像から顔を検出する関数
def detect_faces(image):
    """
    画像から顔を検出する関数
    :param image: 入力画像
    :return: 検出された顔の座標のリスト
    """
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 画像中の顔を検出
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # 検出された顔の座標を返す
    return faces

# 画像に顔IDと経過時間を描画する関数
def draw_face_info(img, face_id, start_x, end_y, elapsed_time):
    """
    画像に顔IDと経過時間を描画する関数
    :param img: 入力画像
    :param face_id: 顔ID
    :param start_x: 顔の左上のx座標
    :param end_y: 顔の右下のy座標
    :param elapsed_time: 経過時間（HH:MM:SS形式の文字列）
    """
    # 顔IDと経過時間を描画
    cv2.putText(img, face_id, (start_x, end_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
    cv2.putText(img, elapsed_time, (start_x, end_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# ビデオループを開始
while True:
    # カメラからフレームを取得
    ret, frame = cap.read()
    if not ret:
        # カメラからフレームを取得できない場合は終了
        print("カメラからフレームを取得できませんでした。")
        break

    # フレームを幅800pxにリサイズ
    img = imutils.resize(frame, width=800)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # モデルに入力としてblobを設定
    net.setInput(blob)
    detections = net.forward()

    # 検出された顔を追跡する準備
    current_faces = []

    # 画像から顔を検出
    faces = detect_faces(img)

    # 各検出結果について処理
    for j in range(0, detections.shape[2]):
        # 信頼度を抽出
        confidence = detections[0, 0, j, 2]
        if confidence > 0.5:
            # バウンディングボックスの座標を計算
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 新しい顔にIDを割り当てるか既存のIDを使用
            face_center = ((startX + endX) // 2, (startY + endY) // 2)
            matched_face_id = None

            # 既存の顔IDと一致するかチェック
            for face_id, face_data in faces_dict.items():
                prev_center = face_data['center']
                if abs(face_center[0] - prev_center[0]) < 50 and abs(face_center[1] - prev_center[1]) < 50:
                    matched_face_id = face_id
                    break

            # 新しい顔の場合は新しいIDを割り当て
            if matched_face_id is None:
                matched_face_id = f"face:{face_id_counter}"
                face_id_counter += 1
                faces_dict[matched_face_id] = {'start_time': time.time(), 'center': face_center}
            else:
                # 既存の顔の場合は中心座標を更新
                faces_dict[matched_face_id]['center'] = face_center

            # 経過時間を計算
            elapsed_time = int(time.time() - faces_dict[matched_face_id]['start_time'])
            # 経過時間をHH:MM:SS形式でフォーマット
            elapsed_time_str = "{:02d}:{:02d}:{:02d}".format(elapsed_time // 3600, (elapsed_time % 3600 // 60), elapsed_time % 60)

            # バウンディングボックスと経過時間を描画
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            draw_face_info(img, matched_face_id, startX, endY, elapsed_time_str)
            current_faces.append(matched_face_id)

    # 検出されなくなった顔をリストから削除
    for face_id in list(faces_dict.keys()):
        if face_id not in current_faces:
            del faces_dict[face_id]

    # 結果画像を表示
    cv2.imshow("Face Detection", img)

    # 'q'キーが押されたらループを終了
    if cv2.waitKey(1) == ord('q'):
        break

# カメラを解放し、ウィンドウとデータベースを閉じる
cap.release()
close_database(conn)
cv2.destroyAllWindows()

print("OK")