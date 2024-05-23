import cv2
import imutils
import numpy as np

#　カメラの準備
cap = cv2.VideoCapture(0)

# モデルを読み込む
prototxt = 'deploy.prototxt'
model = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# カメラ画像を読み込み，顔検出して表示するループ
while True:
    ret, frame = cap.read()

    # カメラ画像を幅400pxにリサイズ
    img = imutils.resize(frame, width=800)
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 物体検出器にblobを適用する
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        # ネットワークが出力したconfidenceの値を抽出する
        confidence = detections[0, 0, i, 2]
        # confidenceの値が0.5以上の領域のみを検出結果として描画する
        if confidence > 0.5:
            # 対象領域のバウンディングボックスの座標を計算する
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # バウンディングボックスとconfidenceの値を描画する
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(img, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    #ウィンドウに画像を表示する
    cv2.imshow("face Detection", img)

    #キー入力を待ち、qが押されたらループを抜ける
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


print("OK")