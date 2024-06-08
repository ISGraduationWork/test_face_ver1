import cv2
import os
import random
import string

def take_screenshot(name=None):
    """
    スクリーンショットを撮る関数
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームが取得できませんでした")
            return None

        # 顔認識
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        # 顔が1つだけ検出された場合のみ矩形を描画
        if len(faces) == 1:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # 左上にメッセージを表示
            cv2.putText(frame, "Press Space to take a Screenshot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)

        # スペースキーを押すとスクリーンショットを撮る
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if len(faces) == 1:  # 顔が1つだけ検出された場合のみスクリーンショットを保存
                if name:
                    screenshot_filename = f"screenshot_{name}.png"
                else:
                    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    screenshot_filename = f"screenshot_{random_name}.png"
                cv2.imwrite(screenshot_filename, frame)
                cap.release()
                cv2.destroyAllWindows()
                return screenshot_filename
            else:
                print("顔が1つ検出されていません")
                break

        # 右上の×マークを押すとウィンドウを閉じる
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    cv2.destroyAllWindows()
    return None
