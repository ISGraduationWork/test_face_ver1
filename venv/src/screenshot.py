import cv2
import os
import random
import string
from datetime import datetime

def draw_corner_rect(frame, x, y, w, h, color=(255, 0, 0), thickness=3, length=40):
    """
    矩形の四隅に小さな線を描画する関数
    :param frame: フレーム画像
    :param x: 矩形の左上隅のX座標
    :param y: 矩形の左上隅のY座標
    :param w: 矩形の幅
    :param h: 矩形の高さ
    :param color: 線の色（BGR形式）
    :param thickness: 線の太さ
    :param length: 角の線の長さ
    """
    # 左上の角
    cv2.line(frame, (x, y), (x + length, y), color, thickness)
    cv2.line(frame, (x, y), (x, y + length), color, thickness)

    # 右上の角
    cv2.line(frame, (x + w, y), (x + w - length, y), color, thickness)
    cv2.line(frame, (x + w, y), (x + w, y + length), color, thickness)

    # 左下の角
    cv2.line(frame, (x, y + h), (x, y + h - length), color, thickness)
    cv2.line(frame, (x, y + h), (x + length, y + h), color, thickness)

    # 右下の角
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)

def take_screenshot(name=None):
    """
    スクリーンショットを撮る関数
    :param name: スクリーンショットのファイル名に使用する名前
    :return: スクリーンショットのファイルパス
    """

    # カメラのキャプチャを開始
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラが開けませんでした")
        return None

    # 顔認識用のカスケード分類器を読み込み
    facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if facedetect.empty():
        print("顔認識用のカスケード分類器の読み込みに失敗しました")
        cap.release()
        cv2.destroyAllWindows()
        return None

    while True:
        # カメラからフレームを取得
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームが取得できませんでした")
            break

        # フレームをグレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 1:
            # 顔が1つだけ検出された場合、矩形の角を描画
            for (x, y, w, h) in faces:
                draw_corner_rect(frame, x, y, w, h)

            # 左上にメッセージを表示
            cv2.putText(frame, "Press Space to take a Screenshot", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        elif len(faces) > 1:
            # 複数の顔が検出された場合にテキストを表示
            cv2.putText(frame, "Multiple faces detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # フレームをウィンドウに表示
        cv2.imshow("Face Recognition", frame)

        # スペースキーが押されたらスクリーンショットを撮る
        if cv2.waitKey(1) & 0xFF == ord(' '):
            if len(faces) == 1:  # 顔が1つだけ検出された場合のみスクリーンショットを保存
                if name:
                    screenshot_filename = f"screenshot_{name}.png"
                else:
                    # タイムスタンプを利用して一意なファイル名を生成
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
                    screenshot_filename = f"screenshot_{random_name}_{timestamp}.png"
                cv2.imwrite(screenshot_filename, frame)
                cap.release()
                cv2.destroyAllWindows()
                return screenshot_filename
            else:
                print("顔が1つ検出されていません")
                break

        # 右上の×マークを押すとウィンドウを閉じる
        if cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:
            break

    # カメラのリソースを解放し、ウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()
    return None
