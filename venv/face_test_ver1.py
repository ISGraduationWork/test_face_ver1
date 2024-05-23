import dlib
import cv2
import numpy as np
import csv
import tkinter as tk
import datetime

# グローバル変数
detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('/Users/GakutoSasabe/Desktop/Research/OpencvEyetracking/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor('/Users/sy200/Graduation_work/test_face_ver1/venv/shape_predictor_68_face_landmarks.dat')
pupil_locate_list = [['date', 'time', 'right_eye_x', 'right_eye_y', 'left_eye_x', 'left_eye_y']]

def is_close(y0, y1):
    """目が閉じているか判定する関数"""
    return abs(y0 - y1) < 10

def get_center(gray_img):
    """二値化された目画像から瞳の重心を求める"""
    moments = cv2.moments(gray_img, False)
    try:
        return int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
    except ZeroDivisionError:
        print("Division by zero in moments calculation")
        return None

def get_eye_parts(parts, left=True):
    """目部分の座標を求める"""
    if left:
        return [parts[i] for i in [36, 37, 38, 40, 41, 39]]
    else:
        return [parts[i] for i in [42, 43, 44, 46, 47, 45]]

def get_eye_image(img, parts, left=True):
    """カメラ画像と見つけた顔の座標から目の画像を求めて表示する"""
    eyes = get_eye_parts(parts, left)
    org_x, org_y = eyes[0].x, eyes[1].y
    if is_close(org_y, eyes[3].y):
        return None
    eye = img[org_y:eyes[3].y, org_x:eyes[-1].x]
    height, width = eye.shape[:2]
    resize_eye = cv2.resize(eye, (int(width * 5.0), int(height * 5.0)))
    window_name = "left" if left else "right"
    cv2.imshow(window_name, resize_eye)
    cv2.moveWindow(window_name, 50 if left else 350, 200)
    return eye

def get_eye_center(img, parts, left=True):
    """Partsから目のセンター位置を求めて、表示する"""
    eyes = get_eye_parts(parts, left)
    x_center = int(eyes[0].x + (eyes[-1].x - eyes[0].x) / 2)
    y_center = int(eyes[1].y + (eyes[3].y - eyes[1].y) / 2)
    cv2.circle(img, (x_center, y_center), 3, (255, 255, 0), -1)
    return x_center, y_center

def get_pupil_location(img, parts, left=True):
    """Partsから瞳の位置を求めて表示する、その過程で目の二値化画像を表示"""
    eyes = get_eye_parts(parts, left)
    org_x, org_y = eyes[0].x, eyes[1].y
    if is_close(org_y, eyes[3].y):
        return None
    eye = img[org_y:eyes[3].y, org_x:eyes[-1].x]
    _, threshold_eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY), 45, 255, cv2.THRESH_BINARY_INV)
    height, width = threshold_eye.shape
    resize_eye = cv2.resize(threshold_eye, (int(width * 5.0), int(height * 5.0)))
    window_name = "left_threshold" if left else "right_threshold"
    cv2.imshow(window_name, resize_eye)
    cv2.moveWindow(window_name, 50 if left else 350, 300)
    center = get_center(threshold_eye)
    if center:
        cv2.circle(img, (center[0] + org_x, center[1] + org_y), 3, (255, 0, 0), -1)
        return center[0] + org_x, center[1] + org_y
    return center

def calculate_relative_pupil_position(img, eye_center, pupil_locate, left=True):
    """目の中心座標と瞳の座標から目の中央に対しての瞳の相対座標を求める"""
    if not eye_center or not pupil_locate:
        return
    relative_pupil_x = pupil_locate[0] - eye_center[0]
    relative_pupil_y = pupil_locate[1] - eye_center[1]
    position = "left" if left else "right"
    cv2.putText(img,
                f"{position} x={relative_pupil_x} y={relative_pupil_y}",
                org=(50, 400 if left else 450),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(0, 255, 0),
                thickness=2,
                lineType=cv2.LINE_4)
    return relative_pupil_x, relative_pupil_y

def calculate_direction(img, parts, pupil_locate):
    """瞳の位置と目の座標から瞳が向いている方向を求めて表示する"""
    if not pupil_locate:
        return
    eyes = get_eye_parts(parts, True)
    left_border = eyes[0].x + (eyes[-1].x - eyes[0].x) / 3
    right_border = eyes[0].x + (eyes[-1].x - eyes[0].x) * 2 / 3
    up_border = eyes[1].y + (eyes[3].y - eyes[1].y) / 3
    down_border = eyes[1].y + (eyes[3].y - eyes[1].y) * 2 / 3

    if pupil_locate[0] < left_border:
        show_text(img, "LEFT", 50, 50)
    elif pupil_locate[0] > right_border:
        show_text(img, "RIGHT", 50, 50)
    else:
        show_text(img, "STRAIGHT", 50, 50)

    if pupil_locate[1] < up_border:
        show_text(img, "UP", 50, 100)
    elif pupil_locate[1] > down_border:
        show_text(img, "DOWN", 50, 100)
    else:
        show_text(img, "MIDDLE", 50, 100)

def show_text(img, text, x, y):
    """画像上にテキストを表示する"""
    cv2.putText(img, text, org=(x, y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_4)

def gui_test():
    """Tkinter GUIテスト"""
    root = tk.Tk()
    Static1 = tk.Label(text='test')
    Static1.pack()
    root.mainloop()

def write_csv(data):
    """データをCSVファイルに書き込む"""
    if not data:
        return
    with open('pupil_locate.csv', 'w', newline='') as f_object:
        writer_object = csv.writer(f_object)
        writer_object.writerows(data)
        print("pupil_locate.csvに出力完了")

def append_pupil_locate_to_list(left_pupil_position, right_pupil_position):
    """現在時刻、右瞳位置、左瞳位置をリストに追加する"""
    if not left_pupil_position or not right_pupil_position:
        return
    for_write_time = datetime.datetime.now()
    locate = [datetime.date.today(), f"{for_write_time.hour}:{for_write_time.minute}:{for_write_time.second}",
            left_pupil_position[0], left_pupil_position[1], right_pupil_position[0], right_pupil_position[1]]
    pupil_locate_list.append(locate)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    dets = detector(frame[:, :, ::-1])
    if len(dets) > 0:
        parts = predictor(frame, dets[0]).parts()
        left_eye_image = get_eye_image(frame, parts, True)
        right_eye_image = get_eye_image(frame, parts, False)
        left_eye_center = get_eye_center(frame, parts, True)
        right_eye_center = get_eye_center(frame, parts, False)
        left_pupil_location = get_pupil_location(frame, parts, True)
        right_pupil_location = get_pupil_location(frame, parts, False)
        left_relative_pupil_position = calculate_relative_pupil_position(frame, left_eye_center, left_pupil_location, True)
        right_relative_pupil_position = calculate_relative_pupil_position(frame, right_eye_center, right_pupil_location, False)
        calculate_direction(frame, parts, left_pupil_location)
        append_pupil_locate_to_list(left_relative_pupil_position, right_relative_pupil_position)
        cv2.imshow("me", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
