import cv2
import tkinter as tk
from tkinter import filedialog
import sqlite3
import os

# データベースの接続
db_path = 'faces.db'
conn = sqlite3.connect(db_path)
c = conn.cursor()

# 顔のデータベーステーブルの作成
c.execute('''CREATE TABLE IF NOT EXISTS faces
            (id INTEGER PRIMARY KEY, name TEXT, face_data BLOB)''')

# 顔を新規登録する関数
def register_face():
    name = name_entry.get()
    if not name:
        status_label.config(text="Error: 名前を入力してください")
        return

    if not image_path:
        status_label.config(text="Error: 画像を選択してください")
        return

    with open(image_path, 'rb') as f:
        face_data = f.read()

    c.execute("INSERT INTO faces (name, face_data) VALUES (?, ?)", (name, sqlite3.Binary(face_data)))
    conn.commit()
    status_label.config(text="Face registered successfully.")

# 画像を参照して選択する関数
def browse_image():
    global image_path
    image_path = filedialog.askopenfilename()
    if image_path:
        status_label.config(text=f"Selected Image: {os.path.basename(image_path)}")

# スクショを取得して登録する関数
def take_screenshot():
    # カメラのキャプチャを開始
    cap = cv2.VideoCapture(0)
    
    # カメラからフレームを取得
    ret, frame = cap.read()
    
    # 顔認識器の読み込み
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # 顔が検出された場合、スクリーンショットを保存
    if len(faces) > 0:
        # 最初の顔の領域を取得
        (x, y, w, h) = faces[0]
        
        # 顔の領域を切り取り
        face_img = frame[y:y+h, x:x+w]
        
        # 顔のスクリーンショットを保存
        cv2.imwrite("face_screenshot.png", face_img)
        status_label.config(text="Face screenshot saved successfully.")
    else:
        status_label.config(text="No face detected.")

    # カメラの解放
    cap.release()

# UIの作成
root = tk.Tk()
root.title("Face Registration")

name_label = tk.Label(root, text="名前:")
name_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

name_entry = tk.Entry(root)
name_entry.grid(row=0, column=1, padx=5, pady=5)

image_button = tk.Button(root, text="画像を選択", command=browse_image)
image_button.grid(row=1, column=0, columnspan=2, padx=5, pady=5)

register_button = tk.Button(root, text="顔を新規登録する", command=register_face)
register_button.grid(row=2, column=0, columnspan=2, padx=5, pady=5)

screenshot_button = tk.Button(root, text="スクショを取得して登録する", command=take_screenshot)
screenshot_button.grid(row=3, column=0, columnspan=2, padx=5, pady=5)

status_label = tk.Label(root, text="")
status_label.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

root.mainloop()

# データベースの切断
conn.close()
