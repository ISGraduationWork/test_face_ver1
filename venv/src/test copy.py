import face_recognition
import matplotlib.pyplot as plt
import os
import sqlite3
import numpy as np
import cv2
import PySimpleGUI as sg
from datetime import datetime
import hashlib
from screenshot import take_screenshot


# データベース接続を確立
conn = sqlite3.connect('face_recognition.db')
c = conn.cursor()

# 顔のデータベーステーブルの作成または接続
c.execute('''
    CREATE TABLE IF NOT EXISTS faces (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        encoding BLOB NOT NULL
    )
''')
conn.commit()

# # 学習させたい（登録したい）顔画像のファイル名をリストに格納
# train_img_names = ["C:\\Users\\noraneko\\.vscode\\test_face_ver1\\venv\\screenshot_neko.png","C:\\Users\\noraneko\\.vscode\\test_face_ver1\\venv\\obama.jpg" ]

# 学習データの顔画像を読み込む
for name in train_img_names:
    train_img = face_recognition.load_image_file(name)
    train_img_location = face_recognition.face_locations(train_img, model="hog")[0]
    encoding = face_recognition.face_encodings(train_img, [train_img_location])[0]
    c.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', (os.path.basename(name), sqlite3.Binary(encoding)))
conn.commit()

# テストデータ（認証する人の顔画像）を読み込む
# test_img_name = "C:\\Users\\noraneko\\.vscode\\test_face_ver1\\venv\\screenshot_nekoneko_test.png"
test_img_name = "C:\\Users\\noraneko\\.vscode\\test_face_ver1\\venv\\obama.jpg"
test_img = face_recognition.load_image_file(test_img_name)
test_img_location = face_recognition.face_locations(test_img, model="hog")[0]
test_img_encoding = face_recognition.face_encodings(test_img, [test_img_location])[0]

# データベースから学習データの名前と特徴量を取得
c.execute('SELECT name, encoding FROM faces')
rows = c.fetchall()
db_encodings = [(row[0], np.frombuffer(row[1], dtype=np.float64)) for row in rows]

# 学習データとテストデータの特徴量を比較し、ユークリッド距離を取得する
dists = []
for name, encoding in db_encodings:
    distance = face_recognition.face_distance([np.frombuffer(encoding, dtype=np.float64)], test_img_encoding)[0]
    dists.append((name, distance))

# 顔が一致するか判定
matched_faces = [(name, dist) for name, dist in dists if dist < 0.40]

if matched_faces:
    print("一致した顔:")
    for name, _ in matched_faces:
        print(name)
else:
    print("一致する顔が見つかりませんでした。")

# データベース接続を閉じる
conn.close()
