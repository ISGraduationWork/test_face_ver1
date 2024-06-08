import sqlite3
import numpy as np

# データベースに接続
conn = sqlite3.connect('face_recognition.db')
c = conn.cursor()

# データベースからencodingの内容を取得
c.execute('SELECT encoding FROM faces')
rows = c.fetchall()

# 取得したバイナリデータをNumPy配列に変換
encodings = [np.frombuffer(row[0], dtype=np.float64) for row in rows]

# 取得した特徴量を表示
for encoding in encodings:
    print(encoding)

print(encodings)

# データベース接続を閉じる
conn.close()
