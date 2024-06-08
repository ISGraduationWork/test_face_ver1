import cv2
import PySimpleGUI as sg
import sqlite3
from datetime import datetime
import hashlib
import face_recognition
import pickle
import os
from screenshot import take_screenshot

def connect_database():
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces(
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    hash TEXT UNIQUE,
                    Registered_time TEXT,
                    face_encodings BLOB
                    )''')
        return conn, c
    except sqlite3.Error as e:
        sg.popup_error(f"データベースエラー: {str(e)}")
        return None, None

def close_database(conn):
    if conn:
        conn.close()

def recognize_face(image_path):
    train_img = face_recognition.load_image_file(image_path)
    train_img_location = face_recognition.face_locations(train_img, model="hog")[0]
    assert len(train_img_location) == 1, "画像から顔の検出に失敗したか、2人以上の顔が検出されました"
    (face_encoding, ) = face_recognition.face_encodings(image_path, train_img_location)[0]
    if not face_encoding:
        return None
    else:
        return face_encoding

def register_face(window, name, image_path, c, conn):
    if not name:
        window['STATUS'].update("名前が入力されていません")
        return False
    elif not image_path:
        window['STATUS'].update("画像が選択されていません")
        return False
    else:
        try:
            image = cv2.imread(image_path)
            _, img_encoded = cv2.imencode('.png', image)
            img_data = img_encoded.tobytes()

            hash_value = hashlib.sha256(img_data).hexdigest()

            c.execute("SELECT * FROM faces WHERE name=?", (name,))
            existing_name = c.fetchone()
            if existing_name:
                sg.popup_error("すでに名前が登録されています。")
                return False

            train_img = face_recognition.load_image_file(image_path)
            train_img_location = face_recognition.face_locations(train_img, model="hog")
            if not train_img_location:
                sg.popup_error("画像から顔の検出に失敗しました")
                return False

            train_img_location = train_img_location[0]
            face_encoding = face_recognition.face_encodings(train_img, [train_img_location])[0]
            face_encodings = pickle.dumps(face_encoding)

            c.execute('SELECT name, face_encodings FROM faces')
            rows = c.fetchall()
            db_encodings = []
            for row in rows:
                stored_name = row[0]
                stored_encodings = pickle.loads(row[1])
                db_encodings.append((stored_name, stored_encodings))

            dists = []
            for name_enc, encoding in db_encodings:
                distance = face_recognition.face_distance([encoding], face_encoding)[0]
                dists.append((name_enc, distance))

            matched_faces = [(name_enc, dist) for name_enc, dist in dists if dist < 0.40]

            if matched_faces:
                existing_names = ", ".join([name_enc for name_enc, _ in matched_faces])
                sg.popup_error(f"この顔はすでに登録されています。登録されている名前: {existing_names}")
                return False
            else:
                Registered_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                c.execute("INSERT INTO faces (name, hash, Registered_time, face_encodings) VALUES (?, ?, ?, ?)",
                        (name, hash_value, Registered_time, face_encodings))
                conn.commit()
                sg.popup_auto_close('登録しました', auto_close_duration=2)
                return True
        except Exception as e:
            sg.popup_error(f"エラーが発生しました：{e}")
            print(e)
            return False

def show_image_options():
    layout = [
        [sg.Button('ファイルから選択', key='FILE', size=(15, 1))],
        [sg.Button('スクリーンショット', key='SCREENSHOT', size=(15, 1))]
    ]
    event, _ = sg.Window('画像の選択方法', layout).read(close=True)
    if event == 'FILE':
        return 'file'
    elif event == 'SCREENSHOT':
        return 'screenshot'
    else:
        return None

def check_input(window, name, image_path):
    if name and image_path:
        window['REGISTER'].update(disabled=False)
    else:
        window['REGISTER'].update(disabled=True)

def browse_image(image_path=None):
    file_path = sg.popup_get_file('画像を選択', file_types=(("Image files", "*.jpg;*.jpeg;*.png"),))
    if file_path:
        return file_path
    else:
        return None

def main():
    conn, c = connect_database()

    layout = [
        [sg.Push(), sg.Text("顔情報を登録"), sg.Push()],
        [sg.Text('名前:'), sg.Input(key='NAME', enable_events=True)],
        [sg.Text('', key='STATUS', size=(34, 1)), sg.Button('画像を選択', key='BROWSE')],
        [sg.Push(), sg.Button('新規登録', key='REGISTER', disabled=True), sg.Push()],
        [sg.Text('', key='NOTIFICATION', size=(40, 2), visible=False, text_color='white', background_color='blue')]
    ]

    window = sg.Window('顔登録', layout)

    image_path = None

    while True:
        event, values = window.read(timeout=None)
        if event == sg.WIN_CLOSED:
            break
        elif event == 'BROWSE':
            option = show_image_options()
            if option == 'file':
                image_path = browse_image(image_path)
                if image_path:
                    window['STATUS'].update(image_path)
                    check_input(window, values['NAME'], image_path)
            elif option == 'screenshot':
                if values['NAME']:  # 名前が入力されている場合
                    image_path = take_screenshot(name=values['NAME'])
                else:
                    image_path = take_screenshot()
                if image_path:
                    window['STATUS'].update(image_path)
                    check_input(window, values['NAME'], image_path)
        elif event == 'REGISTER':
            name = values['NAME']
            register_face(window, name, image_path, c, conn)
            if image_path and os.path.exists(image_path):
                os.remove(image_path)
        elif event == 'NAME':
            check_input(window, values['NAME'], image_path)

    window.close()
    close_database(conn)

if __name__ == "__main__":
    main()
