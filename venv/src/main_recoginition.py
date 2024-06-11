import cv2
import PySimpleGUI as sg
import sqlite3
from datetime import datetime
import hashlib
import face_recognition
import pickle
import os
from database import connect_database, close_database
from screenshot import take_screenshot
from gui import show_image_options, check_input, browse_image

# 顔認識を行う関数
def recognize_face(image_path):
    train_img = face_recognition.load_image_file(image_path)
    train_img_locations = face_recognition.face_locations(train_img, model="hog")

    if len(train_img_locations) != 1:
        raise ValueError("画像から顔の検出に失敗したか、2人以上の顔が検出されました")

    else:
        face_encodings = face_recognition.face_encodings(train_img, train_img_locations)[0]

    if not face_encodings.all():
            return None
    else:
        return face_encodings



# 顔情報をデータベースに登録する関数
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
                window['NAME'].update('')
                return False

            face_encoding = recognize_face(image_path)
            if not face_encoding.all():
                sg.popup_error("顔のエンコーディングに失敗しました。")
                Delete_image(window, image_path)
                return False

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
                Delete_image(window, image_path)
                return False
            else:
                Registered_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                c.execute("INSERT INTO faces (name, hash, Registered_time, face_encodings) VALUES (?, ?, ?, ?)",
                        (name, hash_value, Registered_time, face_encodings))
                conn.commit()
                sg.popup_auto_close('登録しました', auto_close_duration=2)
                Delete_image(window, image_path)

                # 登録後フィールドをリセット
                window['NAME'].update('')
                window['REGISTER'].update(disabled=True)

                return True
        except Exception as e:
            sg.popup_error(f"エラーが発生しました：{e}")
            print(e)
            return False

# 写真や画像を削除する関数
def Delete_image(window, image_path):
    # フォルダから削除
    if image_path and os.path.exists(image_path):
        os.remove(image_path)
    # インターフェースから削除
    window['STATUS'].update('')

# メイン関数
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
        elif event == 'NAME':
            check_input(window, values['NAME'], image_path)

    window.close()
    close_database(conn)

if __name__ == "__main__":
    main()
