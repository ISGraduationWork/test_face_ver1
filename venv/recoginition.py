import cv2
import PySimpleGUI as sg
import sqlite3
from datetime import datetime
import hashlib
import face_recognition
from screenshot import take_screenshot

def connect_database():
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        # 顔のデータベーステーブルの作成または接続
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                    (id INTEGER PRIMARY KEY, name TEXT, hash TEXT UNIQUE, Registered_time TEXT, face_encodings TEXT)''')
        return conn, c
    except sqlite3.Error as e:
        sg.popup_error(f"データベースエラー: {str(e)}")
        return None, None

def close_database(conn):
    if conn:
        conn.close()

def recognize_face(image_path):
    """
    入力された画像から顔認識をして顔認証情報を取り出す関数
    """
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        return None
    else:
        return face_encodings[0].tolist()

def register_face(window, name, image_path, c, conn):
    if not name:
        window['STATUS'].update("名前が入力されていません")
        return
    elif not image_path:
        window['STATUS'].update("画像が選択されていません")
        return
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
                return

            c.execute("SELECT * FROM faces WHERE hash=?", (hash_value,))
            existing_face = c.fetchone()
            if existing_face:
                sg.popup_error("この顔はすでに登録されています。")
                return

            face_encodings = recognize_face(image_path)
            if not face_encodings:
                sg.popup_error("顔が画像中に検出されませんでした。")
                return

            # 顔認証情報が同じであればすでに顔が登録されていると判断
            c.execute("SELECT * FROM faces")
            existing_faces = c.fetchall()
            for face in existing_faces:
                if face_encodings == eval(face[4]):  # リスト同士の比較
                    sg.popup_error("すでに顔が登録されています。")
                    return

            Registered_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            c.execute("INSERT INTO faces (name, hash, Registered_time, face_encodings) VALUES (?, ?, ?, ?)",
                        (name, hash_value, Registered_time, str(face_encodings)))
            conn.commit()

            sg.popup_auto_close('登録しました', auto_close_duration=2)

            window['NAME'].update('')
            window['STATUS'].update('')
            window['REGISTER'].update(disabled=True)

        except sqlite3.IntegrityError:
            sg.popup_error("この顔はすでに登録されています。")
        except Exception as e:
            sg.popup_error(f"登録に失敗しました: {str(e)}")



def show_image_options():
    layout = [
        [
            sg.Button('ファイルから選択', key='FILE', size=(15, 1))
            ],
        [
            sg.Button('スクリーンショット', key='SCREENSHOT', size=(15, 1))
            ]
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
        [
            sg.Push(),
            sg.Text("顔情報を登録"),
            sg.Push()
            ],
        [
            sg.Text('名前:'),
            sg.Input(
                key='NAME',
                enable_events=True)
            ],
        [
            sg.Text('', key='STATUS', size=(34, 1)),
            sg.Button('画像を選択', key='BROWSE')
            ],
        [
            sg.Push(),
            sg.Button(
                    '新規登録',
                    key='REGISTER',
                    disabled=True),
            sg.Push()],
        [
            sg.Text(
                    '',
                    key='NOTIFICATION',
                    size=(40, 2), visible=False,
                    text_color='white',
                    background_color='blue')
            ]
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
