import face_recognition
import cv2
import hashlib
from datetime import datetime
import PySimpleGUI as sg

def recognize_face(image_path):
    image = face_recognition.load_image_file(image_path)
    face_encodings = face_recognition.face_encodings(image)

    if not face_encodings:
        return None
    else:
        return face_encodings[0].tolist()

def authenticate_face(image_path, c):
    try:
        face_encodings = recognize_face(image_path)
        if not face_encodings:
            sg.popup_error("顔が画像中に検出されませんでした。")
            return

        c.execute("SELECT * FROM faces")
        registered_faces = c.fetchall()

        max_similarity = 0
        matching_face = None
        for face in registered_faces:
            db_face_encodings = eval(face[4])
            distance = face_recognition.face_distance([db_face_encodings], face_encodings)
            similarity = 1 - distance[0]
            if similarity > max_similarity:
                max_similarity = similarity
                matching_face = face

        if matching_face:
            sg.popup_auto_close(f"認証成功！: {matching_face[1]}", auto_close_duration=2)
        else:
            sg.popup_auto_close("認証失敗", auto_close_duration=2)

    except Exception as e:
        sg.popup_error(f"認証に失敗しました: {str(e)}")

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

            c.execute("SELECT * FROM faces")
            existing_faces = c.fetchall()
            for face in existing_faces:
                if face_encodings == eval(face[4]):
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
