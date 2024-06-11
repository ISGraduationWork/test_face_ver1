import PySimpleGUI as sg
import os
from screenshot import take_screenshot
from database import connect_database, close_database
import face_recognition

# 画像選択方法UI
def show_image_options():
    layout = [
        [sg.Text("画像選択方法を選択してください。")],
        [sg.Button('ファイルから選択', key='FILE', size=(15, 1))],
        [sg.Button('スクリーンショット', key='SCREENSHOT', size=(15, 1))]
    ]
    event, _ = sg.Window('画像の選択方法', layout).read(close=True)

    # 選択されたボタンに応じて結果を返す
    if event == 'FILE':
        return 'file'
    elif event == 'SCREENSHOT':
        return 'screenshot'
    else:
        return None

# 名前と画像入力有無を確認し、登録ボタンの有効/無効を切り替える
def check_input(window, name, image_path):
    if name and image_path:
        window['REGISTER'].update(disabled=False)
    else:
        window['REGISTER'].update(disabled=True)

# 画像選択UIを表示し、ユーザーが選択した画像ファイルのパスを返す
def browse_image(image_path):
    file_path = sg.popup_get_file('画像を選択', file_types=(("Image files", "*.jpg;*.jpeg;*.png"),))
    if file_path:
        return file_path
    else:
        return None
