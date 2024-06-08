import PySimpleGUI as sg
from screenshot import take_screenshot

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

def browse_image():
    file_path = sg.popup_get_file('画像を選択', file_types=(("Image files", "*.jpg;*.jpeg;*.png"),))
    if file_path:
        return file_path
    else:
        return None
