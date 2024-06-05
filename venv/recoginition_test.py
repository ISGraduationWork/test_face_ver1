import PySimpleGUI as sg

def browse_image(image_path):
    """
    画像をブラウズしてパスを取得する関数
    """
    file_path = sg.popup_get_file('画像を選択', file_types=(("Image files", "*.jpg;*.jpeg;*.png"),))
    if file_path:
        return file_path
    else:
        return None

def register_face(window, name, image_path):
    """
    顔を登録する関数
    """
    if not name:
        window['STATUS'].update("名前が入力されていません")
        return
    elif not image_path:
        window['STATUS'].update("画像が選択されていません")
        return
    else:
        # ここで顔登録の処理を実装します
        sg.popup_auto_close('登録しました', auto_close_duration=2)
        window['NAME'].update('')  # フォームをクリア
        window['STATUS'].update('')
        window['REGISTER'].update(disabled=True)

def show_image_options():
    """
    画像選択方法を表示する関数
    """
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
    """
    入力をチェックしてボタンの状態を更新する関数
    """
    if name and image_path:
        window['REGISTER'].update(disabled=False)
    else:
        window['REGISTER'].update(disabled=True)

def main():
    # グローバル変数の定義
    image_path = None

    # レイアウト
    layout = [
        [
            sg.Text('名前:'), sg.Input(key='NAME', enable_events=True)
            ],
        [
            sg.Text('', key='STATUS', size=(34, 1)),
            sg.Button('画像を選択', key='BROWSE')
            ],
        [
            sg.Push(),
            sg.Button('新規登録する', key='REGISTER', disabled=True),
            sg.Push(),
            ],
        [
            sg.Text('', key='NOTIFICATION', size=(40, 2), visible=False, text_color='white', background_color='blue')
            ]
    ]

    # ウィンドウの生成
    window = sg.Window('Face Registration', layout)

    # GUI処理
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
                    check_input(window, values['NAME'], image_path)  # 入力をチェックしてボタンの状態を更新
        elif event == 'REGISTER':
            name = values['NAME']
            register_face(window, name, image_path)  # フォームをクリアする処理を追加
        elif event == 'NAME':
            check_input(window, values['NAME'], image_path)  # 入力をチェックしてボタンの状態を更新

    window.close()

if __name__ == "__main__":
    main()
