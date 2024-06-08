import PySimpleGUI as sg
from database import connect_database, close_database
from face_recognition_utils import register_face, authenticate_face
from gui import show_image_options, check_input, browse_image
from screenshot import take_screenshot

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
                image_path = browse_image()
                if image_path:
                    window['STATUS'].update(image_path)
                    check_input(window, values['NAME'], image_path)
            elif option == 'screenshot':
                if values['NAME']:
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
