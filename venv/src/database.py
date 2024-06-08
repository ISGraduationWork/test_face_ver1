import sqlite3
import PySimpleGUI as sg

def connect_database():
    try:
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS faces
                    (id INTEGER PRIMARY KEY, name TEXT, hash TEXT UNIQUE, Registered_time TEXT, face_encodings TEXT)''')
        return conn, c
    except sqlite3.Error as e:
        sg.popup_error(f"データベースエラー: {str(e)}")
        return None, None

def close_database(conn):
    if conn:
        conn.close()
