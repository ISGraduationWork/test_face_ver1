import sqlite3
import PySimpleGUI as sg

def connect_database():
    """
    データベースに接続し、必要に応じてテーブルを作成します。

    Returns:
        tuple: (sqlite3.Connection, sqlite3.Cursor) データベース接続とカーソルのタプル。
            エラーが発生した場合は (None, None) を返します。
    """
    try:
        # データベースに接続
        conn = sqlite3.connect('faces.db')
        c = conn.cursor()
        # テーブルが存在しない場合は作成
        c.execute('''CREATE TABLE IF NOT EXISTS faces(
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    hash TEXT UNIQUE,
                    Registered_time TEXT,
                    face_encodings BLOB
                    )''')
        return conn, c
    except sqlite3.OperationalError as e:
        # データベースの操作エラーをキャッチ
        sg.popup_error(f"データベース操作エラー: {str(e)}")
        return None, None
    except sqlite3.DatabaseError as e:
        # その他のデータベース関連エラーをキャッチ
        sg.popup_error(f"データベースエラー: {str(e)}")
        return None, None
    except Exception as e:
        # 予期しないエラーをキャッチ
        sg.popup_error(f"予期しないエラー: {str(e)}")
        return None, None

def close_database(conn):
    """
    データベース接続を閉じます。

    Args:
        conn (sqlite3.Connection): データベース接続オブジェクト。
    """
    if conn:
        try:
            conn.close()
        except sqlite3.Error as e:
            # データベースのクローズエラーをキャッチ
            sg.popup_error(f"データベースクローズエラー: {str(e)}")
        except Exception as e:
            # 予期しないエラーをキャッチ
            sg.popup_error(f"予期しないエラー: {str(e)}")
