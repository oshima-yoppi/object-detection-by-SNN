## Blenderで実行するpythonスクリプト
# システムアクセス用ライブラリのインポート
import sys, os
# blenderライブラリのインポート
import bpy
def get_const_filepath():
    ## エディタ実行時に追加ライブラリを参照するため
    ## ファイル読み込みディレクトリをシステムパスに追加する
    # 自身のファイル名を取得
    script_filename = os.path.basename(__file__)
    # Blenderの読み込みファイルリストからファイルパスを取得
    script_filepath = bpy.data.texts[script_filename].filepath
    # 読み込み元のディレクトリパスを取得
    const_filepath = str(script_filepath).rsplit('\\',1)[0]
    if const_filepath in sys.path:
        pass
    else:
        sys.path += [os.path.dirname(const_filepath)]
    # script_dirpath = os.path.dirname(script_filepath)
    # 読み込み元のディレクトリパスをシステムパスに追加
    #sys.path += [script_dirpath]

print(sys.path)
from const import *
from ..module.const import *

print(IMG_HEIGHT)