


# ファイル説明
- `cmd.py`：V２Eでイベントデータを生成するための関数定義・
- `compute_loss`：学習で用いる損失関数の定義
- `const.py`：非常に重要ファイル。多くの定数がここで定義されている。ネットワークやパス、イベントデータの閾値など。ここで定数を変更するだけで、全ての実行ファイルに反映される。
- `const_blender`：blenderでの設定ファイル。位置などを含んだカメラ設定、パスなどの設定を行う。
- `convert_label.py`：ＤＥＭのラベルをカメラ座標系のラベルに変換するための関数
- `custom_data`：データセット周りを扱う関数
- `hazard.py`：安全危険領域を計算するための関数。[論文Error Analysis on Grid-Based Slopeand Aspect Algorithms](https://www.researchgate.net/profile/Qiming-Zhou-2/publication/259591318_Error_Analysis_on_Grid-Based_Slope_and_Aspect_Algorithms/links/0046352cd280b05e93000000/Error-Analysis-on-Grid-Based-Slope-and-Aspect-Algorithms.pdf)を参考にした。
- `network.py`：ネットワークをいろいろ定義
- `view.py`：イベントデータを確認するための関数