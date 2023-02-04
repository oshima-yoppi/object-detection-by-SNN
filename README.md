
# イベント駆動型コンピューティングによる月面の障害物検出

クレータなど月面は起伏が激しく、安全領域を抽出する必要がある。  
イベント駆動により低消費電力かつ低レーテンシで動作するイベントカメラとスパイキングニュラールネットワークを活用して、安全領域の推定を行う。

# DEMO

"hoge"の魅力が直感的に伝えわるデモ動画や図解を載せる

# Features

"hoge"のセールスポイントや差別化などを説明する

# 必要ライブラリ
anacondaで環境作るのおすすめです。  
ライブラリはおおきくGPU関連(pytorchなど)とそれ以外のライブラリに分かれます。  
*  **GPU関連のライブラリ**：gpuを使用したい場合(機械学習はGPU無いとかなりきついです。ほぼ必須です)は、pytorchのライブラリを使用します。ただ自分が使用しているGPUの種類(RTX30~~など)によってCUDAやpytorchのライブラリのバージョンを指定する必要があります。ですので[金子研究室のサイト](https://www.kkaneko.jp/tools/wsl/wsl_tensorflow2.html)を見て環境構築することを強くお勧めします。
* **GPU 関連以外のライブラリ**：GPU関連以外のライブラリに関してはは`conda_emv.yml`に記載してあります。次のコマンドで環境を作ってください。
```bash
conda create -n 環境名 -f conda_env.yml
```


# フォルダ説明
- `dem`：デム関係。デムを生成したり、ハザードマップを生成するやつ  
-  `easy_task`：簡単なタスク。イベントデータ疑似生成させ、安全かどうかの分類やセマセグなどのタスクができるかをチェック。
-  `v2e-master`：動画からイベントデータを疑似生成させる。
