# 7segment-display-reader
7セグメントディスプレイの数値を認識するプログラムです。<br>
ローカルPC(CPU推論) or Raspberry Pi上での実行を想定しています。<br>
2021/11/16：より軽量なモデルに差し替えました(Raspberry Pi4：4桁の推論で15～20fps程度）<br>
<img src="https://user-images.githubusercontent.com/37477845/120185356-0b224e00-c24d-11eb-9e02-cc53e7f633d3.gif" width="40%">　<img src="https://user-images.githubusercontent.com/37477845/120185427-28571c80-c24d-11eb-9ff0-e4470e6aa266.gif" width="40%">

本リポジトリは以下の内容を含みます。
* サンプルプログラム
* 7セグメント画像識別モデル(TF-Lite)
* 学習データ、および、学習用ノートブック

# Requirements
* OpenCV 3.4.2 or Later
* Tensorflow 2.4.1 or Later
* matplotlib 3.3.2 or Later 

# Usage
実行方法は以下です。<br>
起動後、マウスで4点(左上、右上、右下、左下の順)検出対象をクリックする<br>
```bash
python 7seg-reader.py
```
<br>
実行時には、以下のオプションが指定可能です。

* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：640
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：480
* --crop_width<br>
7セグメント画像1枚の切り抜き横幅<br>
デフォルト：96
* --crop_height<br>
7セグメント画像1枚の切り抜き縦幅<br>
デフォルト：96
* --num_digits<br>
7セグメントディスプレイの桁数<br>
デフォルト：4
* --check_count<br>
7セグメント画像識別時に直近何回の数値をもとに判断するか<br>
指定数値を保持し、最多の数値を識別値とする<br>
デフォルト：5
* --use_binarize<br>
7セグメント画像の識別時に2値化(大津の2値化)を使用するか否か<br>
デフォルト：指定なし
* --use_binarize_inverse<br>
2値化を行う際に値を反転するか否か<br>
デフォルト：指定なし
* --binarize_th<br>
2値化の閾値(0～255)を指定<br>
このオプションを指定した場合、大津の2値化ではなく単純2値化を実施<br>
デフォルト：None

# Training
#### 1.学習データ
以下のデータを混合し、学習データ：検証データ = 3：1 で分割して使用
* [01.dataset](01.dataset)<br>
2種類の7セグメント表示器を撮影したデータセット<br>
「0」～「9」「表示なし」のデータで約42,000枚
* [Kazuhito00/7seg-image-generator](https://github.com/Kazuhito00/7seg-image-generator)<br>
OpenCVの描画関数で疑似的に作成したデータセット<br>
「0」～「9」「-」「表示なし」のデータで48,000枚

#### 2.モデル訓練
「[01-01.train_model.ipynb](01-01.train_model.ipynb)」をJupyter Notebookで開いて上から順に実行してください。<br>
Google Colaboratory上での実行を想定していますが、ローカルPCでも動作出来ると思います。<br>

#### X.モデル構造
「[01-01.train_model.ipynb](01-01.train_model.ipynb)」で用意しているモデルのイメージは以下です。
<img src="https://user-images.githubusercontent.com/37477845/120166466-8f69d680-c237-11eb-978a-7e14628db468.png" width="50%"><br><br>

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
7segment-display-reader is under [Apache v2 license](LICENSE).
