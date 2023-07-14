import numpy as np
import matplotlib.pyplot as plt


# lblファイルの下記の記述を参照
#   SAMPLE_TYPE = MSB_INTEGER  ... Big endian符号付き整数
#   SAMPLE_BITS = 16           ... 1画素あたり16bits
n = int(input())
data = np.fromfile(f"origin/{n}.img", dtype=">i2")

# lblファイルの以下の記述より配列の形を変更
#   BANDS = 1
#   LINES = 10800
#   LINE_SAMPLES = 10800
BANDS = 1
LINES = 12288
LINE_SAMPLES = 12288
image = data.reshape(BANDS, LINE_SAMPLES, LINES)

# lblファイルの以下の記述より、データ加工なしにそのまま使う
#   OFFSET = 0.000000
#   SCALING_FACTOR = 1.000000

# 濃淡がはっきりしないため平均値周辺を使って濃淡を出す
AVERAGE = -1882.145583
delta = np.abs(AVERAGE) * 0.5
# データを表示(最初のバンドを表示)
plt.imshow(image[0], cmap="gray", vmin=AVERAGE - delta, vmax=AVERAGE + delta)
plt.show()
