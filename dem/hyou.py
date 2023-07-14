import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 表のデータを作成
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# 表の行と列のラベル
row_labels = ["Row 1", "Row 2", "Row 3"]
col_labels = ["Column 1", "Column 2", "Column 3"]

# プロット領域を作成
fig, ax = plt.subplots()

# 表を描画
ax.axis("off")
table = ax.table(
    cellText=data, rowLabels=row_labels, colLabels=col_labels, loc="center"
)

# セルのスタイルを設定
# table.auto_set_font_size(False)
# table.set_fontsize(14)
# table.scale(1.2, 1.2)

# プロットを表示
plt.show()
