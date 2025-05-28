from oct2py import octave
import numpy as np

# NumPy配列を作成
x = np.array([[1, 2], [3, 4]], dtype=float)

# Octaveの魔方陣関数を使用
magic_square = octave.magic(3)
print(magic_square)

# NumPy配列をOctaveに渡して計算
result = octave.sum(x)  # 行列の各列の合計
print(result)

