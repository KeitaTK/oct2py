from oct2py import Oct2Py
import numpy as np
import matplotlib.pyplot as plt

# Octaveインスタンスを作成
octave = Oct2Py()

# 微分方程式の定義: dy/dt = 3*y^2 + sin(t)
# octave.eval("f = @(t, y) 3 * y.^2 + sin(t)")
# octave.eval("f = @(t, y) -2*y + 1")
# octave.eval("f = @(t, y) 2*t +y")
# octave.eval("f = @(t, y) sin(t)")
# octave.eval("f = @(t, y) -0.5*y + cos(2*t)")
octave.eval("f = @(t, y) -0.5*y + exp(-0.2*t) .* sin(3*t)")

# 初期条件と時間範囲 (Python側の変数として定義)
t_span_py = [0, 10]
y0_py = 0

# Octaveのode45を使用して微分方程式を解く
# Pythonの変数をOctaveコマンド文字列に埋め込む
# f-string を使用して t_span_py と y0_py の値をコマンドに挿入します。
# t_span_py は Octave のベクトル形式 [t_start, t_end] にします。
command = f"[t, y] = ode45(f, [{t_span_py[0]}, {t_span_py[1]}], {y0_py});"
octave.eval(command)

# 結果をPythonの変数に取得
t = octave.pull('t')
y = octave.pull('y')

# 結果をプロット
plt.figure(figsize=(10, 6))
plt.plot(t, y, 'b-', linewidth=2, label='anser')
plt.xlabel('t')
plt.ylabel('y(t)')
plt.grid(True)
plt.legend()
plt.show()
