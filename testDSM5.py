import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from oct2py import Oct2Py

# ===== DSM読み込み関数 =====
def read_dsm(file_path):
    with rasterio.open(file_path) as src:
        dsm_data = src.read(1).astype(np.float32)
        dsm_data[dsm_data < 0] = np.nan  # 負の値はNaNに変換
        return dsm_data

# ===== ファイル読み込み =====
dsm_path = 'dsm1.tif'
elevation_data = read_dsm(dsm_path)

# ===== 相対標高へのシフト（0以外の最小値を基準に）=====
# 0以外の値を抽出して最小値を求める
nonzero_mask = (elevation_data != 0) & (~np.isnan(elevation_data))
nonzero_min = np.min(elevation_data[nonzero_mask])

# 0以外の値から最小値を引いてシフト（元が0の場所は0のまま）
elevation_shifted = elevation_data.copy()
elevation_shifted[nonzero_mask] -= nonzero_min

# データサイズと座標生成
height, width = elevation_shifted.shape
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
X, Y = np.meshgrid(x, y)

print(f"データサイズ: {width} x {height}")
print(f"最小値でシフトした後の標高範囲: {np.nanmin(elevation_shifted):.1f} - {np.nanmax(elevation_shifted):.1f} m")

# ===== Octave による平滑化 =====
oc = Oct2Py()

# NaNを0に置き換えて Octave に渡す
oct_dsm = np.nan_to_num(elevation_shifted, nan=0.0)
oc.push('Z', oct_dsm)

# imageパッケージとガウシアンフィルタ
oc.eval("pkg load image")
oc.eval("h = fspecial('gaussian', [7, 7], 1.5);")
oc.eval("Z_smooth = imfilter(Z, h, 'replicate');")
elevation_smooth = oc.pull('Z_smooth')

# 元がNaNだった場所は再びNaNに
elevation_smooth[np.isnan(elevation_data)] = np.nan

# ===== 3Dプロット（相対標高）=====
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 背景スタイル
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

surf = ax.plot_surface(X, Y, np.ma.masked_invalid(elevation_smooth),
                       cmap='terrain',
                       linewidth=0,
                       antialiased=True,
                       alpha=0.95,
                       shade=True)

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(0, np.nanmax(elevation_smooth))

ax.view_init(elev=45, azim=45)

cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('相対標高 (m)', rotation=270, labelpad=15)

ax.set_xlabel('X座標 (ピクセル)', labelpad=10)
ax.set_ylabel('Y座標 (ピクセル)', labelpad=10)
ax.set_zlabel('相対標高 (m)', labelpad=10)
ax.set_title('DSM 3Dマップ（相対標高 + Octave平滑化）', pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== 勾配（傾斜）計算 =====
oc.push('Z', elevation_smooth)
oc.eval('[Gx, Gy] = gradient(Z);')
Gx, Gy = oc.pull('Gx'), oc.pull('Gy')
slope_magnitude = np.sqrt(Gx**2 + Gy**2)

# ===== 傾斜マップ表示 =====
plt.figure(figsize=(10, 6))
plt.imshow(slope_magnitude, cmap='viridis')
plt.colorbar(label='傾斜（勾配の大きさ）')
plt.title('傾斜マップ（相対標高 + Octave平滑化）')
plt.xlabel('X座標（ピクセル）')
plt.ylabel('Y座標（ピクセル）')
plt.tight_layout()
plt.show()
