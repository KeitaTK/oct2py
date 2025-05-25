import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from oct2py import Oct2Py

# DSM読み込み関数
def read_dsm(file_path):
    with rasterio.open(file_path) as src:
        dsm_data = src.read(1).astype(np.float32)
        dsm_data[dsm_data < 0] = np.nan
        return dsm_data

# DSMファイルパス
dsm_path = 'dsm1.tif'
elevation_data = read_dsm(dsm_path)

# DSMデータのサイズ
height, width = elevation_data.shape
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
X, Y = np.meshgrid(x, y)

print(f"データサイズ: {width} x {height}")
print(f"標高範囲（元データ）: {np.nanmin(elevation_data):.1f} - {np.nanmax(elevation_data):.1f} m")

# # ===== Octaveによるメッシュ平滑化 =====
# oc = Oct2Py()

# # NaNを0で埋める（OctaveがNaNに弱いため）
# oct_dsm = np.nan_to_num(elevation_data, nan=0.0)

# # Octaveにデータを渡してガウシアンフィルタでスムージング
# oc.push('Z', oct_dsm)
# oc.eval("h = fspecial('gaussian', [7, 7], 1.5);")  # 7x7カーネル, σ=1.5
# oc.eval("Z_smooth = imfilter(Z, h, 'replicate');")
# elevation_smooth = oc.pull('Z_smooth')

# # NaNを戻す（元のDSMでNaNだった場所）
# elevation_smooth[np.isnan(elevation_data)] = np.nan

# ===== Octaveによるメッシュ平滑化 =====
oc = Oct2Py()

# NaNを0で埋める
oct_dsm = np.nan_to_num(elevation_data, nan=0.0)

oc.push('Z', oct_dsm)

# Imageパッケージを読み込む
oc.eval("pkg load image")  # ← 追加

# ガウシアンフィルタを生成して平滑化
oc.eval("h = fspecial('gaussian', [7, 7], 1.5);")
oc.eval("Z_smooth = imfilter(Z, h, 'replicate');")
elevation_smooth = oc.pull('Z_smooth')


print(f"標高範囲（平滑化後）: {np.nanmin(elevation_smooth):.1f} - {np.nanmax(elevation_smooth):.1f} m")

# ===== 1. 平滑化済みDSMの3Dプロット（CloudCompare風） =====
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

surf = ax.plot_surface(X, Y, elevation_smooth,
                       cmap='gist_earth',
                       linewidth=0,
                       antialiased=True,
                       alpha=0.9,
                       shade=True)

ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(np.nanmin(elevation_smooth), np.nanmax(elevation_smooth))

ax.view_init(elev=45, azim=45)

cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('標高 (m)', rotation=270, labelpad=15)

ax.set_xlabel('X座標 (ピクセル)', labelpad=10)
ax.set_ylabel('Y座標 (ピクセル)', labelpad=10)
ax.set_zlabel('標高 (m)', labelpad=10)
ax.set_title('DSM 3Dマップ（平滑化 + CloudCompare風）', pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== 2. Octaveで勾配（傾斜）計算も追加 =====
oc.push('Z', elevation_smooth)  # 平滑化後の標高データで勾配を計算
oc.eval('[Gx, Gy] = gradient(Z);')
Gx, Gy = oc.pull('Gx'), oc.pull('Gy')
slope_magnitude = np.sqrt(Gx**2 + Gy**2)

plt.figure(figsize=(10, 6))
plt.imshow(slope_magnitude, cmap='viridis')
plt.colorbar(label='傾斜（勾配の大きさ）')
plt.title('DSMの傾斜マップ（Octaveによる平滑化＋勾配計算）')
plt.xlabel('X座標（ピクセル）')
plt.ylabel('Y座標（ピクセル）')
plt.tight_layout()
plt.show()
