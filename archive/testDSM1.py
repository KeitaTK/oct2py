import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DSMファイルを読み込み
def read_dsm(file_path):
    with rasterio.open(file_path) as src:
        dsm_data = src.read(1)  # 最初のバンドを読み込み
        dsm_data = dsm_data.astype(np.float32)
        # NoDataや負の値をNaNに変換
        dsm_data[dsm_data < 0] = np.nan
        return dsm_data

# DSMファイルのパス
dsm_path = 'dsm1.tif'

# データ読み込み
elevation_data = read_dsm(dsm_path)

# 座標系を作成（実際のピクセル座標）
height, width = elevation_data.shape
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
X, Y = np.meshgrid(x, y)

print(f"データサイズ: {width} x {height}")
print(f"標高範囲: {np.nanmin(elevation_data):.1f} - {np.nanmax(elevation_data):.1f} m")

# 3Dプロット（CloudCompare風）
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# **CloudCompare風の設定**
# 背景色を暗めに設定
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.xaxis.pane.set_alpha(0.1)
ax.yaxis.pane.set_alpha(0.1)
ax.zaxis.pane.set_alpha(0.1)

# **地形に適したカラーマップとサーフェス表示**
surf = ax.plot_surface(X, Y, elevation_data, 
                      cmap='gist_earth',  # 地形に適したカラーマップ
                      linewidth=0, 
                      antialiased=True,
                      alpha=0.9,
                      shade=True)  # 陰影を有効化

# **実際のデータ範囲で軸を設定**
ax.set_xlim(0, width)
ax.set_ylim(0, height)
ax.set_zlim(np.nanmin(elevation_data), np.nanmax(elevation_data))

# **CloudCompare風の視点角度設定**
ax.view_init(elev=45, azim=45)

# カラーバーを追加
cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('標高 (m)', rotation=270, labelpad=15)

# **軸ラベルとタイトル**
ax.set_xlabel('X座標 (ピクセル)', labelpad=10)
ax.set_ylabel('Y座標 (ピクセル)', labelpad=10)
ax.set_zlabel('標高 (m)', labelpad=10)
ax.set_title('DSM 3Dマップ (CloudCompare風)', pad=20)

# グリッドを薄く表示
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




# import rasterio
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # DSMファイルを読み込み
# def read_dsm(file_path):
#     with rasterio.open(file_path) as src:
#         dsm_data = src.read(1)  # 最初のバンドを読み込み
#         dsm_data = dsm_data.astype(np.float32)
#         # NoDataや負の値をNaNに変換
#         dsm_data[dsm_data < 0] = np.nan
#         return dsm_data

# # DSMファイルのパス
# dsm_path = 'dsm1.tif'

# # データ読み込み
# elevation_data = read_dsm(dsm_path)

# # 座標系を作成
# height, width = elevation_data.shape
# x = np.arange(0, width, 1)
# y = np.arange(0, height, 1)
# X, Y = np.meshgrid(x, y)

# # **データを点群用にフラット化**
# X_flat = X.flatten()
# Y_flat = Y.flatten()
# Z_flat = elevation_data.flatten()

# # **NaNの値を除去**
# valid_mask = ~np.isnan(Z_flat)
# X_points = X_flat[valid_mask]
# Y_points = Y_flat[valid_mask]
# Z_points = Z_flat[valid_mask]

# print(f"データサイズ: {width} x {height}")
# print(f"有効な点数: {len(Z_points)}")
# print(f"標高範囲: {np.min(Z_points):.1f} - {np.max(Z_points):.1f} m")

# # 3D点群プロット
# fig = plt.figure(figsize=(14, 10))
# ax = fig.add_subplot(111, projection='3d')

# # **点群として表示**
# scatter = ax.scatter(X_points, Y_points, Z_points, 
#                     c=Z_points,  # 高さに応じて色分け
#                     cmap='terrain',  # 地形用カラーマップ
#                     s=1,  # 点のサイズ
#                     alpha=0.6)  # 透明度

# # 軸の範囲設定
# ax.set_xlim(0, width)
# ax.set_ylim(0, height)
# ax.set_zlim(np.min(Z_points), np.max(Z_points))

# # カラーバーを追加
# cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
# cbar.set_label('標高 (m)', rotation=270, labelpad=15)

# # ラベルとタイトル
# ax.set_xlabel('X座標 (ピクセル)')
# ax.set_ylabel('Y座標 (ピクセル)')
# ax.set_zlabel('標高 (m)')
# ax.set_title('DSM 点群表示')

# # 視点角度設定
# ax.view_init(elev=30, azim=45)

# plt.tight_layout()
# plt.show()
