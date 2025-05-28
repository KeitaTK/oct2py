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
# 'dsm1.tif'が存在するパスを指定してください。
# 例: dsm_path = 'path/to/your/dsm1.tif'
# この例ではカレントディレクトリにあると仮定します。
dsm_path = 'dsm1.tif'
try:
    elevation_data = read_dsm(dsm_path)
except rasterio.errors.RasterioIOError:
    print(f"エラー: DSMファイル '{dsm_path}' が見つかりません。パスを確認してください。")
    print("ダミーデータで続行します。")
    # ダミーデータを作成して実行できるようにする
    elevation_data = np.random.rand(100, 150) * 50 + 1000
    elevation_data[10:20, 10:20] = 0 # 0の値を含む領域
    elevation_data[30:35, 40:45] = -9999 # NaNに変換される値
    elevation_data = read_dsm_from_array(elevation_data) # ダミーデータをread_dsmと同様の処理

def read_dsm_from_array(array_data): # ダミーデータ用
    dsm_data = array_data.astype(np.float32)
    dsm_data[dsm_data < 0] = np.nan
    return dsm_data

if 'elevation_data' not in locals(): # ファイル読み込み失敗時のフォールバック
    elevation_data = np.zeros((50,50)) # 最小限のデータでエラーを防ぐ

# ===== 相対標高へのシフト（0以外の最小値を基準に）=====
# 0以外の値を抽出して最小値を求める
nonzero_mask = (elevation_data != 0) & (~np.isnan(elevation_data))
if np.any(nonzero_mask): # 0以外の有効な値がある場合のみ処理
    nonzero_min = np.min(elevation_data[nonzero_mask])
    # 0以外の値から最小値を引いてシフト（元が0の場所は0のまま）
    elevation_shifted = elevation_data.copy()
    elevation_shifted[nonzero_mask] -= nonzero_min
else: # 全て0またはNaNの場合
    elevation_shifted = elevation_data.copy()


# データサイズと座標生成
height, width = elevation_shifted.shape
x = np.arange(0, width, 1)
y = np.arange(0, height, 1)
X, Y = np.meshgrid(x, y)

print(f"データサイズ: {width} x {height}")
if np.any(~np.isnan(elevation_shifted)):
    print(f"最小値でシフトした後の標高範囲: {np.nanmin(elevation_shifted):.1f} - {np.nanmax(elevation_shifted):.1f} m")
else:
    print("有効な標高データがありません。")


# ===== Octave による平滑化 =====
oc = Oct2Py()

# NaNを0に置き換えて Octave に渡す
oct_dsm = np.nan_to_num(elevation_shifted, nan=0.0)
oc.push('Z', oct_dsm)

# imageパッケージとガウシアンフィルタ
try:
    oc.eval("pkg load image;") # Octaveコマンドの末尾にセミコロン推奨
    oc.eval("h = fspecial('gaussian', [7, 7], 1.5);")
    oc.eval("Z_smooth = imfilter(Z, h, 'replicate');")
    elevation_smooth = oc.pull('Z_smooth')
except Exception as e:
    print(f"Octaveでの処理中にエラーが発生しました: {e}")
    print("平滑化なしのデータで続行します。")
    elevation_smooth = elevation_shifted.copy()


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

# nanmaxがエラーにならないようにチェック
z_max_plot = np.nanmax(elevation_smooth) if np.any(~np.isnan(elevation_smooth)) else 1.0


surf = ax.plot_surface(X, Y, np.ma.masked_invalid(elevation_smooth),
                       cmap='terrain',
                       linewidth=0,
                       antialiased=True,
                       alpha=0.95,
                       shade=True)

# X軸の表示方向を反転
ax.set_xlim(width, 0) # 変更点: X軸の範囲を (最大値, 0) に設定
ax.set_ylim(0, height) # Y軸は通常通り
ax.set_zlim(0, z_max_plot)

ax.view_init(elev=45, azim=45) # azimの値を調整しても視覚的な回転が得られます

cbar = fig.colorbar(surf, shrink=0.6, aspect=20, pad=0.1)
cbar.set_label('Relative Elevation (m)', rotation=270, labelpad=15)

ax.set_xlabel('X Coordinate (pixels)', labelpad=10)
ax.set_ylabel('Y Coordinate (pixels)', labelpad=10)
ax.set_zlabel('Relative Elevation (m)', labelpad=10)
ax.set_title('DSM 3D Map (Relative Elevation + Octave Smoothing)', pad=20)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ===== 勾配（傾斜）計算 =====
# Octaveに渡すデータが空でないことを確認
if elevation_smooth.size > 0 :
    oc.push('Z_for_gradient', np.nan_to_num(elevation_smooth, nan=0.0)) # NaNを0で埋めて渡す
    try:
        oc.eval('[Gx, Gy] = gradient(Z_for_gradient);') # 既にロード済みのimageパッケージは不要
        Gx_oct, Gy_oct = oc.pull('Gx'), oc.pull('Gy')
        # 元がNaNだった場所の勾配も0やNaNにする（オプション）
        Gx_oct[np.isnan(elevation_smooth)] = np.nan
        Gy_oct[np.isnan(elevation_smooth)] = np.nan
        slope_magnitude = np.sqrt(Gx_oct**2 + Gy_oct**2)
    except Exception as e:
        print(f"Octaveでの勾配計算中にエラー: {e}")
        slope_magnitude = np.zeros_like(elevation_smooth) # エラー時はゼロ配列
else:
    Gx_oct = np.array([])
    Gy_oct = np.array([])
    slope_magnitude = np.array([])


# ===== 傾斜マップ表示 =====
if slope_magnitude.size > 0 and np.any(~np.isnan(slope_magnitude)):
    plt.figure(figsize=(10, 6))
    plt.imshow(slope_magnitude, cmap='viridis')
    
    # X軸の表示方向を反転
    plt.gca().invert_xaxis() # 変更点: 現在のAxesのX軸を反転
    
    plt.colorbar(label='Slope (Gradient Magnitude)')
    plt.title('Slope Map (Relative Elevation + Octave Smoothing)')
    plt.xlabel('X Coordinate (pixels)')
    plt.ylabel('Y Coordinate (pixels)')
    plt.tight_layout()
    plt.show()
else:
    print("傾斜マップを表示するための有効なデータがありません。")