import rasterio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_dsm_pointcloud(file_path, sample_step=5, point_size=2, alpha=0.7):
    """DSMファイルを点群として表示"""
    
    # データ読み込み
    with rasterio.open(file_path) as src:
        elevation_data = src.read(1).astype(np.float32)
        elevation_data[elevation_data < 0] = np.nan
    
    # サンプリング
    elevation_sampled = elevation_data[::sample_step, ::sample_step]
    height, width = elevation_sampled.shape
    
    # 座標作成
    x = np.arange(0, width * sample_step, sample_step)
    y = np.arange(0, height * sample_step, sample_step)
    X, Y = np.meshgrid(x, y)
    
    # データをフラット化してNaN除去
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = elevation_sampled.flatten()
    
    valid_mask = ~np.isnan(Z_flat)
    X_points = X_flat[valid_mask]
    Y_points = Y_flat[valid_mask]
    Z_points = Z_flat[valid_mask]
    
    # 3D点群プロット
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(X_points, Y_points, Z_points, 
                        c=Z_points, cmap='terrain',
                        s=point_size, alpha=alpha)
    
    # 軸設定
    ax.set_xlim(0, elevation_data.shape[1])
    ax.set_ylim(0, elevation_data.shape[0])
    ax.set_zlim(np.min(Z_points), np.max(Z_points))
    
    # ラベルとカラーバー
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    ax.set_zlabel('標高 (m)')
    ax.set_title(f'DSM点群表示 (サンプリング: {sample_step})')
    
    fig.colorbar(scatter, shrink=0.6)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"表示点数: {len(Z_points):,}")

# 使用例
dsm_file = 'dsm1.tif'

# 基本表示（5ピクセルごと）
# plot_dsm_pointcloud(dsm_file)

# 高密度表示（2ピクセルごと）
# plot_dsm_pointcloud(dsm_file, sample_step=2, point_size=1, alpha=0.5)

# 低密度表示（10ピクセルごと）
# plot_dsm_pointcloud(dsm_file, sample_step=10, point_size=5, alpha=0.9)

plot_dsm_pointcloud(dsm_file, sample_step=20, point_size=8, alpha=1.0)

# plot_dsm_pointcloud(dsm_file, sample_step=50, point_size=15, alpha=1.0)
