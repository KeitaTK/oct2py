import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from oct2py import octave
import rasterio
import numpy as np
from scipy.io import savemat

def dsm_octave_gnuplot(dsm_file, sample_step=60, max_points=2000):
    """gnuplotツールキットを使用したDSM表示"""
    
    # DSM処理（前と同じ）
    with rasterio.open(dsm_file) as src:
        elevation_data = src.read(1).astype(np.float32)
        elevation_data[elevation_data < 0] = np.nan
    
    elevation_sampled = elevation_data[::sample_step, ::sample_step]
    height, width = elevation_sampled.shape
    
    x = np.arange(0, width * sample_step, sample_step)
    y = np.arange(0, height * sample_step, sample_step)
    X, Y = np.meshgrid(x, y)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = elevation_sampled.flatten()
    
    valid_mask = ~np.isnan(Z_flat)
    X_points = X_flat[valid_mask]
    Y_points = Y_flat[valid_mask]
    Z_points = Z_flat[valid_mask]
    
    if len(Z_points) > max_points:
        indices = np.random.choice(len(Z_points), max_points, replace=False)
        X_points = X_points[indices]
        Y_points = Y_points[indices]
        Z_points = Z_points[indices]
    
    print(f"点数: {len(Z_points)}")
    
    savemat('temp_pointcloud.mat', {
        'x': X_points, 'y': Y_points, 'z': Z_points
    })
    
    try:
        # **gnuplotツールキットを明示的に設定**
        octave.eval("graphics_toolkit('gnuplot');")
        octave.eval("disp('Using gnuplot toolkit');")
        
        octave.eval("load('temp_pointcloud.mat');")
        octave.eval("""
        figure;
        scatter3(x, y, z, 5, z);
        xlabel('X coordinate');
        ylabel('Y coordinate'); 
        zlabel('Elevation');
        title('DSM Point Cloud (gnuplot)');
        colorbar;
        view(45, 30);
        grid on;
        """)
        
        print("gnuplotでプロット成功")
        
    except Exception as e:
        print(f"エラー: {e}")

# 実行
dsm_octave_gnuplot('dsm1.tif')

