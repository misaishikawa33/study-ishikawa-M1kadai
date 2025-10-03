import numpy as np
import matplotlib.pyplot as plt

# 点の数
N = 100

# θ_i を計算
theta_i = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)

# 元の楕円点列
x = 300 * np.cos(theta_i)
y = 200 * np.sin(theta_i)

# 正規分布ノイズを生成（平均0, 標準偏差3.0）
noise_x = np.random.normal(loc=0.0, scale=3.0, size=N)
noise_y = np.random.normal(loc=0.0, scale=3.0, size=N)

# ノイズを加えたデータ
x_noisy = x + noise_x
y_noisy = y + noise_y

# 描画
plt.figure(figsize=(6, 6))
plt.plot(x, y, 'o', label='Original Points', markersize=3)           # 元の点
plt.plot(x_noisy, y_noisy, 'o', label='Noisy Points', markersize=3)   # ノイズ付き点（×印）
plt.axis('equal')
plt.grid(True)
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Original and Noisy Ellipse Points')
plt.legend()
plt.savefig('ellipse_points_noisy.png', dpi=300, bbox_inches='tight')  # 画像保存（任意）
plt.show()
