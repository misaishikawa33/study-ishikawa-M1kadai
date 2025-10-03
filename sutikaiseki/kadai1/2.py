import numpy as np
import matplotlib.pyplot as plt

# 点の数
N = 100

# θ_i を計算
theta_i = -np.pi / 4 + (11 * np.pi) / (12 * N) * np.arange(N)

# 楕円上の点を計算
x = 300 * np.cos(theta_i)
y = 200 * np.sin(theta_i)

# プロット
plt.figure(figsize=(6, 6))# 図のサイズを指定
plt.plot(x, y, 'o', label='Ellipse Points', markersize=3) # 点をプロット,oは点のマーカーを指定（丸）

plt.axis('equal')# アスペクト比を等しくする
plt.grid(True) #補助線
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Points on the Ellipse')
plt.legend()

# 保存（ファイル名は任意：ここでは ellipse_points.png）
plt.savefig('ellipse_points.png', dpi=300, bbox_inches='tight')  # 解像度300dpiで余白カット
plt.show()

